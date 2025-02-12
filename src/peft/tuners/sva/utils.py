# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from itertools import cycle
from typing import Dict, Iterable, Optional, Union

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D
from transformers.tokenization_utils_base import BatchEncoding

from peft.tuners.tuners_utils import _find_minimal_target_modules, check_target_module_exists
from peft.utils.constants import MIN_TARGET_MODULES_FOR_OPTIMIZATION
from peft.utils.incremental_pca import IncrementalPCA

from .config import SvaConfig
from .layer import SvaLayer


class _Hook:
    """
    A base class for hooks that prepares layer inputs.
    """

    def __init__(
        self,
        name: str,
        prepare_layer_inputs_fn: Optional[callable] = None,
        gather_distributed_inputs: bool = True,
    ):
        self.name = name
        self.gather_distributed_inputs = gather_distributed_inputs
        if prepare_layer_inputs_fn is None:
            self._prepare_layer_inputs_fn = self._prepare_layer_inputs_fn_default
        else:
            self._prepare_layer_inputs_fn = prepare_layer_inputs_fn
        self.model_input = None

    @staticmethod
    def _prepare_layer_inputs_fn_default(layer_input, model_input, layer_name) -> torch.Tensor:
        if isinstance(layer_input, torch.Tensor):
            pass
        elif isinstance(layer_input, (tuple, list)):
            layer_input = layer_input[0]
        else:
            raise ValueError(
                f"unsupported input type {type(layer_input)} for prepare_layer_inputs_fn in layer {layer_name}, "
                "please provide a custom prepare_layer_inputs_fn"
            )
        # if the input has more than 2 dimensions, we flatten all but the last dimension
        if layer_input.ndim > 2:
            layer_input = layer_input.view(-1, layer_input.size(-1))
        return layer_input

    @torch.no_grad()
    def prepare_layer_inputs(self, layer_input):
        return self._prepare_layer_inputs_fn(layer_input, self.model_input, self.name)

    def gather_layer_inputs(self, layer_input):
        if dist.is_initialized() and self.gather_distributed_inputs:
            world_size = dist.get_world_size()

            # First gather sizes from all processes more efficiently
            local_size = torch.tensor([layer_input.shape[0]], device=layer_input.device)
            all_sizes = torch.empty(world_size, dtype=local_size.dtype, device=layer_input.device)
            dist.all_gather_into_tensor(all_sizes, local_size)
            all_sizes = all_sizes.tolist()

            # Find maximum size and pad tensors
            padded_input = layer_input.new_zeros((max(all_sizes), *layer_input.shape[1:]))
            padded_input[: layer_input.shape[0]] = layer_input

            # Gather padded tensors
            gathered_inputs = [torch.zeros_like(padded_input) for _ in range(world_size)]
            dist.all_gather(gathered_inputs, padded_input.contiguous())

            # Remove padding for each gathered tensor
            gathered_inputs = [tensor[:size] for tensor, size in zip(gathered_inputs, all_sizes)]

            # Concatenate along batch dimension
            return torch.cat(gathered_inputs, dim=0)
        return layer_input


class SVDHook(_Hook):
    """
    A forward hook for calculating incremental SVD on layer inputs. The hook is designed to be registered to a PyTorch
    module using the `register_forward_hook` method.

    This hook performs a step of incremental Singular Value Decomposition (SVD) on the inputs of a specified layer
    during the forward pass of a neural network. The hook also tracks convergence of the computed components using
    cosine similarity between the current and previous components.

    Args:
        name (str): Name of the layer to which this hook is attached.
        n_components (int): Number of principal components to compute.
        sim_thresh (Union[float, torch.Tensor]): Similarity threshold for convergence.
        prepare_layer_inputs_fn (Optional[callable]): Function to prepare layer inputs for SVD.
    """

    def __init__(
        self,
        n_components: int,
        sim_thresh: Union[float, torch.Tensor],
        is_backward_hook: bool = False,
        **base_class_kwargs,
    ):
        super().__init__(**base_class_kwargs)
        self.n_components = n_components
        self.sim_thresh = sim_thresh
        self.is_backward_hook = is_backward_hook
        if isinstance(sim_thresh, torch.Tensor) and len(sim_thresh.shape) > 0:
            check1 = sim_thresh.size(0) == n_components or sim_thresh.size(0) == 1
            check2 = len(sim_thresh.shape) == 1
            if not (check1 and check2):
                raise ValueError(
                    "if sim_thresh is a tensor with more than 0 dimensions it must have shape (n_components,) or (1,)"
                )
        self.svd = IncrementalPCA(
            n_components=n_components,
            copy=True,
            lowrank=True,
            lowrank_seed=42,
        )
        self.model_input = None
        self.converged = torch.zeros((n_components,), dtype=torch.bool)

    @torch.no_grad()
    def __call__(self, model, input, output):
        previous_components = None
        if hasattr(self.svd, "components_"):
            previous_components = self.svd.components_.clone().detach()
        if self.is_backward_hook:
            states = self.prepare_layer_inputs(output)
        else:
            states = self.prepare_layer_inputs(input)
        states = self.gather_layer_inputs(states)
        # check if batch sizes is more than the number of components
        if states.size(0) < self.n_components:
            print(f"skipping SVD for {self.name} because there are less than {self.n_components} examples")
            return
        self.svd.partial_fit(states.to(torch.float32))
        # add if statement to check if we are in the first step where previous_components is None
        if previous_components is None:
            return
        components = self.svd.components_
        if len(components.shape) == 1:
            components = components.reshape(1, -1)
            previous_components = previous_components.reshape(1, -1)
        # consider as converged if enough components have converged via cossim
        sim = torch.nn.functional.cosine_similarity(components, previous_components)
        self.converged = sim >= self.sim_thresh


# This is used to determine if inputs of two different layers are equal. For such cases, SVD
# needs to be done for only for one of the equal inputs.
class HashHook(_Hook):
    """
    A forward hook for hashing layer inputs. The hook is designed to be registered to a PyTorch module using the
    `register_forward_hook` method.

    This hook hashes the inputs of a specified layer during the forward pass of a neural network and stores the hash
    values for later analysis or comparison.

    Args:
        name (str): Name of the layer to which this hook is attached. hashed_inputs (list): List of hashed inputs.
        prepare_layer_inputs_fn (Optional[callable]): Function to prepare layer inputs for hashing.
    """

    def __init__(self, **base_class_kwargs):
        super().__init__(**base_class_kwargs)
        self.hashed_inputs = []

    @staticmethod
    def hash_fn(tensor):
        return hash(tuple(tensor.view(-1).tolist()))

    @torch.no_grad()
    def __call__(self, model, input, output):
        x = self.prepare_layer_inputs(input)
        x = self.gather_layer_inputs(x)
        self.hashed_inputs.append(self.hash_fn(x.cpu()))


class ForceGradientFunction(torch.autograd.Function):
    """Forces gradient computation even when parameters have requires_grad=False"""
    
    @staticmethod
    def forward(ctx, inputs):
        inputs["input_ids"] = inputs["input_ids"].view_as(inputs["input_ids"])
        return inputs
    
    @staticmethod
    def backward(ctx, grad_output):
        return


def find_equal_values(dictionary: dict) -> dict:
    """
    Find keys in a dictionary that have the same value.

    This function takes a dictionary and returns a new dictionary containing keys that have the same value. The keys in
    the output dictionary are the values from the input dictionary, and the values are lists of keys that share the
    same value.
    """
    value_dict = defaultdict(list)
    for k, v in dictionary.items():
        value_dict[v].append(k)
    return {k: v for k, v in value_dict.items() if len(v) > 1}


def get_device_with_meta_params(model: torch.nn.Module) -> torch.device:
    """
    Get the device of the model's parameters. Useful if some parameters are on meta device.
    """
    devices = list({p.device for p in model.parameters() if p.device.type != "meta"})
    if len(devices) > 1:
        warnings.warn(f"Could not determine device, model has multiple devices: {devices}")
        return
    return devices[0]


def move_inputs_to_device(inputs, device: Union[str, torch.device]):
    """
    Move the inputs to the specified device. Adapted from hf.Trainer.
    """
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, Mapping):
        return type(inputs)({k: move_inputs_to_device(v, device) for k, v in inputs.items()})
    elif isinstance(inputs, (tuple, list)):
        return type(inputs)(move_inputs_to_device(v, device) for v in inputs)
    else:
        warnings.warn(f"input of type {type(inputs)} could not be moved to the correct device")
        return inputs


def prepare_model_inputs_fn_language_modeling(model_input, peft_config: SvaConfig):
    """
    Get the indices of the items that should be used for SVD.

    Attributes:
        model_input (dict): The model inputs.
        peft_config (SvaConfig): The configuration for the LoRA layers.
    """
    if not isinstance(model_input, (dict, BatchEncoding)):
        raise ValueError("When using `prepare_model_inputs_fn_language_modeling` inputs must be a dictionary")
    mask = model_input.get("attention_mask", torch.ones_like(model_input["input_ids"])).bool()
    if peft_config.use_label_mask and hasattr(model_input, "labels"):
        mask = torch.logical_and(mask, model_input["labels"] != peft_config.label_mask_value)
    return mask.nonzero()


def prepare_layer_inputs_fn_language_modeling(layer_input, model_input, layer_name) -> torch.Tensor:
    """
    if not all items in the input should be used for SVD, this function can be used to get the indices of the items
    that should be used.

    Attributes:
        layer_input (torch.Tensor): The layer inputs.
        model_input (torch.Tensor):
            The model inputs or if `prepare_model_inputs_fn` is not None the output of this function.
        layer_name (str): The name of the layer.

    Returns:
        torch.Tensor: The input to the SVD.
    """
    # if layer inputs are not a tensor, we simply get the first item
    if isinstance(layer_input, torch.Tensor):
        pass
    elif isinstance(layer_input, (tuple, list)):
        layer_input = layer_input[0]
    else:
        raise ValueError(
            f"unsupported input type {type(layer_input)} for prepare_layer_inputs_fn in layer {layer_name}, "
            "please provide a custom prepare_layer_inputs_fn"
        )
    # in this case model_input is the output of `prepare_model_inputs_fn_language_modeling`
    return layer_input[model_input.T.unbind()]


def forward_fn_language_modeling(model, inputs, compute_loss=True):
    if not compute_loss:
        inputs.pop("labels", None)
    outputs = model(**inputs)
    if compute_loss:
        return outputs.loss
    return


def _get_sva_state_dict(
    model: torch.nn.Module,
    dataloader: Iterable,
    peft_config: Optional[SvaConfig],
    target_module_check_fn: callable,
    forward_fn: Optional[callable],
    prepare_model_inputs_fn: Optional[callable],
    prepare_layer_inputs_fn: Union[callable, Dict[str, callable], None],
    gather_distributed_inputs: bool,
    show_progress_bar: bool,
) -> dict:

    def _check_convergence(model, hook, handle, convergence_dict, name, backward_hook):
        converged = torch.all(hook.converged)
        # if a layer has switched from not converged to converged in the current step
        if (not convergence_dict[name][backward_hook]) and converged and handle:
            handle.remove()
            handle = None
            convergence_dict[name][backward_hook] = True
        # if a layer has switched from converged to not converged in the current step
        elif convergence_dict[name][backward_hook] and not converged:
            module = model.get_submodule(name)
            if backward_hook:
                handle = module.register_full_backward_hook(hook)
            else:
                handle = module.register_forward_hook(hook)
            convergence_dict[name][backward_hook] = False
        return convergence_dict, handle

    # dataloader is not empty
    if len(dataloader) == 0:
        raise ValueError("dataloader is empty")

    # check if dist is initialized
    if dist.is_initialized() and gather_distributed_inputs:
        warnings.warn(
            "torch.distributed is initialized and `gather_distributed_inputs` is True, "
            "therefore SVA initialization will gather tensors from all ranks. "
            "Ensure the model does not receive the same inputs on different ranks."
        )

    training = model.training
    device = get_device_with_meta_params(model)
    model.eval()

    # get model inputs
    inputs = next(iter(dataloader))
    if device is not None:
        inputs = move_inputs_to_device(inputs, device)
    if prepare_model_inputs_fn is not None:
        model_inputs_for_hooks = prepare_model_inputs_fn(inputs, peft_config)
    else:
        model_inputs_for_hooks = deepcopy(inputs)

    hooks = {}
    target_layers = []
    for name, module in model.named_modules():
        if not target_module_check_fn(name, module):
            continue
        if isinstance(prepare_layer_inputs_fn, Mapping):
            fn = prepare_layer_inputs_fn.pop(name, None)
        else:
            fn = prepare_layer_inputs_fn
        hook = HashHook(name=name, prepare_layer_inputs_fn=fn, gather_distributed_inputs=gather_distributed_inputs)
        hook.model_input = model_inputs_for_hooks
        handle = module.register_forward_hook(hook)
        hooks[name] = (hook, handle)
        target_layers.append(name)

    if isinstance(prepare_layer_inputs_fn, Mapping) and len(prepare_layer_inputs_fn) > 0:
        raise ValueError(
            "prepare_layer_inputs_fn is a mapping but the following module names were not found in the model: "
            f"{prepare_layer_inputs_fn.keys()}"
        )

    # forward for one batch to check which layer inputs are equal to avoid unneeded svd calculations
    forward_fn(model, inputs)
    hash_dict = {k: h[0].hashed_inputs[0] for k, h in hooks.items()}
    # equal input maps groups layers which receive the same input. One layer is defined as the key and receives an svd
    # hook. For the remaining layers the svd results can be skipped.
    equal_inputs = list(find_equal_values(hash_dict).values())
    equal_inputs_map = {vv: v[0] for v in equal_inputs for vv in v[1:]}

    # initialize svd hooks
    for name in list(hooks.keys()):
        hook, handle = hooks.pop(name)
        handle.remove()
        if name in equal_inputs_map:
            continue
        hook_forward = SVDHook(
            n_components=peft_config.r,
            sim_thresh=peft_config.tau,
            name=name,
            prepare_layer_inputs_fn=hook._prepare_layer_inputs_fn,
            gather_distributed_inputs=gather_distributed_inputs,
            is_backward_hook=False,
        )
        hook_backward = SVDHook(
            n_components=peft_config.r,
            sim_thresh=peft_config.tau,
            name=name,
            prepare_layer_inputs_fn=hook._prepare_layer_inputs_fn,
            gather_distributed_inputs=gather_distributed_inputs,
            is_backward_hook=True,
        )
        module = model.get_submodule(name)
        handle_forward = module.register_forward_hook(hook_forward)
        handle_backward = module.register_full_backward_hook(hook_backward)
        hooks[name] = ((hook_forward, handle_forward), (hook_backward, handle_backward))
    layer_hook_map = {**dict(zip(hooks.keys(), hooks.keys())), **equal_inputs_map}

    # TEMP HACK
    first_module = [m for n,m in model.named_modules() if n.endswith("embed_tokens")][0]
    grad_hook_handle = first_module.register_forward_hook(lambda module, input, output: output.requires_grad_(True))
    # start svd calculation
    if show_progress_bar and (not dist.is_initialized() or dist.get_rank() == 0):
        pbar = tqdm(iter(cycle(dataloader)), position=0, leave=False)
        use_tqdm = True
    else:
        pbar = iter(cycle(dataloader))
        use_tqdm = False
    convergence_dict = {k: [False, False] for k in hooks.keys()}
    all_backward_converged = False
    for inputs in pbar:
        if device is not None:
            inputs = move_inputs_to_device(inputs, device)
        if prepare_model_inputs_fn is not None:
            model_inputs_for_hooks = prepare_model_inputs_fn(inputs, peft_config)
        else:
            model_inputs_for_hooks = deepcopy(inputs)

        for name in list(hooks.keys()):
            hook_forward, handle_forward = hooks[name][0]
            hook_backward, handle_backward = hooks[name][1]
            convergence_dict, handle_forward = _check_convergence(model, hook_forward, handle_forward, convergence_dict, name, False)
            convergence_dict, handle_backward = _check_convergence(model, hook_backward, handle_backward, convergence_dict, name, True)
            hook_forward.model_input = model_inputs_for_hooks
            hook_backward.model_input = model_inputs_for_hooks
            hooks[name] = ((hook_forward, handle_forward), (hook_backward, handle_backward))

        f, b = zip(*convergence_dict.values())
        all_forward_converged = all(f)
        all_backward_converged = all(b)

        if use_tqdm:
            layer_converged = [all(x) for x in convergence_dict.values()] + [
                all(convergence_dict[v]) for v in equal_inputs_map.values()
            ]
            pbar.set_description(f"{sum(layer_converged)}/{len(layer_converged)} layers have converged")

        if all_forward_converged and all_backward_converged:
            break
        
        if all_backward_converged:
            ctx = torch.no_grad()
            grad_hook_handle.remove()
        else:
            ctx = nullcontext()

        with ctx:
            loss = forward_fn(model, inputs, compute_loss=not all_backward_converged)
        if not all_backward_converged:
            loss.backward()
            model.zero_grad()

        # in case some hooks have to skip the svd calculation because the number of tokens is less than the number of
        # components
        if not all(hasattr(h[0].svd, "components_") for hh in hooks.values() for h in hh):
            continue

    # check all custom hooks have been removed
    for method in ["_forward_hooks", "_backward_hooks"]:
        remaining_hooks = {n for n, m in model.named_modules() for v in getattr(m, method).values() if isinstance(v, _Hook)}
        if len(remaining_hooks) > 0:
            raise ValueError(
                f"Found active hooks added by SVA that weren't properly removed: {remaining_hooks}. "
                "Please report this issue at https://github.com/huggingface/peft/issues"
            )

    sva_state_dict = {}
    for name in target_layers:
        hook_f, _ = hooks[layer_hook_map[name]][0]
        hook_b, _ = hooks[layer_hook_map[name]][1]
        u_A = hook_f.svd.components_
        u_B = hook_b.svd.components_
        sva_state_dict[f"{name}.sva_A.default"] = u_A
        sva_state_dict[f"{name}.sva_B.default"] = u_B.T

    # restore model state
    model.train(training)

    # move tensors to device
    if device is not None:
        sva_state_dict = {k: v.to(device) for k, v in sva_state_dict.items()}

    return sva_state_dict


def _load_sva_state_dict(
    model: torch.nn.Module,
    sva_state_dict: dict,
    adapter_name: str,
):
    peft_config = model.peft_config[adapter_name]
    update_layer_kwargs = {
        "adapter_name": adapter_name,
        "r": peft_config.r,
        "sva_dropout": peft_config.sva_dropout,
    }
    missing_sva_inits = []
    new_target_modules = []
    other_module_names = []
    for name, module in model.named_modules():
        name_in_base_model = name.replace("base_model.model.", "")
        if not isinstance(module, SvaLayer):
            other_module_names.append(name_in_base_model)
            continue
        sva_A = sva_state_dict.pop(f"{name}.sva_A.default", None)
        sva_B = sva_state_dict.pop(f"{name}.sva_B.default", None)
        if isinstance(sva_A, torch.Tensor) and isinstance(sva_B, torch.Tensor):
            module.update_layer(sva_A=sva_A, sva_B=sva_B, **update_layer_kwargs)
            new_target_modules.append(name_in_base_model)
        else:
            module = module.get_base_layer()
            missing_sva_inits.append(name_in_base_model)

    # update target modules if some lora layers have been removed due to their SVA rank being 0
    if len(new_target_modules) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION:
        new_target_modules = _find_minimal_target_modules(new_target_modules, other_module_names + missing_sva_inits)
    model.peft_config[adapter_name].target_modules = new_target_modules

    if len(missing_sva_inits) > 0:
        warnings.warn(
            "the following adapter layers were converted back to torch.nn.Linear because they "
            f"were not found in the sva state_dict: {missing_sva_inits}"
        )


def get_sva_state_dict(
    model: torch.nn.Module,
    dataloader: Iterable,
    peft_config: Optional[SvaConfig] = None,
    forward_fn: Optional[callable] = forward_fn_language_modeling,
    prepare_model_inputs_fn: Optional[callable] = prepare_model_inputs_fn_language_modeling,
    prepare_layer_inputs_fn: Union[callable, Dict[str, callable], None] = prepare_layer_inputs_fn_language_modeling,
    adapter_name: str = "default",
    gather_distributed_inputs: bool = True,
    show_progress_bar: bool = True,
) -> dict:
    """
    Compute the SVD for each layer in the model.

    This function computes the Singular Value Decomposition (SVD) for each layer in the model. It uses the incremental
    PCA method to compute the SVD components. The function also checks for convergence of the computed components using
    cosine similarity. The rank distribution for each layer is determined based on the explained variance ratio.

    Args:
        model (torch.nn.Module): The model to compute the SVD for. Does not need to be a PeftModel.
        dataloader (Iterable): The dataloader to use for the forward pass.
        peft_config (Optional[SvaConfig]):
            The configuration for the LoRA layers. Only required if `model` is not a PeftModel.
        forward_fn (callable):
            The forward function to use for the forward pass. Takes two arguments: `model` and `inputs`. Default
            behavior is `return model(**inputs)`
        prepare_model_inputs_fn (Optional[callable]):
            This function receives the model inputs and the peft_config and passes the output to
            `prepare_layer_inputs_fn`. Can be used to modify the input to the SVD computation based on the original
            model inputs. For example for language modeling the attention mask is used to determine which indices are
            padding tokens and should not be used for SVD. Any function defined here expects two arguments:
            `model_input` and `peft_config`. `peft.tuners.sva.utils.prepare_model_inputs_fn_language_modeling` is used
            by default.
        prepare_layer_inputs_fn (Union[callable, Dict[str, callable], None]):
            This function receives the layer inputs, the model inputs (potentially modified by
            `prepare_model_inputs_fn`) and the name of the layer and returns the inputs that should be used for SVD for
            that particular layer. Any custom function defined here expects three arguments: `layer_input`,
            `model_input`, and `layer_name` and should return a 2d tensor. The default logic can be found in
            peft.tuners.sva.utils.prepare_layer_inputs_fn_language_modeling and works for language modeling. In this
            case model_inputs is the mask used to determine which indices should be used for SVD (created by
            `prepare_model_inputs_fn_language_modeling`).
        adapter_name (str): The name of the adapter to compute the SVD for.
        gather_distributed_inputs (bool):
            Whether to gather the layer inputs from all ranks. Default is True meaning in a distributed setting the
            layer inputs will be gathered from all ranks for the SVD computation. For non-distributed settings this
            argument is ignored. Set to False if you are using a non-distributed dataloader in a distributed setting.
        show_progress_bar (bool): Whether to show a progress bar. Default is True.

    Returns:
        sva_state_dict (dict): The state dictionary containing the SVD components for each layer.
    """

    def target_module_check_fn_peft_model(name, module):
        "check if a module is an adapter module via base_layer attribute"
        return hasattr(module, "base_layer")

    def target_module_check_fn_default(name, module, peft_config):
        "check if a module is an adapter module via target_modules"
        is_target_module = True
        if peft_config.target_modules is not None:
            is_target_module = check_target_module_exists(peft_config, name)
        # Conv1D for GPT2 support
        return isinstance(module, (torch.nn.Linear, Conv1D)) and is_target_module

    is_peft_model = hasattr(model, "peft_config")

    # get peft_config
    if is_peft_model and peft_config is None:
        peft_config = model.peft_config[adapter_name]
        if not isinstance(peft_config, SvaConfig):
            raise ValueError("peft_config must be an instance of SvaConfig")
    elif peft_config is None:
        raise ValueError("peft_config is required if model is not a PeftModel")

    # setup context and target module check function
    if is_peft_model:
        ctx = model.disable_adapter()
        target_module_check_fn = target_module_check_fn_peft_model
    else:
        ctx = nullcontext()
        target_module_check_fn = partial(target_module_check_fn_default, peft_config=peft_config)

    with ctx:
        sva_state_dict = _get_sva_state_dict(
            model=model,
            dataloader=dataloader,
            peft_config=peft_config,
            target_module_check_fn=target_module_check_fn,
            forward_fn=forward_fn,
            prepare_model_inputs_fn=prepare_model_inputs_fn,
            prepare_layer_inputs_fn=prepare_layer_inputs_fn,
            gather_distributed_inputs=gather_distributed_inputs,
            show_progress_bar=show_progress_bar,
        )
    return sva_state_dict


def initialize_sva_weights(
    model: torch.nn.Module,
    dataloader: Optional[Iterable] = None,
    sva_state_dict: Optional[dict] = None,
    forward_fn: Optional[callable] = forward_fn_language_modeling,
    prepare_model_inputs_fn: Optional[callable] = prepare_model_inputs_fn_language_modeling,
    prepare_layer_inputs_fn: Union[callable, Dict[str, callable], None] = prepare_layer_inputs_fn_language_modeling,
    adapter_name: str = "default",
    gather_distributed_inputs: bool = True,
    show_progress_bar: bool = True,
):
    """
    Initialize the weights of the LoRA layers using the SVA method.

    This function initializes the weights of the LoRA layers using the SVA method. It computes the SVD for each adapter
    layer and updates the weights accordingly.

    Args:
        model (PeftModel): The peft model to compute the SVD for.
        dataloader (Optional[Iterable]):
            The dataloader to use for the forward pass. If None, sva_state_dict needs to be provided.
        sva_state_dict (Optional[dict]):
            The state_dict to load into the model. If None, a dataloader needs to be provided and the state_dict will
            be computed using `get_sva_state_dict`.
        forward_fn (callable):
            The forward function to use for the forward pass. Takes two arguments: `model` and `inputs`. Default
            behavior is `return model(**inputs)`
        prepare_model_inputs_fn (Optional[callable]):
            This function receives the model inputs and the peft_config and passes the output to
            `prepare_layer_inputs_fn`. Can be used to modify the input to the SVD computation based on the original
            model inputs. For example for language modeling the attention mask is used to determine which indices are
            padding tokens and should not be used for SVD. Any function defined here expects two arguments:
            `model_input` and `peft_config`. `peft.tuners.sva.utils.prepare_model_inputs_fn_language_modeling` is used
            by default.
        prepare_layer_inputs_fn (Union[callable, Dict[str, callable], None]):
            This function receives the layer inputs, the model inputs (potentially modified by
            `prepare_model_inputs_fn`) and the name of the layer and returns the inputs that should be used for SVD for
            that particular layer. Any custom function defined here expects three arguments: `layer_input`,
            `model_input`, and `layer_name` and should return a 2d tensor. The default logic can be found in
            peft.tuners.sva.utils.prepare_layer_inputs_fn_language_modeling and works for language modeling. In this
            case model_inputs is the mask used to determine which indices should be used for SVD (created by
            `prepare_model_inputs_fn_language_modeling`).
        adapter_name (str): The name of the adapter to initialize the weights for.
        gather_distributed_inputs (bool):
            Whether to gather the layer inputs from all ranks. Default is True meaning in a distributed setting the
            layer inputs will be gathered from all ranks for the SVD computation. For non-distributed settings this
            argument is ignored. Set to False if you are using a non-distributed dataloader in a distributed setting.
        show_progress_bar (bool): Whether to show a progress bar. Default is True.

    Returns:
        model (torch.nn.Module): The model with the initialized LoRA weights.
    """
    if not hasattr(model, "peft_config"):
        raise ValueError("model must be a PeftModel")

    # sva currently only works with a single active adapter
    # Important: when removing this requirement, make sure sva init works correctly if the new rank is 0.
    if len(model.active_adapters) > 1:
        raise ValueError("`initialize_sva_weights` currently only works with a single active adapter")

    # compute svd
    if sva_state_dict is None:
        if dataloader is None:
            raise ValueError("dataloader is required if sva_state_dict is not provided")
        sva_state_dict = get_sva_state_dict(
            model=model,
            dataloader=dataloader,
            forward_fn=forward_fn,
            prepare_model_inputs_fn=prepare_model_inputs_fn,
            prepare_layer_inputs_fn=prepare_layer_inputs_fn,
            adapter_name=adapter_name,
            gather_distributed_inputs=gather_distributed_inputs,
            show_progress_bar=show_progress_bar,
        )

    _load_sva_state_dict(model, sva_state_dict, adapter_name)
