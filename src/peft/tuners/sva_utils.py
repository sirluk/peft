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
from enum import Enum
from itertools import cycle
from typing import Dict, Iterable, Optional, Union

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding

from peft.config import PeftConfig
from peft.utils.incremental_pca import IncrementalPCA
from peft.utils.other import get_pattern_key


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
        if self.is_backward_hook:  # SVD on gradients w.r.t. layer outputs
            states = self.prepare_layer_inputs(output)
        else:  # SVD on layer inputs
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


def prepare_model_inputs_fn_language_modeling(model_input, peft_config: PeftConfig):
    """
    Get the indices of the items that should be used for SVD.

    Attributes:
        model_input (dict): The model inputs.
        peft_config (PeftConfig): The configuration for the LoRA layers.
    """
    if not isinstance(model_input, (dict, BatchEncoding)):
        raise ValueError("When using `prepare_model_inputs_fn_language_modeling` inputs must be a dictionary")
    mask = model_input.get("attention_mask", torch.ones_like(model_input["input_ids"])).bool()
    if peft_config.to_dict().get("use_label_mask", False) and hasattr(model_input, "labels"):
        mask = torch.logical_and(mask, model_input["labels"] != peft_config.to_dict().get("label_mask_value", -100))
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


class SortingStrategy(Enum):
    SIMPLE = "simple"
    KFAC = "kfac"


class SingularVectorInitializer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: Iterable,
        peft_config: PeftConfig,
        rank: int,
        rho: int,
        tau: float,
        target_module_check_fn: callable,
        forward_fn: Optional[callable] = forward_fn_language_modeling,
        prepare_model_inputs_fn: Optional[callable] = prepare_model_inputs_fn_language_modeling,
        prepare_layer_inputs_fn: Union[
            callable, Dict[str, callable], None
        ] = prepare_layer_inputs_fn_language_modeling,
        gather_distributed_inputs: bool = True,
        whiten: bool = False,
        rank_pattern: Optional[dict[str, int]] = None,
        compute_forward_svd: bool = False,
        compute_backward_svd: bool = False,
        sorting_strategy: str = "simple",
        show_progress_bar: bool = True,
    ):
        self.model = model
        self.dataloader = dataloader
        self.peft_config = peft_config
        self.rank = rank
        self.rho = rho
        self.tau = tau
        self.whiten = whiten
        self.target_module_check_fn = target_module_check_fn
        self.forward_fn = forward_fn
        self.prepare_model_inputs_fn = prepare_model_inputs_fn
        self.prepare_layer_inputs_fn = prepare_layer_inputs_fn
        self.gather_distributed_inputs = gather_distributed_inputs
        self.rank_pattern = rank_pattern if rank_pattern is not None else {}
        self.compute_forward_svd = compute_forward_svd
        self.compute_backward_svd = compute_backward_svd
        self.sorting_strategy = SortingStrategy(sorting_strategy)
        self.show_progress_bar = show_progress_bar

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

        if not compute_forward_svd and not compute_backward_svd:
            raise ValueError("compute_forward_svd and compute_backward_svd cannot both be False")

        # for unusually high rho values, define an upper limit
        rho_threshold = 1000
        if rho > rho_threshold:
            max_dim = max(max(p.shape) for p in model.parameters())
            rho_ceil = max_dim // rank
            rho = min(rho, rho_ceil)

        self.model_training = model.training
        self.model_device = get_device_with_meta_params(model)

        self.hooks = {}
        self.target_layers = []
        self.max_components = {}
        self.rank_budget = 0

        self.layer_hook_map = {}
        self.equal_inputs_map = {}

        self.grad_hook_handle = None

    def _get_sorting_metric(self, svd):
        if self.sorting_strategy == SortingStrategy.SIMPLE:
            return svd.explained_variance_ratio_
        if self.sorting_strategy == SortingStrategy.KFAC:
            return svd.singular_values_**2
        raise ValueError(f"sorting_strategy {self.sorting_strategy} not supported")

    def _get_rank_counts_single(self, backward_svd: bool):
        """
        Computes the rank distribution for each layer based on the explained variance ratio.
        When rank_pattern flag is False, all values in max_components are the same
        """
        exp_vars = {
            k: self._get_sorting_metric(h[backward_svd][0].svd)[: self.max_components[k]]
            for k, h in self.hooks.items()
        }
        keys, values = zip(*[(k, c) for k, name in self.layer_hook_map.items() for c in exp_vars[name]])
        idx = torch.stack(values).argsort(descending=True)
        counts = Counter([keys[i] for i in idx[: self.rank_budget]])
        return {k: counts.get(k, 0) for k in self.layer_hook_map.keys()}  # add layers with 0 rank

    def _get_rank_counts_kfac(self):
        exp_vars = {}
        for k, layer_hooks in self.hooks.items():
            mc = self.max_components[k]
            (hook_forward, _), (hook_backward, _) = layer_hooks
            exp_vars_forward = self._get_sorting_metric(hook_forward.svd)[:mc]
            exp_vars_backward = self._get_sorting_metric(hook_backward.svd)[:mc]
            prod = exp_vars_forward[:, None] * exp_vars_backward[None, :]
            sorted_values, sorted_idx = prod.view(-1).sort(descending=True)
            i, j = sorted_idx // mc, sorted_idx % mc
            max_idx = torch.stack([i[:mc], j[:mc]])
            exp_vars[k] = (sorted_values[:mc], max_idx + 1)
        keys, values = zip(*[(k, v) for k, name in self.layer_hook_map.items() for v in exp_vars[name][0]])
        idx = torch.stack(values).argsort(descending=True)
        counts = Counter([keys[i] for i in idx[: self.rank_budget]])
        # counts contains information about how many combinations to take from each layer
        # we still need to get the rank required for each hook as it might be less than counts because we take combinations of singular values
        combinations = {}
        dummy_rank = max_idx.new_zeros(3, 1)
        for k, name in self.layer_hook_map.items():
            count = counts.get(name, 0)
            combinations[k] = exp_vars.get(name, dummy_rank)[1][:, :count]
            # forward_ranks[k] = comb[0].max().item()
            # backward_ranks[k] = comb[1].max().item()
        return combinations

    def _get_rank_distribution(self):
        def map_equal_inputs(ranks, equal_inputs_map):
            for k, k_hook in equal_inputs_map.items():
                # ensure hook layers have the highest rank if they are equal to another layer
                rank, rank_hook = ranks[k], ranks[k_hook]
                if isinstance(rank, torch.Tensor):
                    check = rank_hook.max() >= rank.max()
                else:
                    check = rank_hook >= rank
                if check:
                    continue
                ranks[k_hook], ranks[k] = rank, rank_hook
            return ranks

        if self.sorting_strategy == SortingStrategy.KFAC:
            combinations = self._get_rank_counts_kfac()
            combinations = map_equal_inputs(combinations, self.equal_inputs_map)
            forward_ranks, backward_ranks = [
                dict(x) for x in zip(*[((k, c[0]), (k, c[1])) for k, c in combinations.items()])
            ]
        else:
            forward_ranks = None
            backward_ranks = None
            if self.compute_forward_svd:
                forward_ranks = self._get_rank_counts_single(backward_svd=False)
            if self.compute_backward_svd:
                backward_ranks = self._get_rank_counts_single(backward_svd=True)
        forward_ranks = map_equal_inputs(forward_ranks, self.equal_inputs_map)
        backward_ranks = map_equal_inputs(backward_ranks, self.equal_inputs_map)
        return forward_ranks, backward_ranks

    @staticmethod
    def _check_convergence(ranks, hook):
        """
        Checks if a layer has converged.
        """
        if isinstance(ranks, torch.Tensor):
            # in case rank dist is a tensor of indices we check if all indices have converged
            idx = ranks.unique().cpu() - 1
            return torch.all(hook.converged[idx]).item()
        return torch.all(hook.converged[:ranks])

    def _update_convergence_dict(self, name, hook, handle, convergence_dict, backward, rank_dist):
        """
        Updates the convergence dictionary.
        """
        converged = self._check_convergence(rank_dist[name], hook)
        # if a layer has switched from not converged to converged in the current step
        if (not convergence_dict[name][backward]) and converged and handle:
            handle.remove()
            handle = None
            if backward:
                self._register_grad_hook()
            convergence_dict[name][backward] = True
        # if a layer has switched from converged to not converged in the current step
        elif convergence_dict[name][backward] and not converged:
            module = self.model.get_submodule(name)
            if backward:
                handle = module.register_full_backward_hook(hook)
                self._remove_grad_hook()
            else:
                handle = module.register_forward_hook(hook)
            convergence_dict[name][backward] = False
        return convergence_dict, handle

    def _get_model_inputs(self, inputs):
        """
        Get the model inputs.
        """
        if self.prepare_model_inputs_fn is not None:
            model_inputs_for_hooks = self.prepare_model_inputs_fn(inputs, self.peft_config)
        else:
            model_inputs_for_hooks = deepcopy(inputs)
        return model_inputs_for_hooks

    def move_inputs_to_device(self, inputs, device: Union[str, torch.device, None]):
        """
        Move the inputs to the specified device. Adapted from hf.Trainer.
        """
        if device is None:
            return inputs
        if hasattr(inputs, "to"):
            return inputs.to(device)
        if isinstance(inputs, Mapping):
            return type(inputs)({k: self.move_inputs_to_device(v, device) for k, v in inputs.items()})
        if isinstance(inputs, (tuple, list)):
            return type(inputs)(self.move_inputs_to_device(v, device) for v in inputs)
        warnings.warn(f"input of type {type(inputs)} could not be moved to the correct device")
        return inputs

    @staticmethod
    def _whiten(u, singular_values):
        return u / singular_values.sqrt().reshape(-1, 1)

    @torch.no_grad()
    def _initialize_hooks(self):
        """
        Initialize the hooks.
        """
        # get model inputs
        inputs = next(iter(self.dataloader))
        inputs = self.move_inputs_to_device(inputs, self.model_device)
        model_inputs_for_hooks = self._get_model_inputs(inputs)
        for name, module in self.model.named_modules():
            if not self.target_module_check_fn(name, module):
                continue
            if isinstance(self.prepare_layer_inputs_fn, Mapping):
                fn = self.prepare_layer_inputs_fn.pop(name, None)
            else:
                fn = self.prepare_layer_inputs_fn
            hook = HashHook(
                name=name, prepare_layer_inputs_fn=fn, gather_distributed_inputs=self.gather_distributed_inputs
            )
            hook.model_input = model_inputs_for_hooks
            handle = module.register_forward_hook(hook)
            self.hooks[name] = (hook, handle)
            self.target_layers.append(name)
            layer_rank = self.rank_pattern.get(get_pattern_key(self.rank_pattern.keys(), name), self.rank)
            self.max_components[name] = round(layer_rank * self.rho)
            self.rank_budget += layer_rank

        if isinstance(self.prepare_layer_inputs_fn, Mapping) and len(self.prepare_layer_inputs_fn) < len(
            self.target_layers
        ):
            missing = [n for n in self.target_layers if n not in self.prepare_layer_inputs_fn.keys()]
            raise ValueError(
                f"prepare_layer_inputs_fn is a mapping but the following module names were not found in the model: {missing}"
            )

        # forward for one batch to check which layer inputs are equal to avoid unneeded svd calculations
        self.forward_fn(self.model, inputs)
        hash_dict = {k: h[0].hashed_inputs[0] for k, h in self.hooks.items()}
        # equal input maps groups layers which receive the same input. One layer is defined as the key and receives an svd
        # hook. For the remaining layers the svd results can be skipped.
        equal_inputs = list(find_equal_values(hash_dict).values())
        self.equal_inputs_map = {vv: v[0] for v in equal_inputs for vv in v[1:]}
        # for layers with equal inputs we need to make sure that the max_components are the same
        for names in equal_inputs:
            max_value = max(self.max_components[n] for n in names)
            for n in names:
                self.max_components[n] = max_value

        # initialize svd hooks
        for name in list(self.hooks.keys()):
            hook, handle = self.hooks.pop(name)
            handle.remove()
            if name in self.equal_inputs_map:
                continue
            module = self.model.get_submodule(name)
            hook_forward = None
            handle_forward = None
            hook_backward = None
            handle_backward = None
            if self.compute_forward_svd:
                hook_forward = SVDHook(
                    n_components=self.rank,
                    sim_thresh=self.tau,
                    name=name,
                    prepare_layer_inputs_fn=hook._prepare_layer_inputs_fn,
                    gather_distributed_inputs=self.gather_distributed_inputs,
                    is_backward_hook=False,
                )
                handle_forward = module.register_forward_hook(hook_forward)
            if self.compute_backward_svd:
                hook_backward = SVDHook(
                    n_components=self.rank,
                    sim_thresh=self.tau,
                    name=name,
                    prepare_layer_inputs_fn=hook._prepare_layer_inputs_fn,
                    gather_distributed_inputs=self.gather_distributed_inputs,
                    is_backward_hook=True,
                )
                handle_backward = module.register_full_backward_hook(hook_backward)
            self.hooks[name] = ((hook_forward, handle_forward), (hook_backward, handle_backward))
        self.layer_hook_map = {**dict(zip(self.hooks.keys(), self.hooks.keys())), **self.equal_inputs_map}

    def _register_grad_hook(self):
        ############## TEMP HACK ##############
        first_module = [m for n, m in self.model.named_modules() if n.endswith("embed_tokens")][0]
        self.grad_hook_handle = first_module.register_forward_hook(
            lambda module, input, output: output.requires_grad_(True)
        )
        ############## TEMP HACK ##############

    def _remove_grad_hook(self):
        if self.grad_hook_handle is not None:
            self.grad_hook_handle.remove()
            self.grad_hook_handle = None

    def get_sva_state_dict(self):
        self.model.eval()
        self._initialize_hooks()

        if self.compute_backward_svd:
            self._register_grad_hook()

        # start svd calculation
        if self.show_progress_bar and (not dist.is_initialized() or dist.get_rank() == 0):
            pbar = tqdm(iter(cycle(self.dataloader)), position=0, leave=False)
            use_tqdm = True
        else:
            pbar = iter(cycle(self.dataloader))
            use_tqdm = False
        all_backward_converged = not self.compute_backward_svd
        convergence_dict = {
            k: [not self.compute_forward_svd, not self.compute_backward_svd] for k in self.hooks.keys()
        }
        forward_ranks = self.max_components.copy()
        backward_ranks = self.max_components.copy()
        for inputs in pbar:
            if self.model_device is not None:
                inputs = self.move_inputs_to_device(inputs, self.model_device)
            model_inputs_for_hooks = self._get_model_inputs(inputs)
            for name in list(self.hooks.keys()):
                hook_forward, handle_forward = self.hooks[name][0]
                hook_backward, handle_backward = self.hooks[name][1]
                if self.compute_forward_svd:
                    convergence_dict, handle_forward = self._update_convergence_dict(
                        name=name,
                        hook=hook_forward,
                        handle=handle_forward,
                        convergence_dict=convergence_dict,
                        backward=False,
                        rank_dist=forward_ranks,
                    )
                    hook_forward.model_input = model_inputs_for_hooks
                if self.compute_backward_svd:
                    convergence_dict, handle_backward = self._update_convergence_dict(
                        name=name,
                        hook=hook_backward,
                        handle=handle_backward,
                        convergence_dict=convergence_dict,
                        backward=True,
                        rank_dist=backward_ranks,
                    )
                    hook_backward.model_input = model_inputs_for_hooks
                self.hooks[name] = ((hook_forward, handle_forward), (hook_backward, handle_backward))

            f, b = zip(*convergence_dict.values())
            all_forward_converged = all(f)
            all_backward_converged = all(b)

            if use_tqdm:
                layer_converged = [all(x) for x in convergence_dict.values()] + [
                    all(convergence_dict[v]) for v in self.equal_inputs_map.values()
                ]
                pbar.set_description(f"{sum(layer_converged)}/{len(layer_converged)} layers have converged")

            if all_forward_converged and all_backward_converged:
                break

            if all_backward_converged:
                ctx = torch.no_grad()
            else:
                ctx = nullcontext()

            with ctx:
                loss = self.forward_fn(self.model, inputs, compute_loss=not all_backward_converged)
            if not all_backward_converged:
                loss.backward()
                self.model.zero_grad()

            # in case some hooks have to skip the svd calculation because the number of tokens is less than the number of
            # components
            if not all(
                hasattr(hook.svd, "components_")
                for layer_hooks in self.hooks.values()
                for hook, _ in layer_hooks
                if hook is not None
            ):
                continue

            # if self.rho > 1.:
            forward_ranks, backward_ranks = self._get_rank_distribution()

        # check all custom hooks have been removed
        for method in ["_forward_hooks", "_backward_hooks"]:
            remaining_hooks = {
                n for n, m in self.model.named_modules() for v in getattr(m, method).values() if isinstance(v, _Hook)
            }
            if len(remaining_hooks) > 0:
                raise ValueError(
                    f"Found active hooks added by SVA that weren't properly removed: {remaining_hooks}. "
                    "Please report this issue at https://github.com/huggingface/peft/issues"
                )

        sva_state_dict = {}
        converged = True
        for name in self.target_layers:
            rank_f = None
            rank_b = None
            if self.compute_forward_svd:
                rank_f = forward_ranks[name]
                hook_f, _ = self.hooks[self.layer_hook_map[name]][0]
                u_A = hook_f.svd.components_
                if self.whiten:
                    u_A = self._whiten(u_A, hook_f.svd.singular_values_)
                converged = converged and self._check_convergence(rank_f, hook_f)
                sva_state_dict[f"{name}.sva_A"] = u_A
            if self.compute_backward_svd:
                rank_b = backward_ranks[name]
                hook_b, _ = self.hooks[self.layer_hook_map[name]][1]
                u_B = hook_b.svd.components_
                if self.whiten:
                    u_B = self._whiten(u_B, hook_b.svd.singular_values_)
                converged = converged and self._check_convergence(rank_b, hook_b)
                sva_state_dict[f"{name}.sva_B"] = u_B.T
            if self.sorting_strategy == SortingStrategy.KFAC:
                metric_forward = self._get_sorting_metric(hook_f.svd)[rank_f]
                metric_backward = self._get_sorting_metric(hook_b.svd)[rank_b]
                alpha = metric_forward.sqrt() * metric_backward.sqrt()
                u_A = u_A.gather(0, rank_f.view(-1, 1).expand(-1, u_A.size(1)))
                u_B = u_B.gather(0, rank_b.view(-1, 1).expand(-1, u_B.size(1)))
                sva_state_dict[f"{name}.sva_A"] = u_A * alpha.view(-1, 1)
                sva_state_dict[f"{name}.sva_B"] = u_B.T
            if not converged:
                if isinstance(rank_f, torch.Tensor):
                    rank = rank_f.size(0)
                elif isinstance(rank_b, torch.Tensor):
                    rank = rank_b.size(0)
                else:
                    rank = rank_f or rank_b
                raise ValueError(
                    f"Layer {name} has not converged but was assigned rank {rank}. "
                    "Please report this issue at https://github.com/huggingface/peft/issues"
                )

        self.model.train(self.model_training)

        # move tensors to device
        if self.model_device is not None:
            sva_state_dict = {k: v.to(self.model_device) for k, v in sva_state_dict.items()}

        return sva_state_dict
