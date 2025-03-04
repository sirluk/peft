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
from contextlib import nullcontext
from functools import partial
from typing import Dict, Iterable, Optional, Union

import torch
from transformers.pytorch_utils import Conv1D

from peft.tuners.sva_utils import (
    SingularVectorInitializer,
    forward_fn_language_modeling,
    prepare_layer_inputs_fn_language_modeling,
    prepare_model_inputs_fn_language_modeling,
)
from peft.tuners.tuners_utils import _find_minimal_target_modules, check_target_module_exists
from peft.utils.constants import MIN_TARGET_MODULES_FOR_OPTIMIZATION
from peft.utils.other import _get_submodules, get_pattern_key

from .config import LoraConfig
from .layer import Embedding, LoraLayer, MultiheadAttention, _ConvNd


UNSUPPORTED_LORA_MODULES = (Embedding, MultiheadAttention, _ConvNd)


@torch.no_grad()
def _load_eva_state_dict(
    model: torch.nn.Module,
    eva_state_dict: dict,
    adapter_name: str,
):
    peft_config = model.peft_config[adapter_name]
    backward = peft_config.eva_config.gradient_based_decision
    update_layer_kwargs = {
        "adapter_name": adapter_name,
        "lora_dropout": peft_config.lora_dropout,
        "use_rslora": peft_config.use_rslora,
        "use_dora": peft_config.use_dora,
        "lora_bias": peft_config.lora_bias,
    }
    missing_eva_inits = []
    new_target_modules = []
    other_module_names = []
    rank_pattern = {}
    alpha_pattern = {}

    def get_lora_weight(module):
        return module.lora_B[adapter_name] if backward else module.lora_A[adapter_name]

    for name, module in model.named_modules():
        name_in_base_model = name.replace("base_model.model.", "")
        if not isinstance(module, LoraLayer):
            other_module_names.append(name_in_base_model)
            continue
        # Regexp matching - Find key which matches current target_name in patterns provided
        r = peft_config.rank_pattern.get(get_pattern_key(peft_config.rank_pattern.keys(), name), peft_config.r)
        alpha = peft_config.alpha_pattern.get(
            get_pattern_key(peft_config.alpha_pattern.keys(), name), peft_config.lora_alpha
        )
        if name in eva_state_dict:
            w = eva_state_dict.pop(name)
            new_rank = w.size(int(backward))
            if new_rank == 0:
                parent, _, target_name = _get_submodules(model, name)
                setattr(parent, target_name, module.get_base_layer())
                continue
            if new_rank != r:
                if peft_config.eva_config.adjust_scaling_factors:
                    alpha *= new_rank / r
            if new_rank != r or get_lora_weight(module).weight.device.type == "meta":
                module.update_layer(r=new_rank, lora_alpha=alpha, init_lora_weights="eva", **update_layer_kwargs)
            get_lora_weight(module).weight.copy_(w)
            new_target_modules.append(name_in_base_model)
        else:
            module.update_layer(r=r, lora_alpha=alpha, init_lora_weights=True, **update_layer_kwargs)
            missing_eva_inits.append(name_in_base_model)
            new_rank = r
        # update rank pattern and alpha pattern
        if new_rank != peft_config.r:
            rank_pattern[name_in_base_model] = new_rank
        if alpha != peft_config.lora_alpha:
            alpha_pattern[name_in_base_model] = alpha

    # update target modules if some lora layers have been removed due to their EVA rank being 0
    new_target_modules = new_target_modules + missing_eva_inits
    if len(new_target_modules) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION:
        new_target_modules = _find_minimal_target_modules(new_target_modules, other_module_names)
    model.peft_config[adapter_name].target_modules = new_target_modules

    # set rank pattern obtained from EVA
    model.peft_config[adapter_name].rank_pattern = rank_pattern

    # when adjust_scaling_factors is True, lora scaling factors have been adjusted after the rank redistribution
    model.peft_config[adapter_name].alpha_pattern = alpha_pattern

    if missing_eva_inits:
        warnings.warn(
            "the following layers were initialized with init_lora_weights=True because they "
            f"were not found in the eva state_dict: {missing_eva_inits}\ncurrently the "
            f"following lora modules are not supported by EVA: {UNSUPPORTED_LORA_MODULES}"
        )


def get_eva_state_dict(
    model: torch.nn.Module,
    dataloader: Iterable,
    peft_config: Optional[LoraConfig] = None,
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
        peft_config (Optional[LoraConfig]):
            The configuration for the LoRA layers. Only required if `model` is not a PeftModel.
        forward_fn (callable):
            The forward function to use for the forward pass. Takes two arguments: `model` and `inputs`. Default
            behavior is `return model(**inputs)`
        prepare_model_inputs_fn (Optional[callable]):
            This function receives the model inputs and the peft_config and passes the output to
            `prepare_layer_inputs_fn`. Can be used to modify the input to the SVD computation based on the original
            model inputs. For example for language modeling the attention mask is used to determine which indices are
            padding tokens and should not be used for SVD. Any function defined here expects two arguments:
            `model_input` and `peft_config`. `peft.tuners.lora.eva.prepare_model_inputs_fn_language_modeling` is used
            by default.
        prepare_layer_inputs_fn (Union[callable, Dict[str, callable], None]):
            This function receives the layer inputs, the model inputs (potentially modified by
            `prepare_model_inputs_fn`) and the name of the layer and returns the inputs that should be used for SVD for
            that particular layer. Any custom function defined here expects three arguments: `layer_input`,
            `model_input`, and `layer_name` and should return a 2d tensor. The default logic can be found in
            peft.tuners.lora.eva.prepare_layer_inputs_fn_language_modeling and works for language modeling. In this
            case model_inputs is the mask used to determine which indices should be used for SVD (created by
            `prepare_model_inputs_fn_language_modeling`).
        adapter_name (str): The name of the adapter to compute the SVD for.
        gather_distributed_inputs (bool):
            Whether to gather the layer inputs from all ranks. Default is True meaning in a distributed setting the
            layer inputs will be gathered from all ranks for the SVD computation. For non-distributed settings this
            argument is ignored. Set to False if you are using a non-distributed dataloader in a distributed setting.
        show_progress_bar (bool): Whether to show a progress bar. Default is True.

    Returns:
        eva_state_dict (dict): The state dictionary containing the SVD components for each layer.
    """

    def target_module_check_fn_peft_model(name, module, unsupported_lora_modules):
        "check if a module is an adapter module via base_layer attribute"
        return hasattr(module, "base_layer") and not isinstance(module, unsupported_lora_modules)

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
    elif peft_config is None:
        raise ValueError("peft_config is required if model is not a PeftModel")

    # setup context and target module check function
    if is_peft_model:
        ctx = model.disable_adapter()
        target_module_check_fn = partial(
            target_module_check_fn_peft_model, unsupported_lora_modules=UNSUPPORTED_LORA_MODULES
        )
    else:
        ctx = nullcontext()
        target_module_check_fn = partial(target_module_check_fn_default, peft_config=peft_config)

    with ctx:
        sva_instance = SingularVectorInitializer(
            model=model,
            dataloader=dataloader,
            peft_config=peft_config,
            rank=peft_config.r,
            rho=peft_config.eva_config.rho,
            tau=peft_config.eva_config.tau,
            target_module_check_fn=target_module_check_fn,
            forward_fn=forward_fn,
            prepare_model_inputs_fn=prepare_model_inputs_fn,
            prepare_layer_inputs_fn=prepare_layer_inputs_fn,
            gather_distributed_inputs=gather_distributed_inputs,
            whiten=peft_config.eva_config.whiten,
            rank_pattern=peft_config.rank_pattern,
            compute_forward_svd=(not peft_config.eva_config.gradient_based_decision),
            compute_backward_svd=peft_config.eva_config.gradient_based_decision,
            sorting_metric="evr",
            return_sva_metric_in_state_dict=False,
            show_progress_bar=show_progress_bar,
        )
        eva_state_dict = sva_instance.get_sva_state_dict()
    # we can remove the last part of the key because we only get one weight per module
    eva_state_dict = {".".join(k.split(".")[:-1]): v for k, v in eva_state_dict.items()}
    return eva_state_dict


def initialize_lora_eva_weights(
    model: torch.nn.Module,
    dataloader: Optional[Iterable] = None,
    eva_state_dict: Optional[dict] = None,
    forward_fn: Optional[callable] = forward_fn_language_modeling,
    prepare_model_inputs_fn: Optional[callable] = prepare_model_inputs_fn_language_modeling,
    prepare_layer_inputs_fn: Union[callable, Dict[str, callable], None] = prepare_layer_inputs_fn_language_modeling,
    adapter_name: str = "default",
    gather_distributed_inputs: bool = True,
    show_progress_bar: bool = True,
):
    """
    Initialize the weights of the LoRA layers using the EVA method.

    This function initializes the weights of the LoRA layers using the EVA method. It computes the SVD for each adapter
    layer and updates the weights accordingly.

    Args:
        model (PeftModel): The peft model to compute the SVD for.
        dataloader (Optional[Iterable]):
            The dataloader to use for the forward pass. If None, eva_state_dict needs to be provided.
        eva_state_dict (Optional[dict]):
            The state_dict to load into the model. If None, a dataloader needs to be provided and the state_dict will
            be computed using `get_eva_state_dict`.
        forward_fn (callable):
            The forward function to use for the forward pass. Takes two arguments: `model` and `inputs`. Default
            behavior is `return model(**inputs)`
        prepare_model_inputs_fn (Optional[callable]):
            This function receives the model inputs and the peft_config and passes the output to
            `prepare_layer_inputs_fn`. Can be used to modify the input to the SVD computation based on the original
            model inputs. For example for language modeling the attention mask is used to determine which indices are
            padding tokens and should not be used for SVD. Any function defined here expects two arguments:
            `model_input` and `peft_config`. `peft.tuners.lora.eva.prepare_model_inputs_fn_language_modeling` is used
            by default.
        prepare_layer_inputs_fn (Union[callable, Dict[str, callable], None]):
            This function receives the layer inputs, the model inputs (potentially modified by
            `prepare_model_inputs_fn`) and the name of the layer and returns the inputs that should be used for SVD for
            that particular layer. Any custom function defined here expects three arguments: `layer_input`,
            `model_input`, and `layer_name` and should return a 2d tensor. The default logic can be found in
            peft.tuners.lora.eva.prepare_layer_inputs_fn_language_modeling and works for language modeling. In this
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

    # eva currently only works with a single active adapter
    # Important: when removing this requirement, make sure eva init works correctly if the new rank is 0.
    if len(model.active_adapters) > 1:
        raise ValueError("`initialize_lora_eva_weights` currently only works with a single active adapter")

    # initialize_lora_eva_weights only works with `init_lora_weights='eva'`
    if model.peft_config[adapter_name].init_lora_weights != "eva":
        raise ValueError("`initialize_lora_eva_weights` can only be used with `init_lora_weights='eva'`")

    # compute svd
    if eva_state_dict is None:
        if dataloader is None:
            raise ValueError("dataloader is required if eva_state_dict is not provided")
        eva_state_dict = get_eva_state_dict(
            model=model,
            dataloader=dataloader,
            forward_fn=forward_fn,
            prepare_model_inputs_fn=prepare_model_inputs_fn,
            prepare_layer_inputs_fn=prepare_layer_inputs_fn,
            adapter_name=adapter_name,
            gather_distributed_inputs=gather_distributed_inputs,
            show_progress_bar=show_progress_bar,
        )

    _load_eva_state_dict(model, eva_state_dict, adapter_name)
