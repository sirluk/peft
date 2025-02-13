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

from .config import SvaConfig
from .layer import SvaLayer


def _load_sva_state_dict(
    model: torch.nn.Module,
    sva_state_dict: dict,
    adapter_name: str,
):
    sva_config = model.peft_config[adapter_name]
    update_layer_kwargs = {
        "adapter_name": adapter_name,
        "r": sva_config.r,
        "sva_dropout": sva_config.sva_dropout,
        "eye_init": sva_config.eye_init,
    }
    missing_sva_inits = []
    new_target_modules = []
    other_module_names = []
    for name, module in model.named_modules():
        name_in_base_model = name.replace("base_model.model.", "")
        if not isinstance(module, SvaLayer):
            other_module_names.append(name_in_base_model)
            continue
        sva_A = sva_state_dict.pop(f"{name}.sva_A", None)
        sva_B = sva_state_dict.pop(f"{name}.sva_B", None)
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
    sva_config: Optional[SvaConfig] = None,
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
        sva_config (Optional[SvaConfig]):
            The configuration for the LoRA layers. Only required if `model` is not a PeftModel.
        forward_fn (callable):
            The forward function to use for the forward pass. Takes two arguments: `model` and `inputs`. Default
            behavior is `return model(**inputs)`
        prepare_model_inputs_fn (Optional[callable]):
            This function receives the model inputs and the sva_config and passes the output to
            `prepare_layer_inputs_fn`. Can be used to modify the input to the SVD computation based on the original
            model inputs. For example for language modeling the attention mask is used to determine which indices are
            padding tokens and should not be used for SVD. Any function defined here expects two arguments:
            `model_input` and `sva_config`. `peft.tuners.sva.utils.prepare_model_inputs_fn_language_modeling` is used
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

    def target_module_check_fn_default(name, module, sva_config):
        "check if a module is an adapter module via target_modules"
        is_target_module = True
        if sva_config.target_modules is not None:
            is_target_module = check_target_module_exists(sva_config, name)
        # Conv1D for GPT2 support
        return isinstance(module, (torch.nn.Linear, Conv1D)) and is_target_module

    is_peft_model = hasattr(model, "peft_config")

    # get sva_config
    if is_peft_model and sva_config is None:
        sva_config = model.peft_config[adapter_name]
        if not isinstance(sva_config, SvaConfig):
            raise ValueError("model.peft_config[adapter_name] must be an instance of SvaConfig")
    elif sva_config is None:
        raise ValueError("sva_config is required if model is not a PeftModel")
    # setup context and target module check function
    if is_peft_model:
        ctx = model.disable_adapter()
        target_module_check_fn = target_module_check_fn_peft_model
    else:
        ctx = nullcontext()
        target_module_check_fn = partial(target_module_check_fn_default, sva_config=sva_config)

    with ctx:
        sva_instance = SingularVectorInitializer(
            model=model,
            dataloader=dataloader,
            peft_config=sva_config,
            rank=sva_config.r,
            rho=sva_config.rho,
            tau=sva_config.tau,
            target_module_check_fn=target_module_check_fn,
            forward_fn=forward_fn,
            prepare_model_inputs_fn=prepare_model_inputs_fn,
            prepare_layer_inputs_fn=prepare_layer_inputs_fn,
            gather_distributed_inputs=gather_distributed_inputs,
            whiten=False,
            rank_pattern=None,
            compute_forward_svd=True,
            compute_backward_svd=True,
            sorting_strategy="kfac",
            show_progress_bar=show_progress_bar,
        )
        sva_state_dict = sva_instance.get_sva_state_dict()
    sva_state_dict = {k + "." + adapter_name: v for k, v in sva_state_dict.items()}
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
            This function receives the model inputs and the sva_config and passes the output to
            `prepare_layer_inputs_fn`. Can be used to modify the input to the SVD computation based on the original
            model inputs. For example for language modeling the attention mask is used to determine which indices are
            padding tokens and should not be used for SVD. Any function defined here expects two arguments:
            `model_input` and `sva_config`. `peft.tuners.sva.utils.prepare_model_inputs_fn_language_modeling` is used
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
    if not isinstance(model.peft_config[adapter_name], SvaConfig):
        raise ValueError("model.peft_config[adapter_name] must be an instance of SvaConfig")

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
