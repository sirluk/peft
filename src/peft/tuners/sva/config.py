# Copyright 2023-present the HuggingFace Inc. team.
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
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class SvaConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`SvaModel`].

    Paper: https://arxiv.org/abs/2310.11454.

    Args:
        r (`int`, *optional*, defaults to `256`):
            SVA parameter dimension ("rank"). Choose higher values than LoRA ranks here, since SVA uses far fewer
            parameters than LoRA (see Table 1).
        target_modules (`Union[List[str], str]`):
            The names of the modules to apply SVA to. Only linear layers are supported.
        projection_prng_key (`int`):
            SVA PRNG init key. Used for initialising sva_A and sva_B for new models or when loading a checkpoint
            that did not include these projections. Defaults to `0`.
        save_projection (`bool`):
            Whether to save the sva_A / sva_B projections in the state dict alongside per layer lambda_b / lambda_d
            weights. This will increase the size of the checkpoint, but guarantee that we can reload the checkpoint on
            all system configurations. Defaults to `True`.
        sva_dropout (`float`):
            The dropout probability for SVA layers.
        d_initial (`float`, *optional*, defaults to `0.1`):
            Initial init value for `sva_lambda_d` vector used when initializing the SVA parameters. Small values
            (<=0.1) are recommended (see Table 6c in the paper).
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for SVA. Can be 'none', 'all' or 'sva_only'. If 'all' or 'sva_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from SVA layers to be set as trainable and saved in the final checkpoint.
        init_weights (`bool`):
            Whether to initialize the weights of the SVA layers with their default initialization. Don't change this
            setting, except if you know exactly what you're doing.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the SVA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the SVA
            transformations on the layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
    """

    r: int = field(default=256, metadata={"help": "SVA attention dimension"})
    sva_alpha: int = field(default=1, metadata={"help": "SVA alpha"})
    rank_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer "
                "pattern is not in the common layers pattern. This should target the `nn.ModuleList` of the "
                "model, which is often called `'layers'` or `'h'`."
            )
        },
    )
    alpha_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer "
                "pattern is not in the common layers pattern. This should target the `nn.ModuleList` of the "
                "model, which is often called `'layers'` or `'h'`."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with SVA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    sva_dropout: float = field(default=0.0, metadata={"help": "SVA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for SVA. Can be 'none', 'all' or 'sva_only'"})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from SVA layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the SVA layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer "
                "pattern is not in the common layers pattern. This should target the `nn.ModuleList` of the "
                "model, which is often called `'layers'` or `'h'`."
            )
        },
    )
    tau: float = field(default=0.99, metadata={"help": "Cosine similarity threshold for early stopping"})
    use_label_mask: bool = field(default=True, metadata={"help": "Use label mask for EVA initialization"})
    label_mask_value: int = field(
        default=-100, metadata={"help": "if use_label_mask=True the value to look for to mask out ignored tokens"}
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SVA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")

