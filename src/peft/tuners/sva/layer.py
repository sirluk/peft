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

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


class SvaLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("sva_weight",)
    other_param_names = ("sva_A", "sva_B")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.sva_dropout = nn.ModuleDict({})

        # For storing vector scale
        self.sva_weight = nn.ParameterDict({})

        # Stores a reference to the sva_A/B BufferDict.
        # Set to `None` otherwise to avoid computation with random weights
        self.sva_A: BufferDict = BufferDict({}, persistent=True)
        self.sva_B: BufferDict = BufferDict({}, persistent=True)

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type: {type(base_layer)}")

        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name,
        r,
        sva_dropout,
        sva_A: torch.Tensor = None,
        sva_B: torch.Tensor = None,
        eye_init: bool = False,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        if sva_dropout > 0.0:
            sva_dropout_layer = nn.Dropout(p=sva_dropout)
        else:
            sva_dropout_layer = nn.Identity()

        weight = self.get_base_layer().weight
        dtype = weight.dtype

        self.sva_dropout.update(nn.ModuleDict({adapter_name: sva_dropout_layer}))
        # Actual trainable parameters
        if eye_init:
            self.sva_weight[adapter_name] = nn.Parameter(torch.eye(r, dtype=dtype), requires_grad=True)
        else:
            self.sva_weight[adapter_name] = nn.Parameter(torch.zeros(r, r, dtype=dtype), requires_grad=True)

        self.sva_A[adapter_name] = self.sva_weight[adapter_name].new_empty(self.r[adapter_name], self.in_features)
        self.sva_B[adapter_name] = self.sva_weight[adapter_name].new_empty(self.out_features, self.r[adapter_name])

        if bool(sva_A is None) != bool(sva_B is None):
            raise ValueError(
                f"sva_A and sva_B must be both None or both not None but got sva_A={sva_A} and sva_B={sva_B}"
            )

        if sva_A is not None:
            self._verify_sva_AB(adapter_name, sva_A, is_a=True)
            self._verify_sva_AB(adapter_name, sva_B, is_a=False)
            self.sva_A[adapter_name].copy_(sva_A)
            self.sva_B[adapter_name].copy_(sva_B)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def _verify_sva_AB(self, adapter_name, frozen, is_a: bool):
        # check input size
        if is_a:
            expected_shape = (self.r[adapter_name], self.in_features)
        else:
            expected_shape = (self.out_features, self.r[adapter_name])
        if frozen.shape != expected_shape:
            k = "sva_A" if is_a else "sva_B"
            raise ValueError(f"{k} has a size {frozen.shape} but {expected_shape} is expected")

    def reset_sva_parameters(self, adapter_name):
        if adapter_name in self.sva_weight.keys():
            nn.init.zeros_(self.sva_weight[adapter_name])


class Linear(nn.Linear, SvaLayer):
    # SVA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        sva_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        SvaLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, sva_dropout)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.sva_weight.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.sva_weight.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        sva_A = self.sva_A[adapter]
        sva_B = self.sva_B[adapter]
        sva_weight = self.sva_weight[adapter]

        device = sva_weight.device
        dtype = sva_weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
        if cast_to_fp32:
            sva_A = sva_A.float()
            sva_B = sva_B.float()
            sva_weight = sva_weight.float()

        output_tensor = transpose(sva_B @ sva_weight @ sva_A, self.fan_in_fan_out)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.sva_weight.keys():
                    continue

                sva_weight = self.sva_weight[active_adapter]
                sva_A = self.sva_A[active_adapter]
                sva_B = self.sva_B[active_adapter]

                dropout = self.sva_dropout[active_adapter]
                x = F.linear(dropout(x), sva_A)
                x = F.linear(x, sva_weight)
                x = F.linear(x, sva_B)
                result = result + x

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "sva." + rep
