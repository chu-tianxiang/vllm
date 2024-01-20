from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

GGML_QUANT_SIZES = {
    0:  (1, 4),
    1:  (1, 2),
    2: (32, 2 + 16),
    3: (32, 2 + 2 + 16),
    6: (32, 2 + 4 + 16),
    7: (32, 2 + 2 + 4 + 16),
    8: (32, 2 + 32),
    9: (32, 4 + 4 + 32),
    10: (256, 2 + 2 + 256 // 16 + 256 // 4),
    11: (256, 2 + 256 // 4 + 256 // 8 + 12),
    12: (256, 2 + 2 + 256 // 2 + 12),
    13: (256, 2 + 2 + 256 // 2 + 256 // 8 + 12),
    14: (256, 2 + 256 // 2 + 256 // 4 + 256 // 16),
    15: (256, 4 + 256 + 256 // 8),
    16: (256, 2 + 256 // 4),
    17: (256, 2 + 256 // 4 + 256 // 32),
}

class GGUFConfig(QuantizationConfig):
    """Config class for GGUF"""

    def __repr__(self) -> str:
        return (f"GGUFConfig()")

    def get_name(self) -> str:
        return "gguf"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GGUFConfig":
        return cls()

    def get_linear_method(self) -> "GGUFLinearMethod":
        return GGUFLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    def merge_weight(self) -> bool:
        return False

    def rope_style(self) -> Optional[bool]:
        return False


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        # The type of weight is unknown until load state dict
        weight = torch.nn.parameter.UninitializedParameter(
            requires_grad=False
        )
        # No need for pack_factor because we don't fuse qkv layers anyway.
        set_weight_attrs(
            weight, {
                "input_dim": 1,
                "output_dim": 0,
            })
        weight_type = Parameter(
            torch.tensor((1), dtype=torch.int, device="cuda"),
            requires_grad=False,
        )
        set_weight_attrs(weight_type, {"ignore_warning": True})
        return {
            "weight": weight,
            "weight_type": weight_type
        }

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(weights["weight_type"], torch.Tensor):
            weights["weight_type"] = int(weights["weight_type"])
            # Check tensor parallel shape here on first pass
            block_size = GGML_QUANT_SIZES[weights["weight_type"]][1]
            if weights["weight"].shape[1] % block_size != 0:
                raise ValueError("Size is not aligned with the quantized weight shape.")

        weight = weights["weight"]
        weight_type = weights["weight_type"]
        infeatures = x.shape[-1]
        outfeatures = weight.shape[0]
        out_shape = (x.shape[:-1] + (weight.shape[0], ))
        reshaped_x = x.reshape(-1, x.shape[-1])

        xshape = x.view(-1, x.shape[-1])
        if xshape.shape[0] == 1:
            out = ops.ggml_mul_mat_vec(weight, reshaped_x, weight_type, outfeatures)
        else:
            weight = ops.ggml_dequantize(weight, weight_type, outfeatures, infeatures)
            out = reshaped_x @ weight.T

        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)
