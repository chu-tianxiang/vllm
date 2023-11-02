from typing import Dict, List

import torch

from vllm.model_executor.quantization_utils.base import QuantizationConfig


class EXL2Config(QuantizationConfig):
    """Config class for EXL2."""

    def __repr__(self) -> str:
        return "EXL2Config"

    @classmethod
    def get_name(cls) -> str:
        return "exl2"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_packed_tensors(cls) -> Dict[str, int]:
        return {"qzeros": 1}

    @classmethod
    def get_transposed_tensor_names(cls) -> List[str]:
        return ["q_weight", "q_scale"]

    def get_row_parallel_tensor_names(self) -> List[str]:
        return []

    def get_col_parallel_tensor_names(self) -> List[str]:
        return ["q_weight", "q_scale", "bias"]

    @classmethod
    def merge_weight(cls) -> bool:
        return False

    @classmethod
    def quantize_vocab(cls) -> bool:
        return True
