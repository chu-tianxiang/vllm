from typing import Type

from vllm.model_executor.quantization_utils.awq import AWQConfig
from vllm.model_executor.quantization_utils.gptq import GPTQConfig
from vllm.model_executor.quantization_utils.base import QuantizationConfig
from vllm.model_executor.quantization_utils.squeezellm import SqueezeLLMConfig
from vllm.model_executor.quantization_utils.exl2 import EXL2Config

_QUANTIZATION_REGISTRY = {
    "awq": AWQConfig,
    "gptq": GPTQConfig,
    "squeezellm": SqueezeLLMConfig,
    "exl2": EXL2Config,
}


def get_quant_class(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_REGISTRY[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quant_class",
]
