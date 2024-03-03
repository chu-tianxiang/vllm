import enum
from enum import Enum
from typing import Any, Dict, List, Optional
from fractions import Fraction

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.utils import is_hip
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)


def pemute_weight(scale, qzeros):
    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    dim = scale.shape[1]
    # unpack and permute qweight
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, -1),
        torch.tensor(list(range(0, 32, 4)), dtype=torch.int32, device=qzeros.device),
        ).bitwise_and(15) + 1
    zeros = zeros.view(zeros.shape[0], -1)
    scale = scale.reshape((-1, len(scale_perm)))[:, scale_perm]
    zeros = zeros.reshape((-1, len(scale_perm)))[:, scale_perm]
    scale = scale.reshape((-1, dim)).contiguous()
    zeros = zeros.reshape((-1, dim)).contiguous()
    return scale, - zeros * scale


class GPTQConfig(QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.sym = sym
        self.pack_factor = Fraction(32, self.weight_bits)
        if self.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is supported for "
                f"GPTQ, but got {self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"GPTQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}, "
                f"sym={self.sym}")

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        sym = cls.get_from_keys(config, ["sym"])
        return cls(weight_bits, group_size, desc_act, sym)

    def get_linear_method(self) -> "GPTQLinearMethod":
        return GPTQLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    def merge_weight(self) -> bool:
        return True


class ExllamaState(Enum):

    UNUSED = enum.auto()
    UNINITIALIZED = enum.auto()
    READY = enum.auto()
    MARLIN_UNINITIALIZED = enum.auto()
    MARLIN_READY = enum.auto()


class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config
        self.workspace = torch.zeros((512,), dtype=torch.int, device="cuda")

    def fit_marlin(self, output_size):
        # Need fix for group_size = -1
        compute_capability = torch.cuda.get_device_capability()
        return compute_capability[0] >= 8 and self.quant_config.group_size == 128 and (
            self.quant_config.weight_bits == 4) and (
            output_size % 256 == 0) and not is_hip()

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        del output_size  # Unused.
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if output_size_per_partition % self.quant_config.pack_factor.numerator != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        exllama_state = ExllamaState.UNINITIALIZED
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None
        # For act-order models, we cannot use Exllama for row parallel layer
        if input_size != input_size_per_partition and self.quant_config.group_size != -1:
            if self.quant_config.desc_act:
                exllama_state = ExllamaState.UNUSED
            else:
                scale_and_zero_size = input_size_per_partition // group_size
                scale_and_zero_input_dim = 0
                if self.fit_marlin(output_size_per_partition):
                    exllama_state = ExllamaState.MARLIN_UNINITIALIZED
        elif self.fit_marlin(output_size_per_partition):
            exllama_state = ExllamaState.MARLIN_UNINITIALIZED

        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            })
        g_idx = Parameter(
            torch.tensor(
                [
                    i // self.quant_config.group_size
                    for i in range(input_size_per_partition)
                ],
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        # Ignore warning from fused linear layers such as QKVParallelLinear.
        set_weight_attrs(g_idx, {"input_dim": 0, "ignore_warning": True})
        qzeros = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": scale_and_zero_input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        scales = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": scale_and_zero_input_dim,
            "output_dim": 1,
        })
        return {
            "qweight": qweight,
            "g_idx": g_idx,
            "qzeros": qzeros,
            "scales": scales,
            "exllama_state": exllama_state,
        }

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_shape = x.shape[:-1] + (weights["scales"].shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        # exllama needs to shuffle the weight after the weight is loaded
        # here we do the shuffle on first forward pass
        if weights["exllama_state"] in (ExllamaState.UNINITIALIZED, ExllamaState.MARLIN_UNINITIALIZED):
            if self.quant_config.desc_act:
                weights["g_idx"] = torch.argsort(weights["g_idx"]).to(
                    torch.int)
            else:
                weights["g_idx"] = torch.empty((1, 1), device="meta")
            if weights["exllama_state"] == ExllamaState.UNINITIALIZED:
                weights["exllama_state"] = ExllamaState.READY
                ops.gptq_shuffle(weights["qweight"], weights["g_idx"],
                                 self.quant_config.weight_bits)
            else:
                weights["exllama_state"] = ExllamaState.MARLIN_READY
                weights["qweight"] = ops.gptq_to_marlin(weights["qweight"], weights["g_idx"])
                weights["scales"], weights["qzeros"] = pemute_weight(
                    weights["scales"], weights["qzeros"])

        if weights["exllama_state"] == ExllamaState.MARLIN_READY:
            output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
            # reorder input for act-order model
            if weights["g_idx"].device != torch.device("meta"):
                reshaped_x = reshaped_x[:, weights["g_idx"]]
            ops.marlin_gemm(reshaped_x, weights["qweight"], output.view(-1, output.shape[-1]),
                            weights["scales"], weights["qzeros"], self.workspace)
        else:
            output = ops.gptq_gemm(reshaped_x, weights["qweight"],
                                   weights["qzeros"], weights["scales"],
                                   weights["g_idx"],
                                   weights["exllama_state"] == ExllamaState.READY,
                                   self.quant_config.weight_bits)
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)
