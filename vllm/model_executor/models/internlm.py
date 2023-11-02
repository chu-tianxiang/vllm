# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.quantized_linear import ParallelLinear
from vllm.model_executor.quantization_utils import QuantizationConfig
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.layers import VocabParallelEmbedding
from vllm.model_executor.weight_utils import (
    hf_model_weights_iterator, load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights, convert_pyslice_to_tensor,
    get_parallel_weight)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class InternLMMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.gate_up_proj = ParallelLinear.column(hidden_size,
                                                  2 * intermediate_size,
                                                  bias=False,
                                                  gather_output=False,
                                                  quant_config=quant_config)
        self.down_proj = ParallelLinear.row(intermediate_size,
                                            hidden_size,
                                            bias=False,
                                            input_is_parallel=True,
                                            quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class InternLMAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rope_theta: float = 10000,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = ParallelLinear.column(
            hidden_size,
            3 * self.total_num_heads * self.head_dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
        )
        self.o_proj = ParallelLinear.row(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            quant_config=quant_config,
        )
        self.attn = PagedAttentionWithRoPE(
            self.num_heads,
            self.head_dim,
            self.scaling,
            base=self.rope_theta,
            max_position=self.max_position_embeddings,
            rotary_dim=self.head_dim)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(positions, q, k, v, k_cache, v_cache,
                                input_metadata, cache_event)
        output, _ = self.o_proj(attn_output)
        return output


class InternLMDecoderLayer(nn.Module):

    def __init__(self,
                 config: LlamaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = InternLMAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
        )
        self.mlp = InternLMMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class InternLMModel(nn.Module):

    def __init__(self,
                 config: LlamaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            InternLMDecoderLayer(config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class InternLMForCausalLM(nn.Module):

    def __init__(self, config, quant_config=None):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = InternLMModel(config, quant_config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.lm_head = ParallelLinear.column(config.hidden_size,
                                             vocab_size,
                                             bias=False,
                                             gather_output=False,
                                             quant_config=None)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   input_metadata)
        return next_tokens

    column_parallel_layers = ["qkv_proj", "gate_proj", "up_proj"]
    row_parallel_layers = ["o_proj", "down_proj"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        column_parallel_weights, row_parallel_weights = get_parallel_weight(
            self)
        column_weight_suffixes = (
            self.quant_config.get_col_parallel_tensor_names()
        ) if self.quant_config is not None else ["weight", "bias"]

        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue

            is_transposed = False
            if self.quant_config is not None:
                is_transposed = self.quant_config.is_transposed(name)
            if is_transposed:
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                loaded_weight = loaded_weight.T

            if "embed_tokens" in name or "lm_head" in name:
                param = state_dict[name]
                load_padded_tensor_parallel_vocab(param, loaded_weight,
                                                  tensor_model_parallel_rank)
                continue

            is_attention_weight = False
            for stride_id, att_weight_name in enumerate(
                ["q_proj", "k_proj", "v_proj"]):
                if att_weight_name not in name:
                    continue
                name = name.replace(att_weight_name, "qkv_proj")
                if name not in state_dict:
                    break
                param = state_dict[name]
                if is_transposed:
                    param = param.T
                shard_size = param.shape[0] // 3
                if any(
                        name.endswith(suffix)
                        for suffix in column_weight_suffixes):
                    loaded_weight = loaded_weight[
                        shard_size * tensor_model_parallel_rank:shard_size *
                        (tensor_model_parallel_rank + 1)]
                    param_slice = param.data[shard_size *
                                             stride_id:shard_size *
                                             (stride_id + 1)]
                else:
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    param_slice = param.data
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, "gate_up_proj")
                if name not in state_dict:
                    break
                param = state_dict[name]
                if is_transposed:
                    param = param.T
                shard_size = param.shape[0] // 2
                if any(
                        name.endswith(suffix)
                        for suffix in column_weight_suffixes):
                    loaded_weight = loaded_weight[
                        shard_size * tensor_model_parallel_rank:shard_size *
                        (tensor_model_parallel_rank + 1)]
                    param_slice = param.data[shard_size *
                                             stride_id:shard_size *
                                             (stride_id + 1)]
                else:
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    param_slice = param.data
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            if name not in state_dict:
                continue
            param = state_dict[name]
            if is_transposed:
                param = param.T
            load_tensor_parallel_weights(self, param, loaded_weight, name,
                                         column_parallel_weights,
                                         row_parallel_weights,
                                         tensor_model_parallel_rank,
                                         is_transposed)
