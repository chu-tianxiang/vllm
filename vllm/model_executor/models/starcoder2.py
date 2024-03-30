# coding=utf-8
# Copyright 2024 BigCode and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch Starcoder2 model."""
from typing import List, Optional

import torch
from torch import nn
from transformers import Starcoder2Config

from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput


class Starcoder2Attention(nn.Module):

    def __init__(self,
                 config: Starcoder2Config,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.use_bias = config.use_bias
        self.sliding_window = config.sliding_window
        if linear_method is not None and not linear_method.quant_config.merge_weight(
        ):
            self.merge_weight = False
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.total_num_heads *
                                               self.head_dim,
                                               bias=self.use_bias,
                                               linear_method=linear_method)
            self.k_proj = ColumnParallelLinear(self.hidden_size,
                                               self.total_num_kv_heads *
                                               self.head_dim,
                                               bias=self.use_bias,
                                               linear_method=linear_method)
            self.v_proj = ColumnParallelLinear(self.hidden_size,
                                               self.total_num_kv_heads *
                                               self.head_dim,
                                               bias=self.use_bias,
                                               linear_method=linear_method)
        else:
            self.merge_weight = True
            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_size,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=self.use_bias,
                linear_method=linear_method,
            )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=self.use_bias,
            linear_method=linear_method,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            sliding_window=self.sliding_window,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self.merge_weight:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)
        else:
            q, _ = self.q_proj(hidden_states)
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Starcoder2MLP(nn.Module):

    def __init__(self,
                 config: Starcoder2Config,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.c_fc = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
            linear_method=linear_method,
        )
        self.c_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            linear_method=linear_method,
        )
        quant_config = getattr(linear_method, "quant_config", None)
        self.act = get_act_fn(config.hidden_act, quant_config,
                              config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class Starcoder2DecoderLayer(nn.Module):

    def __init__(self,
                 config: Starcoder2Config,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Starcoder2Attention(config,
                                             linear_method=linear_method)
        self.mlp = Starcoder2MLP(config, linear_method=linear_method)
        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                     eps=config.norm_epsilon)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Starcoder2Model(nn.Module):

    def __init__(self,
                 config: Starcoder2Config,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # TODO: consider padding_idx (currently removed)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size,
                                                   linear_method=linear_method)
        self.layers = nn.ModuleList([
            Starcoder2DecoderLayer(config, linear_method=linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states, kv_caches[i],
                                  attn_metadata)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Starcoder2ForCausalLM(nn.Module):

    def __init__(self,
                 config: Starcoder2Config,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        self.model = Starcoder2Model(config, linear_method=linear_method)
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
        )
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        if self.linear_method is not None and not self.linear_method.quant_config.merge_weight(
        ):
            stacked_params_mapping = []

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision,
                self.config):
            if "rotary_emb.inv_freq" in name:
                continue

            if "embed_tokens" in name:
                # Copy word embedding to lm_head
                head_name = name.replace("model.embed_tokens", "lm_head")
                if head_name in params_dict and self.config.tie_word_embeddings:
                    lm_head_param = params_dict[head_name]
                    weight_loader = getattr(lm_head_param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(lm_head_param, loaded_weight)

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
