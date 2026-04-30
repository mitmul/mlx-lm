# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "plamo3"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 2048
    window_size: int = 2048
    sliding_window: Optional[int] = None
    sliding_window_pattern: int = 8
    rope_theta: float = 1_000_000.0
    rope_local_theta: float = 10_000.0
    intermediate_size: int = 13312
    vocab_size: int = 32000
    image_token_id: Optional[int] = None
    image_feature_size: Optional[int] = None
    image_proj_type: str = "linear"
    linear_type: str = "normal"

    def __post_init__(self):
        if self.sliding_window is not None:
            self.window_size = self.sliding_window


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        offset: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = mx.zeros(hidden_size)
        self.variance_epsilon = eps
        self.offset = offset

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return mx.fast.rms_norm(
            hidden_states, self.weight + self.offset, self.variance_epsilon
        )


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        head_dim = config.head_dim
        self.scale = head_dim**-0.5

        self.q_num_heads = config.num_attention_heads
        self.qk_dim = self.v_dim = head_dim
        self.k_num_heads = self.v_num_heads = config.num_key_value_heads
        assert self.q_num_heads % self.k_num_heads == 0

        self.q_proj_dim = self.q_num_heads * self.qk_dim
        self.k_proj_dim = self.k_num_heads * self.qk_dim
        self.v_proj_dim = self.v_num_heads * self.v_dim
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.q_proj_dim + self.k_proj_dim + self.v_proj_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.q_num_heads * self.v_dim, self.hidden_size, bias=False
        )

        self.q_norm = RMSNorm(self.qk_dim, eps=self.config.rms_norm_eps, offset=1.0)
        self.k_norm = RMSNorm(self.qk_dim, eps=self.config.rms_norm_eps, offset=1.0)

        self.full_attn = (layer_idx + 1) % config.sliding_window_pattern == 0
        base = (
            self.config.rope_theta
            if self.full_attn
            else self.config.rope_local_theta
        )
        self.rope = nn.RoPE(self.qk_dim, traditional=False, base=base)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ):
        B, T, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = mx.split(
            qkv, [self.q_proj_dim, self.q_proj_dim + self.k_proj_dim], axis=-1
        )
        q = q.reshape(B, T, self.q_num_heads, self.qk_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.k_num_heads, self.qk_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.v_num_heads, self.v_dim).transpose(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if cache is None:
            q = self.rope(q)
            k = self.rope(k)
        elif self.full_attn:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            k, v = cache.update_and_fetch(k, v)
            cached_k, cached_v = k, v
            if T == 1 and k.shape[-2] > self.config.window_size:
                cached_k = k = k[..., -self.config.window_size :, :]
                cached_v = v = v[..., -self.config.window_size :, :]
            key_len = k.shape[-2]
            q = self.rope(q, offset=key_len - T)
            k = self.rope(k)

        output = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            B, T, self.q_num_heads * self.v_dim
        )

        if (
            cache is not None
            and not self.full_attn
            and cache.offset > self.config.window_size
        ):
            cache.keys = cached_k[..., -self.config.window_size :, :]
            cache.values = cached_v[..., -self.config.window_size :, :]
            cache.offset = self.config.window_size

        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size * 2, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


class Plamo3DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.mixer = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.pre_mixer_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0
        )
        self.post_mixer_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / 5
        )
        self.pre_mlp_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0
        )
        self.post_mlp_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / (5**1.5)
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ):
        residual = hidden_states
        hidden_states = self.pre_mixer_norm(hidden_states)
        hidden_states_sa = self.mixer(
            hidden_states=hidden_states,
            mask=mask,
            cache=cache,
        )
        hidden_states = residual + self.post_mixer_norm(hidden_states_sa)

        residual = hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)
        hidden_states_mlp = self.mlp(hidden_states)
        return residual + self.post_mlp_norm(hidden_states_mlp)


class Plamo3Decoder(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.layers = [
            Plamo3DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]
        self.layer_types = [
            "full_attention"
            if (i + 1) % config.sliding_window_pattern == 0
            else "sliding_attention"
            for i in range(config.num_hidden_layers)
        ]

    def __call__(self, x: mx.array, cache):
        if cache is None:
            cache = [None] * len(self.layers)

        full_idx = next(
            (i for i, t in enumerate(self.layer_types) if t == "full_attention"), None
        )
        sliding_idx = next(
            (i for i, t in enumerate(self.layer_types) if t != "full_attention"), None
        )

        full_mask = (
            create_attention_mask(x, cache[full_idx]) if full_idx is not None else None
        )
        sliding_mask = (
            create_attention_mask(
                x,
                cache[sliding_idx],
                window_size=self.layers[sliding_idx].mixer.config.window_size,
            )
            if sliding_idx is not None
            else None
        )

        for layer, c, layer_type in zip(self.layers, cache, self.layer_types):
            mask = full_mask if layer_type == "full_attention" else sliding_mask
            x = layer(x, mask=mask, cache=c)
        return x


class Plamo3Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = Plamo3Decoder(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)
        return self.norm(self.layers(h, cache))


class Model(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Plamo3Model(config)
        self.vocab_size = config.vocab_size

        if not config.tie_word_embeddings:
            lm_head_vocab_size = ((self.vocab_size + 15) // 16) * 16
            self.lm_head: nn.Module = nn.Linear(
                config.hidden_size, lm_head_vocab_size, bias=False
            )

    def sanitize(self, weights: dict[Any, Any]) -> dict[Any, Any]:
        if self.config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        outputs = self.model(
            inputs=inputs,
            cache=cache,
        )
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(outputs)
        else:
            logits = self.lm_head(outputs)[..., : self.vocab_size]
        return logits

    @property
    def layers(self):
        return self.model.layers.layers
