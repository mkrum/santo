from dataclasses import dataclass
from typing import List

import jax
import jax.nn
import jax.random
import jax.numpy as jnp
from einops import rearrange

from collections import namedtuple

from santo.utils import partial


@dataclass(frozen=True)
class Layer:
    @classmethod
    def initialize(cls, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        ...


@dataclass(frozen=True)
class MLP(Layer):
    in_dim: int
    out_dim: int

    def initialize(self, key):
        # TODO: better initalization
        W = jax.random.normal(key, (self.in_dim, self.out_dim))
        b = jax.random.normal(key, (self.out_dim,))

        params = namedtuple("MLPParams", ["W", "b"])
        return params(W, b)

    @partial
    def __call__(self, W, b, x):
        x_out = jnp.matmul(x, W)
        output = x_out + b
        return output


@dataclass(frozen=True)
class SelfAttention(Layer):
    in_dim: int

    q_dim: int
    v_dim: int

    # TODO: This is not the best way to do this, should use one MLP and chunk
    # the results
    @property
    def q_mlp(self):
        return MLP(self.in_dim, self.q_dim)

    @property
    def k_mlp(self):
        return MLP(self.in_dim, self.q_dim)

    @property
    def v_mlp(self):
        return MLP(self.in_dim, self.v_dim)

    def initialize(self, key):
        q_mlp_params = self.q_mlp.initialize(key)
        k_mlp_params = self.k_mlp.initialize(key)
        v_mlp_params = self.v_mlp.initialize(key)

        params = namedtuple("AttentionParams", ["QMLP", "KMLP", "VMLP"])
        return params(q_mlp_params, k_mlp_params, v_mlp_params)

    @partial
    def __call__(self, q_mlp_params, k_mlp_params, v_mlp_params, x):
        query = self.q_mlp(**q_mlp_params)(x)
        keys = self.k_mlp(**k_mlp_params)(x)
        values = self.v_mlp(**v_mlp_params)(x)

        raw_weights = jnp.matmul(query, rearrange(keys, "b s r -> b r s"))

        weights = jax.nn.softmax(raw_weights / jnp.sqrt(self.q_dim))
        output = jnp.matmul(weights, values)
        return output


@dataclass(frozen=True)
class Embedding(Layer):
    vocab_size: int
    out_dim: int

    def initialize(self, key):
        # TODO: better initalization
        W = jax.random.normal(key, (self.vocab_size, self.out_dim))
        params = namedtuple("EmbeddingParams", ["W"])
        return params(W)

    @partial
    def __call__(self, W, x):
        input_data = jax.nn.one_hot(x, num_classes=self.vocab_size)
        return jnp.matmul(input_data, W)


@dataclass(frozen=True)
class NonParametricLayer(Layer):
    def initialize(self, key):
        return ()


@dataclass(frozen=True)
class ReLU(NonParametricLayer):
    def __call__(self):
        return jax.nn.relu


@dataclass(frozen=True)
class LogSoftmax(NonParametricLayer):
    def __call__(self):
        return jax.nn.log_softmax


@dataclass(frozen=True)
class Sequential(Layer):
    layers: List

    def initialize(self, key):
        # TODO: I think this key needs to be consumed in some way?
        return [l.initialize(key) for l in self.layers]

    @partial
    def __call__(self, params, x):
        for p, l in zip(params, self.layers):
            x = l(*p)(x)
        return x
