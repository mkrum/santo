from dataclasses import dataclass
from typing import List

import jax
import jax.nn
import jax.random
import jax.numpy as jnp


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

        return {"W": W, "b": b}

    def __call__(self, W, b, x):
        x_out = jnp.matmul(x, W)
        output = x_out + b
        return output


@dataclass(frozen=True)
class Embedding(Layer):
    vocab_size: int
    out_dim: int

    def initialize(self, key):
        # TODO: better initalization
        W = jax.random.normal(key, (self.vocab_size, self.out_dim))
        return {"W": W}

    def __call__(self, W, x):
        # TODO:
        input_data = jax.nn.one_hot(x, num_classes=self.vocab_size)
        return jnp.matmul(input_data, W)


@dataclass(frozen=True)
class NonParametricLayer(Layer):
    def initialize(self, key):
        return dict()


@dataclass(frozen=True)
class ReLU(NonParametricLayer):
    def __call__(self, x):
        return jax.nn.relu(x)


@dataclass(frozen=True)
class LogSoftmax(NonParametricLayer):
    def __call__(self, x):
        return jax.nn.log_softmax(x)


@dataclass(frozen=True)
class Sequential(Layer):
    layers: List

    def initialize(self, key):
        # TODO: I think this key needs to be consumed in some way?
        return [l.initialize(key) for l in self.layers]

    def __call__(self, params, inp):
        x = inp
        for p, l in zip(params, self.layers):
            x = l(**p, x=x)
        return x
