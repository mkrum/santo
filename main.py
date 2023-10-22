from dataclasses import dataclass

import jax
import jax.nn
import jax.random
import jax.numpy as jnp

from einops import rearrange

from santo.dataset import PlayByPlayDataset, batch
from santo.updates import TOTAL


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

    def __call__(self, x, W, b):
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

    def __call__(self, x, W):
        return jnp.matmul(x, W)


def model_step(params, layers, input_data, target_one_hot):
    representation = forward(params, layers, input_data)

    loss = compute_loss(representation, target_one_hot, mask)
    return loss


def compute_loss(representation: jnp.array, target_one_hot: jnp.array, mask: jnp.array):
    logits = jax.nn.log_softmax(representation)

    relevant_logits = jnp.sum(logits * target_one_hot, axis=2)
    batch_loss = (relevant_logits * mask).sum(axis=1) / mask.sum(axis=1)
    loss = -1 * batch_loss.mean()
    return loss


def eval_model(eval_data, params, layers):
    total = 0.0
    total_correct = 0.0
    for data in batch(key, eval_data, 8, drop_last=False):
        targets = data[:, 1:]
        data = data[:, :-1]
        mask = data != -1

        input_data = jax.nn.one_hot(data, num_classes=vocab_size)
        target_one_hot = jax.nn.one_hot(targets, num_classes=vocab_size)

        representation = forward(params, layers, input_data)

        logits = jax.nn.log_softmax(representation)
        predictions = jnp.argmax(logits, axis=2)

        mask = data != -1

        batch_correct = ((predictions == targets) * mask).sum()
        total_correct += batch_correct
        total += mask.sum()

    print(total_correct / total)
    return total_correct / total


vocab_size = len(TOTAL)

key = jax.random.PRNGKey(0)


def forward(params, layers, inp):
    x = inp
    for p, l in zip(params, layers):
        x = l(x, **p)

    return x


layers = [Embedding(vocab_size, 16), MLP(16, vocab_size)]
params = [l.initialize(key) for l in layers]

train_data = PlayByPlayDataset([2013, 2014, 2015])
eval_data = PlayByPlayDataset([2016])

eval_model(eval_data, params, layers)

lr = 1e-2
for data in batch(key, train_data, 8):
    targets = data[:, 1:]
    data = data[:, :-1]
    mask = data != -1

    input_data = jax.nn.one_hot(data, num_classes=vocab_size)
    target_one_hot = jax.nn.one_hot(targets, num_classes=vocab_size)

    loss, grads = jax.value_and_grad(model_step)(
        params, layers, input_data, target_one_hot
    )

    for p, g in zip(params, grads):
        for k in p.keys():
            p[k] -= lr * g[k]

    print(loss)

eval_model(eval_data, params, layers)
