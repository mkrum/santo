from dataclasses import dataclass

import jax
import jax.nn
import jax.random
import jax.numpy as jnp

from santo.model import MLP, SelfAttention, Embedding, Sequential, ReLU, LogSoftmax
from santo.dataset import PlayByPlayDataset, batch
from santo.opt import Adam, SGD
from santo.updates import TOTAL


def compute_loss(params, model, data, targets):
    mask = data != -1

    logits = model(params)(data)

    target_one_hot = jax.nn.one_hot(targets, num_classes=vocab_size)

    relevant_logits = jnp.sum(logits * target_one_hot, axis=2)
    batch_loss = (relevant_logits * mask).sum(axis=1) / mask.sum(axis=1)
    loss = -1 * batch_loss.mean()
    return loss


def eval_model(eval_data, params, model):
    total = 0.0
    total_correct = 0.0
    for data in batch(key, eval_data, 8, drop_last=False):
        targets = data[:, 1:]
        data = data[:, :-1]

        mask = data != -1

        logits = model(params)(data)
        predictions = jnp.argmax(logits, axis=2)

        batch_correct = ((predictions == targets) * mask).sum()
        total_correct += batch_correct
        total += mask.sum()

    print(total_correct / total)
    return total_correct / total


vocab_size = len(TOTAL)

key = jax.random.PRNGKey(0)

layers = Sequential(
    [
        Embedding(vocab_size, 16),
        ReLU(),
        # SelfAttention(16, 16, 16),
        ReLU(),
        MLP(16, vocab_size),
        LogSoftmax(),
    ]
)
params = layers.initialize(key)

opt = SGD(lr=1e-2)
opt_params = SGD.intialize(params)

train_data = PlayByPlayDataset([2013, 2014, 2015])
eval_data = PlayByPlayDataset([2016])

eval_model(eval_data, params, layers)

lr = 1e-2

for data in batch(key, train_data, 8):
    targets = data[:, 1:]
    data = data[:, :-1]

    loss, grads = jax.value_and_grad(compute_loss)(params, layers, data, targets)
    params, opt_params = opt.step(opt_params)(params, grads)

    print(loss)

eval_model(eval_data, params, layers)
