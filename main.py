from dataclasses import dataclass

import jax
import jax.nn
import jax.random
import jax.numpy as jnp

from einops import rearrange

from santo.dataset import PlayByPlayDataset, batch
from santo.updates import TOTAL


def model_step(embedding_matrix, mlp_matrix, input_data, target_one_hot):
    embeddings = jnp.matmul(input_data, embedding_matrix)
    representation = jnp.matmul(embeddings, mlp_matrix)

    loss = compute_loss(representation, target_one_hot, mask)
    return loss


def compute_loss(representation: jnp.array, target_one_hot: jnp.array, mask: jnp.array):
    logits = jax.nn.log_softmax(representation)

    relevant_logits = jnp.sum(logits * target_one_hot, axis=2)
    batch_loss = (relevant_logits * mask).sum(axis=1) / mask.sum(axis=1)
    loss = -1 * batch_loss.mean()
    return loss


vocab_size = len(TOTAL)

key = jax.random.PRNGKey(0)

embedding_matrix = jax.random.normal(key, (vocab_size, 16))
mlp_matrix = jax.random.normal(key, (16, vocab_size))

train_data = PlayByPlayDataset([2013, 2014, 2015])

lr = 1e-2
for data in batch(key, train_data, 8):
    targets = data[:, 1:]
    data = data[:, :-1]
    mask = data != -1
    input_data = jax.nn.one_hot(data, num_classes=vocab_size)
    target_one_hot = jax.nn.one_hot(targets, num_classes=vocab_size)

    loss, (embedding_grad, mlp_grad) = jax.value_and_grad(model_step, (0, 1))(
        embedding_matrix, mlp_matrix, input_data, target_one_hot
    )

    embedding_matrix -= lr * embedding_grad
    mlp_matrix -= lr * mlp_grad

    print(loss)
