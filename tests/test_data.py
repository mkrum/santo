import jax.random

import santo.data
import santo.dataset


def test_evn():
    games = santo.data.load_evn("./data/1992/1992CHN.EVN")


def test_eva():
    games = santo.data.load_evn("./data/1992/1992MIL.EVA")


def test_box_score():
    games = santo.data.load_evn("./data/1992/1992CHN.EVN")


def test_dataset():
    dataset = santo.dataset.PlayByPlayDataset(list(range(1913, 2022)))


def test_batch(batch_size=8):
    dataset = santo.dataset.PlayByPlayDataset([2021])
    key = jax.random.PRNGKey(0)

    # Random numbers to make sure that the drop_last behavoir kicks in
    for batch_size in [2, 3, 8, 9]:
        batched = list(santo.dataset.batch(key, dataset, batch_size, drop_last=False))
        assert sum([b.shape[0] for b in batched]) == len(dataset)

        batched = list(santo.dataset.batch(key, dataset, batch_size, drop_last=True))
        clipped_size = (len(dataset) // batch_size) * batch_size
        assert sum([b.shape[0] for b in batched]) == clipped_size
