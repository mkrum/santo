import jax.numpy as jnp
from typing import List
from pathlib import Path
from typing import List

from santo.game import GameState
from santo.parse import parse
import santo.data

BAD_GAMES = [
    "SLA191908190",
    "BRO192107210",
    "NYA195905290",
    "SFN196006080",
    "SFN197008250",
    "KCA197304250",
    "CAL198706090",
    "FLO199705070",
    "KCA200607040",
    "PHI201006250",
    "PHI201006260",
    "PHI201006270",
    "SEA201106240",
    "SEA201106250",
    "SEA201106260",
    "MIL201304190",
    "TBA201505010",
    "TBA201505020",
    "TBA201505030",
    "MIL201709150",
    "MIL201709160",
    "MIL201709170",
]


def rollout(game) -> GameState:
    manfred = game.date.year >= 2020
    state = GameState(manfred=manfred)
    plays = game.get_plays()
    for p in plays:
        state = parse(state, p.play)

    return state


def state_to_vector(state: GameState) -> jnp.array:
    data = list(map(int, list(state.history)))
    return jnp.array(data)


def load_games(path) -> List[jnp.array]:
    games = santo.data.load_evn(path)

    vectors = []
    for g in games:
        # Ignore known bad games and double headers
        if g.game_id in BAD_GAMES or g.game_id[-1] != "0":
            continue

        state = rollout(g)
        vector = state_to_vector(state)
        vectors.append(vector)

    return vectors


class PlayByPlayDataset:
    def __init__(self, years: List[int]):
        self.data = self._load_data(years)

    def _load_data(self, years):
        arrays = []
        for year in years:
            data_path = f"./data/{year}/data.npy"
            arrays.append(jnp.load(data_path))

        padded_width = max([a.shape[1] for a in arrays])
        padded_arrays = []
        for array in arrays:
            padded_arrays.append(
                jnp.pad(
                    array,
                    ((0, 0), (0, padded_width - array.shape[1])),
                    mode="constant",
                    constant_values=-1,
                )
            )

        dataset = jnp.concatenate(padded_arrays, axis=0)
        return dataset

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    """
    Generates the .npy files for the every year
    """
    import glob

    years = list(range(1913, 2023))

    total_tokens = 0
    for year in years:
        print(year)
        all_files = glob.glob(f"./data/{year}/*.EV*")
        all_tokens = []
        for f in all_files:
            tokens = load_games(all_files[0])
            all_tokens += tokens

        max_length = max([len(t) for t in all_tokens])
        data = -1 * jnp.ones((len(all_tokens), max_length))
        for idx, tokens in enumerate(all_tokens):
            padded_tokens = jnp.pad(
                tokens,
                (0, max_length - len(tokens)),
                mode="constant",
                constant_values=-1,
            )
            data = data.at[idx].set(padded_tokens)
        jnp.save(f"./data/{year}/data.npy", data)
