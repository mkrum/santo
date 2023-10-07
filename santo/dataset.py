import jax.numpy as jnp
from typing import List

from santo.game import GameState
from santo.parse import parse
import santo.data


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
        state = rollout(g)
        vector = state_to_vector(state)
        vectors.append(vector)

    return vectors


if __name__ == "__main__":
    import glob

    year = 1992
    all_files = glob.glob(f"./data/{year}/*.EV*")
    total_tokens = 0
    for f in all_files:
        test = load_games(all_files[0])
        total_tokens += sum([len(t) for t in test])
    breakpoint()
