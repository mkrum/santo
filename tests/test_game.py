import glob
import pandas as pd

from santo.parse import parse
from santo.game import GameState
from santo.utils import load_game_log
from santo.tests import check_correctness
import santo.data
import logging


def test_adding_outs():
    state = GameState()
    state = state.add_out()


def test_strikeout():
    state = GameState()
    state = parse(state, "K")
    state = parse(state, "K")
    state = parse(state, "K")


def test_game_log():
    data = load_game_log("./data/1992/gl1992.txt")
    all_files = glob.glob("./data/1992/*.EV*")

    for f in all_files:
        games = santo.data.load_evn(f)

        for game in games:
            try:
                assert check_correctness(data, game)
            except:
                logging.getLogger().setLevel(logging.DEBUG)
                print(game.game_id)
                exit()
                check_correctness(data, game)


def test_game_log_percent():
    data = load_game_log("./data/1992/gl1992.txt")
    all_files = glob.glob("./data/1992/*.EV*")

    total = 0.0
    failure = 0.0

    for f in all_files:
        games = santo.data.load_evn(f)

        total += len(games)

        for game in games:
            if not check_correctness(data, game):
                failure += 1.0

    success_rate = 1.0 - (failure / total)
    print(round(100.0 * success_rate, 2))
