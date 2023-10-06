import glob
import logging

import pytest

from santo.utils import load_game_log
from santo.tests import check_correctness
import santo.data

logging.disable(level=logging.ERROR)


@pytest.mark.parametrize(
    "year", [1922, 1932, 1942, 1952, 1962, 1972, 1982, 1992, 2012, 2022]
)
def test_game_log_percent(year):
    print(year)
    data = load_game_log(f"./data/{year}/gl{year}.txt")
    all_files = glob.glob(f"./data/{year}/*.EV*")

    total = 0.0
    failure = 0.0

    for f in all_files:
        games = santo.data.load_evn(f)

        total += len(games)

        for game in games:
            if not check_correctness(data, game):
                failure += 1.0
                print(game.game_id)

    success_rate = 1.0 - (failure / total)
    print(round(100.0 * success_rate, 2))
    assert success_rate > 0.99
