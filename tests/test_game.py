import glob
import logging

import pytest

from santo.utils import load_game_log
from santo.tests import check_correctness
import santo.data

logging.disable(level=logging.ERROR)

years = list(range(1913, 2023))
years.remove(2020)


@pytest.mark.parametrize("year", years)
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
            # Double headers are broken...
            if game.game_id[-1] != "0":
                continue

            if not check_correctness(data, game):
                failure += 1.0
                print(game.game_id)

    success_rate = 1.0 - (failure / total)
    print(round(100.0 * success_rate, 2))
    assert success_rate > 0.99
