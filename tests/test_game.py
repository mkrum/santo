import glob
import pandas as pd

from santo.parse import parse
from santo.game import GameState
from santo.utils import load_game_log
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


def check_correctness(data, game):
    game_entry = data[
        (data["HomeTeam"] == game.home_team)
        & (data["VisitingTeam"] == game.away_team)
        & (data["Date"] == int(game.date.strftime("%Y%m%d")))
        & (data["GameNumber"] == game.game_number)
    ]
    plays = game.get_plays()

    state = GameState()

    current_inning = 0
    for p in plays:
        if p.inning != current_inning:
            current_inning = p.inning
            logging.debug("-" * 10)

        logging.debug(p.play)
        assert p.inning == state.inning
        assert bool(p.is_home_team) == state.home_team_up
        state = parse(state, p.play)
        logging.debug(state)

    sim_outcome = (state.score["home"], state.score["away"])
    real_outcome = (
        int(game_entry["HomeScore"]),
        int(game_entry["VisitingScore"]),
    )
    assert sim_outcome == real_outcome


def test_game_log():
    logging.getLogger().setLevel(logging.DEBUG)

    data = load_game_log("./data/1992/gl1992.txt")
    all_files = glob.glob("./data/1992/*.EV*")

    for f in all_files:
        games = santo.data.load_evn(f)

        for game in games:
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
            try:
                check_correctness(data, game)
            except:
                failure += 1.0
    success_rate = 1.0 - (failure / total)
    print(round(100.0 * success_rate, 2))
