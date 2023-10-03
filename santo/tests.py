import logging
from santo.parse import parse
from santo.game import GameState


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
        logging.debug(p.inning == state.inning)
        logging.debug(bool(p.is_home_team) == state.home_team_up)
        assert p.inning == state.inning
        assert bool(p.is_home_team) == state.home_team_up
        state = parse(state, p.play)
        logging.debug(state)

    sim_outcome = (state.score["home"], state.score["away"])
    real_outcome = (
        int(game_entry["HomeScore"]),
        int(game_entry["VisitingScore"]),
    )
    return sim_outcome == real_outcome
