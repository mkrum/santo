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

    manfred = game.date.year >= 2020
    state = GameState(manfred=manfred)

    current_inning = 0
    for p in plays:
        if p.inning != current_inning:
            current_inning = p.inning
            logging.debug("-" * 10)

        logging.debug(p.play)
        logging.debug(f"Innings match {p.inning == state.inning}")
        logging.debug(f"At bat matches {bool(p.is_home_team) == state.home_team_up}")

        if p.inning != state.inning:
            return False
        if bool(p.is_home_team) != state.home_team_up:
            return False

        try:
            state = parse(state, p.play)
        except AssertionError as e:
            logging.exception(e)
            return False

        logging.debug(state)

    sim_outcome = (state.score["home"], state.score["away"])
    real_outcome = (
        int(game_entry["HomeScore"]),
        int(game_entry["VisitingScore"]),
    )
    return sim_outcome == real_outcome
