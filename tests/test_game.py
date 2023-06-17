
import pandas as pd

from santo.game import GameState, parse
from santo.utils import load_game_log
import santo.data

def test_adding_outs():
    state = GameState()
    print(state)
    state = state.add_out()
    print(state)

def test_strikeout():
    state = GameState()
    state = parse(state, "K")
    state = parse(state, "K")
    state = parse(state, "K")
    print(state)

def test_game_log():
    data = load_game_log("./data/1992/gl1992.txt")
    games = santo.data.load_evn("./data/1992/1992CHN.EVN")
    game = games[0]
    plays = game.get_plays()

    state = GameState()

    for p in plays:
        state = parse(state, p.play)

    print(state)
    
