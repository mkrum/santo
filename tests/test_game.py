
import glob
import pandas as pd

from santo.parse import parse
from santo.game import GameState
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
    
    all_files = glob.glob("./data/1992/*.EV*")
    for f in all_files:
        games = santo.data.load_evn(f)
    
        for game in games:
            plays = game.get_plays()
    
            state = GameState()
    
            for p in plays:
                state = parse(state, p.play)
    
