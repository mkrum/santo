from santo.game import GameState
from santo.events import get_valid_advances
import random

state = GameState()

while not state.is_over:
    advances = get_valid_advances(state)
    action = random.randint(0, len(advances) - 1)
    adv = list(advances[action])
    print(adv)
    for a in adv:
        print(state)
        print(a)
        state = a(state)

    if state.inning_is_over():
        state = state.end_inning()
    print(state)
