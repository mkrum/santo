from santo.game import GameState
from santo.events import get_valid_advances
from santo.updates import PlayBreak, InningBreak
import random

state = GameState()

while not state.is_over:
    advances = get_valid_advances(state)
    action = random.randint(0, len(advances) - 1)
    adv = list(advances[action])

    state = state.add_history(PlayBreak())

    for a in adv:
        state = a(state)

    if state.inning_is_over():
        state = state.add_history(InningBreak())
        state = state.end_inning()

print(state.history)
print(len(list(map(int, list(state.history)))))
