
from santo.utils import BATTING_ZONES
from santo.game import GameState
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Event:

    raw_string: str

    def get_runners(self) -> List[str]:

        if not ("." in self.raw_string):
            return []

        runners = self.raw_string.split(".")[-1].split(";")
        return runners

    def handle_runners(self, state) -> GameState:
        new_state = state
        for r in self.get_runners():
            new_state = new_state.move_runners(r)
        return new_state

    @classmethod
    def from_string(cls, play_string: str) -> "Event":
        return cls(play_string)

    def __call__(self, state: GameState) -> GameState:
        return self.handle_runners(state)

    def __add__(self, other_event):
        return UnionEvent(self, other_event)

@dataclass(frozen=True)
class IdentityEvent(Event):
    raw_string:str = ""
    def __call__(self, state: GameState) -> GameState:
        return state

@dataclass(frozen=True)
class UnionEvent:
    event_one: Event
    event_two: Event

    def __call__(self, state: GameState) -> GameState:

        return event_two(event_one(state))

@dataclass(frozen=True)
class StrikeOutEvent(Event):

    def __call__(self, state: GameState) -> GameState:
        return state.add_out()

@dataclass(frozen=True)
class OutEvent(Event):

    def __call__(self, state: GameState) -> GameState:
        return state.add_out()

@dataclass(frozen=True)
class WalkEvent(Event):

    @classmethod
    def from_string(cls, string: str) -> "OutEvent":
        return cls(string)

    def __call__(self, state: GameState) -> GameState:
        return state.force_advance()

@dataclass(frozen=True)
class HitEvent(Event):
    
    @property
    def hit_type(self): 
        return self.raw_string[0]

    @classmethod
    def from_string(cls, string: str) -> "OutEvent":
        return cls(string)

    def __call__(self, state: GameState) -> GameState:
        bags = None
        if self.hit_type == 'S':
            bags = 1
        elif self.hit_type == 'D':
            bags = 2
        elif self.hit_type == 'T':
            bags = 3

        return state.add_runner(bags)

@dataclass(frozen=True)
class HomeRunEvent(Event):

    def __call__(self, state: GameState) -> GameState:
        return state.home_run()

@dataclass(frozen=True)
class PassedBallEvent(Event):
    ...

@dataclass(frozen=True)
class ErrorEvent(Event):
    ...

@dataclass(frozen=True)
class IntentionalWalkEvent(Event):
    ...

@dataclass(frozen=True)
class CaughtStealingEvent(Event):
    ...

@dataclass(frozen=True)
class PickedOffEvent(Event):
    ...

@dataclass(frozen=True)
class BalkEvent(Event):
    ...

@dataclass(frozen=True)
class RunnersAdvanceEvent(Event):
    ...

@dataclass(frozen=True)
class FoulBallErrorEvent(Event):
    ...

@dataclass(frozen=True)
class FieldersChoiceEvent(Event):
    ...

@dataclass(frozen=True)
class PickedOffCaughtStealingEvent(Event):
    ...
