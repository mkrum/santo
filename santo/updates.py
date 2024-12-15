import re
from typing import List, Set, Any
from dataclasses import dataclass

from itertools import product
from santo.utils import Base
from santo.game import GameState

POSSIBLE_ADVANCES = list(
    filter(lambda x: x[0] <= x[1], product(list(Base), list(Base), [True, False]))
)

EVENTS = [
    "IdentityEvent",
    "OutEvent",
    "StrikeOutEvent",
    "WalkEvent",
    "HitEvent",
    "SingleEvent",
    "DoubleEvent",
    "TrippleEvent",
    "StolenBaseEvent",
    "CaughtStealingEvent",
    "PickedOffEvent",
    "FieldersChoiceEvent",
    "CatcherInterferenceEvent",
    "PickedOffCaughtStealingEvent",
    "HitByPitchEvent",
    "HomeRunEvent",
    "ErrorEvent",
    "DefensiveIndifferenceEvent",
    "OtherAdvanceEvent",
    "WildPitchEvent",
    "BalkEvent",
    "RunnersAdvanceEvent",
    "FoulBallErrorEvent",
    "IntentionalWalkEvent",
    "SecondaryErrorEvent",
    "PassedBallEvent",
]

BREAKS = [
    "InningBreak",
    "GameBreak",
]

TOTAL = POSSIBLE_ADVANCES + EVENTS + BREAKS


@dataclass(frozen=True)
class Update:
    def __int__(self):
        ...

    @classmethod
    def from_int(cls, idx):
        ...


@dataclass(frozen=True)
class RunnerAdvance(Update):
    """
    This represents the atomic unit of change for a game state. It represents
    both players moving between bases and players being called out.
    """

    from_base: Base
    to_base: Base
    is_out: bool
    explicit: bool = False
    modifications: Any = None

    @classmethod
    def from_string(cls, raw_string: str) -> "RunnerAdvance":
        from_base = Base.from_short_string(raw_string[0])
        to_base = Base.from_short_string(raw_string[2])

        is_out = raw_string[1] != "-"

        # This is a little complicated, but if there is a advancment like
        # 2X3(E8) the player is safe, even though the advance is marked with an
        # X. However! 2X3(E8)(865) is an out, since there was an error, but then
        # a successful out.
        modifer_strings = re.findall(r"\((.*?)\)", raw_string)
        if any([("E" in m) for m in modifer_strings]):
            additional_out_string = re.findall(r"\((\d*?)\)", raw_string)
            is_out = len(additional_out_string) != 0

        return cls(from_base, to_base, is_out, explicit=True)

    def is_valid(self):
        return from_base >= to_base

    def __len__(self):
        return self.to_base.value - self.from_base.value

    def __call__(self, state: GameState) -> GameState:
        return state.apply(self)

    def __int__(self):
        return POSSIBLE_ADVANCES.index((self.from_base, self.to_base, self.is_out))

    @classmethod
    def from_int(cls, idx):
        (from_base, to_base, is_out) = POSSIBLE_ADVANCES[idx]
        return cls(from_base, to_base, is_out)


@dataclass(frozen=True)
class PlayBreak(Update):
    event: str

    def __int__(self):
        return TOTAL.index(self.event)

    @classmethod
    def from_int(cls, idx):
        event_type = TOTAL[idx]
        return cls(event_type)


@dataclass(frozen=True)
class InningBreak(Update):
    def __int__(self):
        return TOTAL.index("InningBreak")

    @classmethod
    def from_int(cls, idx):
        assert idx == TOTAL.index("InningBreak")
        return cls()


@dataclass(frozen=True)
class GameBreak(Update):
    def __int__(self):
        return TOTAL.index("GameBreak")

    @classmethod
    def from_int(cls, idx):
        assert idx == TOTAL.index("GameBreak")
        return cls()
