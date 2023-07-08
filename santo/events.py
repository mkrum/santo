from santo.utils import Base
from santo.game import GameState
from dataclasses import dataclass
from typing import List, Set


@dataclass(frozen=True)
class RunnerAdvance:
    from_base: Base
    to_base: Base
    is_out: bool
    explicit: bool = False

    @classmethod
    def from_string(cls, raw_string: str) -> "RunnerAdvance":
        from_base = Base.from_short_string(raw_string[0])
        to_base = Base.from_short_string(raw_string[2])
        is_out = raw_string[1] != "-"
        return cls(from_base, to_base, is_out, explicit=True)

    def __call__(self, state: GameState) -> GameState:
        new_state = state
        if self.is_out:
            new_state = new_state.add_out(self.from_base)
        else:
            new_state = state.remove_runner(self.from_base)
            new_state = new_state.add_runner(self.to_base)
        return new_state


def simplify_runner_advances(advances: List[RunnerAdvance]) -> List[RunnerAdvance]:
    """
    Events sometimes overspecify runners advancing, so it is possible to end up
    with two records for a single advance. For example, SB2.1-3(E2) is a runner
    stealing 2nd (1-2) followed immediately by a throwing error, allowing the
    runner to advance to third (1-3). We only want to apply the 1-3 in this
    situation.
    """
    unique_bases = {}
    for a in advances:
        # Find all of the movements for the same runner
        current = unique_bases.get(a.from_base, [])
        current.append(a)
        unique_bases[a.from_base] = current

    simplified_advances = []
    for k, same_advances in unique_bases.items():
        # Some cases an event might lead us to believe a runner is out, when
        # they are safe. In these cases, we default to the explicit runner
        # advancement notations
        all_implicit = all([not x.explicit for x in same_advances])

        if not all_implicit:
            same_advances = list(filter(lambda x: x.explicit, same_advances))

        same_advances = list(
            sorted(same_advances, key=lambda x: x.to_base.value, reverse=True)
        )
        simplified_advances.append(same_advances[0])

    # Move lead runners first
    ordered_advances = list(
        sorted(simplified_advances, key=lambda x: x.from_base.value, reverse=True)
    )
    return ordered_advances


@dataclass(frozen=True)
class Event:
    raw_string: str

    def get_runners(self) -> List[str]:
        if not ("." in self.raw_string):
            return []

        runners = self.raw_string.split(".")[-1].split(";")
        advances = list(map(RunnerAdvance.from_string, runners))
        return advances

    def handle_runners(
        self, state, other_runners: List[RunnerAdvance] = None
    ) -> GameState:
        new_state = state

        runners = self.get_runners()

        if other_runners:
            runners = simplify_runner_advances(runners + other_runners)

        for advance_runners in runners:
            new_state = advance_runners(new_state)
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
    raw_string: str = ""

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
        new_state = state.add_out(Base.BATTER)

        if len(self.raw_string) > 1 and self.raw_string[1] == "+":
            other_event = self.raw_string.split("+")[1]
            if other_event[:2] == "CS":
                other_event = CaughtStealingEvent.from_string(other_event)
            return other_event(new_state)
        else:
            return self.handle_runners(new_state)


@dataclass(frozen=True)
class OutEvent(Event):
    def get_players_out(self) -> List[RunnerAdvance]:
        players_out = set([RunnerAdvance(Base.BATTER, Base.FIRST, True)])

        if "(" in self.raw_string:
            # TODO: do this right
            marked = list(
                map(lambda x: x.split(")")[0], self.raw_string.split("(")[1:])
            )

            # Sometimes these also notify whether the run was earned or not
            marked = list(filter(lambda x: not (x in ["RBI", "UR"]), marked))
            runners = list(map(Base.from_short_string, marked))

            for r in runners:
                players_out.add(RunnerAdvance(r, r.next_base(), True))

        else:
            players_out.add(RunnerAdvance(Base.BATTER, Base.FIRST, True))

        return list(players_out)

    def __call__(self, state: GameState) -> GameState:
        players_out = self.get_players_out()
        return self.handle_runners(state, players_out)


@dataclass(frozen=True)
class WalkEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        return state.force_advance(Base.BATTER)


@dataclass(frozen=True)
class HitByPitchEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        return state.force_advance(Base.BATTER)


@dataclass(frozen=True)
class HitEvent(Event):
    @property
    def hit_type(self):
        return self.raw_string[0]

    @classmethod
    def from_string(cls, string: str) -> "OutEvent":
        return cls(string)

    def __call__(self, state: GameState) -> GameState:
        hit_type = self.hit_type
        if hit_type == "S":
            advance = RunnerAdvance(Base.BATTER, Base.FIRST, False)
        elif hit_type == "D":
            advance = RunnerAdvance(Base.BATTER, Base.SECOND, False)
        elif hit_type == "T":
            advance = RunnerAdvance(Base.BATTER, Base.THIRD, False)
        return self.handle_runners(state, [advance])


@dataclass(frozen=True)
class HomeRunEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        advance = RunnerAdvance(Base.BATTER, Base.HOME, False)
        return self.handle_runners(state, [advance])


@dataclass(frozen=True)
class PassedBallEvent(Event):
    ...


@dataclass(frozen=True)
class ErrorEvent(Event):
    ...


@dataclass(frozen=True)
class IntentionalWalkEvent(WalkEvent):
    ...


@dataclass(frozen=True)
class CaughtStealingEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        stolen_base = self.raw_string[2]

        if stolen_base == "2":
            advance = RunnerAdvance(Base.FIRST, Base.SECOND, True)
        elif stolen_base == "3":
            advance = RunnerAdvance(Base.SECOND, Base.THIRD, True)
        elif stolen_base == "H":
            advance = RunnerAdvance(Base.THIRD, Base.HOME, True)

        return self.handle_runners(state, [advance])


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


@dataclass(frozen=True)
class StolenBaseEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        stolen_base = self.raw_string[2]

        if stolen_base == "2":
            advance = RunnerAdvance(Base.FIRST, Base.SECOND, False)
        elif stolen_base == "3":
            advance = RunnerAdvance(Base.SECOND, Base.THIRD, False)
        elif stolen_base == "H":
            advance = RunnerAdvance(Base.THIRD, Base.HOME, False)

        return self.handle_runners(state, [advance])
