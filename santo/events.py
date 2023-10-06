import re
from typing import List, Set
from dataclasses import dataclass

from santo.utils import Base
from santo.game import GameState


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

        modifer_strings = re.findall(r"\((.*?)\)", raw_string)

        if any([("E" in m) for m in modifer_strings]):
            is_out = False

        return cls(from_base, to_base, is_out, explicit=True)

    def __call__(self, state: GameState) -> GameState:
        new_state = state

        # Sometimes more than three runners are recorded as out. In this case,
        # we want to just stop counting to not trigger any errors
        if new_state.outs == 3:
            return new_state

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
        # A K is not always an out! A batter can advance to fist on a wild pitch
        new_state = state
        if not any([r.from_base == Base.BATTER for r in self.get_runners()]):
            new_state = new_state.add_out(Base.BATTER)

        if len(self.raw_string) > 1 and self.raw_string[1] == "+":
            other_event = self.raw_string.split("+")[1]

            assert other_event[:2] in [
                "CS",
                "SB",
                "WP",
                "PB",
                "OA",
                "PO",
            ], f"Unkown strikeout event {other_event}"

            if other_event[:2] == "CS":
                other_event = CaughtStealingEvent.from_string(other_event)
            elif other_event[:2] == "SB":
                other_event = StolenBaseEvent.from_string(other_event)
            elif other_event[:2] == "WP":
                other_event = WildPitchEvent.from_string(other_event)
            elif other_event[:2] == "PB":
                other_event = PassedBallEvent.from_string(other_event)
            elif other_event[:2] == "OA":
                other_event = OtherAdvanceEvent.from_string(other_event)
            elif other_event[:2] == "PO":
                other_event = PickedOffEvent.from_string(other_event)

            return other_event(new_state)
        else:
            return self.handle_runners(new_state)


@dataclass(frozen=True)
class OutEvent(Event):
    def get_players_out(self) -> List[RunnerAdvance]:
        players_out = set()
        play_string = self.raw_string.split("/")[0]

        marked = re.findall(r"\((.*?)\)", play_string)

        # Sometimes these also notify whether the run was earned or not or
        # have other, strange markings
        marked = list(filter(lambda x: (x in ["B", "1", "2", "3", "H"]), marked))

        runners = list(map(Base.from_short_string, marked))

        for r in runners:
            players_out.add(RunnerAdvance(r, r.next_base(), True))

        # If the string ends in a number and no marking, it is assumed to be the
        # Batter. For example, 64(1)3 is two outs, not one.
        if play_string[-1] != ")":
            players_out.add(RunnerAdvance(Base.BATTER, Base.FIRST, True))

        return list(players_out)

    def __call__(self, state: GameState) -> GameState:
        players_out = self.get_players_out()

        # If the batter is not listed as being out, it is implicitily a single
        if not any([r.from_base == Base.BATTER for r in players_out]):
            players_out.append(RunnerAdvance(Base.BATTER, Base.FIRST, False))

        return self.handle_runners(state, players_out)


@dataclass(frozen=True)
class WalkEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        new_state = state

        if len(self.raw_string) > 1 and self.raw_string[1] == "+":
            new_state = state.force_advance(Base.BATTER)

            other_event = self.raw_string.split("+")[1]

            assert other_event[:2] in [
                "CS",
                "SB",
                "WP",
                "PB",
                "OA",
            ], f"Unkown walk event {other_event}"

            if other_event[:2] == "CS":
                other_event = CaughtStealingEvent.from_string(other_event)
            elif other_event[:2] == "SB":
                other_event = StolenBaseEvent.from_string(other_event)
            elif other_event[:2] == "WP":
                other_event = WildPitchEvent.from_string(other_event)
            elif other_event[:2] == "PB":
                other_event = PassedBallEvent.from_string(other_event)
            elif other_event[:2] == "OA":
                other_event = OtherAdvanceEvent.from_string(other_event)

            return other_event(new_state)
        else:
            new_state = self.handle_runners(new_state)
            return new_state.force_advance(Base.BATTER)


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
    def from_string(cls, string: str) -> "HitEvent":
        return cls(string)

    def __call__(self, state: GameState) -> GameState:
        hit_type = self.hit_type
        assert hit_type in ["S", "D", "T"], f"Unkown hit type {hit_type}"
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
    def __call__(self, state: GameState) -> GameState:
        pickoff_base = self.raw_string[2]

        if pickoff_base == "1":
            advance = RunnerAdvance(Base.FIRST, Base.FIRST, True)
        elif pickoff_base == "2":
            advance = RunnerAdvance(Base.SECOND, Base.SECOND, True)
        elif pickoff_base == "3":
            advance = RunnerAdvance(Base.THIRD, Base.THIRD, True)
        return self.handle_runners(state, [advance])


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
    def __call__(self, state: GameState) -> GameState:
        # FC/SH indicates a FC sacrifice HIT, so there is an implict B-1
        advance = []
        if "FC/SH" == self.raw_string[:5]:
            advance = [RunnerAdvance(Base.BATTER, Base.FIRST, False)]

        return self.handle_runners(state, advance)


@dataclass(frozen=True)
class PickedOffCaughtStealingEvent(Event):
    ...


@dataclass(frozen=True)
class WildPitchEvent(Event):
    ...


@dataclass(frozen=True)
class StolenBaseEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        sb_strings = self.raw_string.split(";")

        advances = []
        for sb_string in sb_strings:
            stolen_base = sb_string[2]

            if stolen_base == "2":
                advance = RunnerAdvance(Base.FIRST, Base.SECOND, False)
            elif stolen_base == "3":
                advance = RunnerAdvance(Base.SECOND, Base.THIRD, False)
            elif stolen_base == "H":
                advance = RunnerAdvance(Base.THIRD, Base.HOME, False)

            advances.append(advance)

        return self.handle_runners(state, advances)


@dataclass(frozen=True)
class DefensiveIndifferenceEvent(Event):
    ...


@dataclass(frozen=True)
class OtherAdvanceEvent(Event):
    ...


@dataclass(frozen=True)
class CatcherInterferenceEvent(Event):
    ...
