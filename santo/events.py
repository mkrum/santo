import re
from typing import List, Set
from itertools import product
from dataclasses import dataclass

from santo.utils import Base
from santo.game import GameState


@dataclass(frozen=True)
class RunnerAdvance:
    """
    This represents the atomic unit of change for a game state. It represents
    both players moving between bases and players being called out.
    """

    from_base: Base
    to_base: Base
    is_out: bool
    explicit: bool = False

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

    def __len__(self):
        return self.to_base.value - self.from_base.value

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


def get_valid_advances(state):
    base_runners = state.get_base_runners()

    possible = {b: [] for b in base_runners}
    for base_runner in base_runners:
        forward_bases = list(filter(lambda b: b > base_runner, Base))
        for fb in forward_bases:
            possible[base_runner].append(RunnerAdvance(base_runner, fb, True))
            possible[base_runner].append(RunnerAdvance(base_runner, fb, False))

    possible_combos = product(*(possible[b] for b in base_runners))

    # Make sure the locations they are going to are unique
    valid_combos = []
    for advances in possible_combos:
        to_locations = [a.to_base for a in advances]

        # if all memembers are unique
        if len(to_locations) == len(set(to_locations)):
            valid_combos.append(advances)

    unique_possible_combos = set(
        map(lambda x: frozenset(simplify_runner_advances(x)), valid_combos)
    )
    # Re-sort after dedupe
    unique_possible_combos = [
        list(sorted(s, key=lambda x: x.from_base.value, reverse=True))
        for s in unique_possible_combos
    ]
    return list(unique_possible_combos)


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
        # Runners make sure runners are sorted, can get messed up sometimes
        runners = sorted(runners, key=lambda k: k.from_base.value, reverse=True)

        if other_runners:
            runners = simplify_runner_advances(runners + other_runners)

        assert self.__class__.are_valid(runners)

        for advance_runners in runners:
            new_state = advance_runners(new_state)

        return new_state

    @classmethod
    def from_string(cls, play_string: str) -> "Event":
        return cls(play_string)

    def __call__(self, state: GameState) -> GameState:
        """
        By Default, an event will just look at the notation for advancing
        runners, and and advance those runners.
        """
        return self.handle_runners(state)

    def has_error(self):
        # I think this works?
        return "E" in self.raw_string

    @classmethod
    def are_valid(cls, advancements):
        return True


@dataclass(frozen=True)
class IdentityEvent(Event):
    raw_string: str = ""

    def __call__(self, state: GameState) -> GameState:
        return state


@dataclass(frozen=True)
class OutEvent(Event):
    def get_players_out(self) -> List[RunnerAdvance]:
        players_out = set()

        play_string = self.raw_string.split("/")[0].split(".")[0]

        marked = re.findall(r"\((.*?)\)", play_string)

        # Sometimes these also notify whether the run was earned or not or
        # have other, strange markings
        marked = list(filter(lambda x: (x in ["B", "1", "2", "3", "H"]), marked))

        runners = list(map(Base.from_short_string, marked))

        for r in runners:
            players_out.add(RunnerAdvance(r, r.next_base(), True))

        # If the string ends in a number and no marking, it is assumed to be the
        # Batter. For example, 64(1)3 is two outs, not one.
        if play_string[-1] != ")" and not ("E" in play_string):
            players_out.add(RunnerAdvance(Base.BATTER, Base.FIRST, True))

        return list(players_out)

    def __call__(self, state: GameState) -> GameState:
        players_out = self.get_players_out()

        # If the batter is not listed as being out, it is implicitily a single
        if not any([r.from_base == Base.BATTER for r in players_out]):
            players_out.append(RunnerAdvance(Base.BATTER, Base.FIRST, False))

        return self.handle_runners(state, players_out)

    @classmethod
    def are_valid(cls, advancements):
        out_events = list(filter(lambda x: x.is_out, advancements))
        return len(out_events) > 0


@dataclass(frozen=True)
class StrikeOutEvent(Event):
    def _handle_plus(self):
        if len(self.raw_string) > 1 and self.raw_string[1] == "+":
            other_event_str = self.raw_string.split("+")[1]

            if other_event_str[:2] == "CS":
                other_event = CaughtStealingEvent.from_string(other_event_str)
            elif other_event_str[:2] == "SB":
                other_event = StolenBaseEvent.from_string(other_event_str)
            elif other_event_str[:2] == "WP":
                other_event = WildPitchEvent.from_string(other_event_str)
            elif other_event_str[:2] == "PB":
                other_event = PassedBallEvent.from_string(other_event_str)
            elif other_event_str[:2] == "OA":
                other_event = OtherAdvanceEvent.from_string(other_event_str)
            elif other_event_str[:4] == "POCS":
                other_event = PickedOffCaughtStealingEvent.from_string(other_event_str)
            elif other_event_str[:2] == "PO":
                other_event = PickedOffEvent.from_string(other_event_str)
            elif other_event_str[:2] == "DI":
                other_event = DefensiveIndifferenceEvent.from_string(other_event_str)
            elif other_event_str[0] == "E":
                other_event = SecondaryErrorEvent.from_string(other_event_str)
            else:
                # I am using an "assert False" here, since I think we are only
                # catching Assertion errors to differentiate between parsing
                # errors and programmatic errors.
                assert False, f"unknown event {other_event}"

            return other_event

    def __call__(self, state: GameState) -> GameState:
        # A K is not always an out! A batter can advance to fist on a wild pitch
        new_state = state
        if not any([r.from_base == Base.BATTER for r in self.get_runners()]):
            new_state = new_state.add_out(Base.BATTER)

        if len(self.raw_string) > 1 and self.raw_string[1] == "+":
            other_event = self._handle_plus()
            return other_event(new_state)
        else:
            return self.handle_runners(new_state)


@dataclass(frozen=True)
class WalkEvent(StrikeOutEvent):
    def __call__(self, state: GameState) -> GameState:
        new_state = state

        if len(self.raw_string) > 1 and self.raw_string[1] == "+":
            other_event = self._handle_plus()
            new_state = other_event(new_state)
            runners = other_event.get_runners()

            if any([r.from_base == Base.BATTER for r in runners]):
                return new_state
            else:
                return new_state.force_advance(Base.BATTER)
        else:
            new_state = self.handle_runners(new_state)
            if any([r.from_base == Base.BATTER for r in self.get_runners()]):
                return new_state
            else:
                return new_state.force_advance(Base.BATTER)


@dataclass(frozen=True)
class HitEvent(Event):
    @property
    def hit_type(self):
        return self.raw_string[0]

    @classmethod
    def from_string(cls, string: str) -> "HitEvent":
        hit_type = string[0]
        assert hit_type in ["S", "D", "T"], f"Unkown hit type {hit_type}"
        if hit_type == "S":
            return SingleEvent(string)
        elif hit_type == "D":
            return DoubleEvent(string)
        elif hit_type == "T":
            return TrippleEvent(string)

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

    def is_hit_type(self, advancement):
        ...

    @classmethod
    def are_valid(cls, advancements):
        hit_moves = list(filter(cls.is_hit_type, advancements))
        return len(hit_moves) > 0


@dataclass(frozen=True)
class SingleEvent(HitEvent):
    @classmethod
    def is_hit_type(cls, advancement):
        return (
            advancement.from_base == Base.BATTER and advancement.to_base >= Base.FIRST
        )


@dataclass(frozen=True)
class DoubleEvent(HitEvent):
    @classmethod
    def is_hit_type(self, advancement):
        return (
            advancement.from_base == Base.BATTER and advancement.to_base >= Base.SECOND
        )


@dataclass(frozen=True)
class TrippleEvent(HitEvent):
    @classmethod
    def is_hit_type(self, advancement):
        return (
            advancement.from_base == Base.BATTER and advancement.to_base >= Base.THIRD
        )


@dataclass(frozen=True)
class StolenBaseEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        sb_strings = self.raw_string.split(";")

        advances = []
        for sb_string in sb_strings:
            # If an error happends, runner advancements will be listed normally
            if sb_string[:2] != "SB":
                continue

            stolen_base = sb_string[2]

            if stolen_base == "2":
                advance = RunnerAdvance(Base.FIRST, Base.SECOND, False)
            elif stolen_base == "3":
                advance = RunnerAdvance(Base.SECOND, Base.THIRD, False)
            elif stolen_base == "H":
                advance = RunnerAdvance(Base.THIRD, Base.HOME, False)
            advances.append(advance)

        return self.handle_runners(state, advances)

    @classmethod
    def are_valid(cls, advancements):
        singlebaseattempt = list(filter(lambda x: len(x) >= 1, advancements))
        return len(singlebaseattempt) > 0


@dataclass(frozen=True)
class CaughtStealingEvent(Event):
    def _get_base(self, string):
        stolen_base = string[2]
        return stolen_base

    def __call__(self, state: GameState) -> GameState:
        cs_strings = list(filter(lambda x: "CS" in x, self.raw_string.split(";")))

        new_state = state
        advances = []
        for cs_string in cs_strings:
            stolen_base = self._get_base(cs_string)

            modifer_strings = re.findall(r"\((.*?)\)", cs_string)
            is_out = True

            if any([("E" in m) for m in modifer_strings]):
                additional_out_string = re.findall(r"\((\d*?)\)", self.raw_string)
                is_out = len(additional_out_string) != 0

            if stolen_base == "2":
                advances.append(RunnerAdvance(Base.FIRST, Base.SECOND, is_out))
            elif stolen_base == "3":
                advances.append(RunnerAdvance(Base.SECOND, Base.THIRD, is_out))
            elif stolen_base == "H":
                advances.append(RunnerAdvance(Base.THIRD, Base.HOME, is_out))

        new_state = self.handle_runners(new_state, advances)
        return new_state


@dataclass(frozen=True)
class PickedOffEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        pickoff_base = self.raw_string[2]

        play_string = self.raw_string.split("/")[0].split(".")[0]

        if "E" in play_string:
            return self.handle_runners(state)
        elif pickoff_base == "1":
            advance = RunnerAdvance(Base.FIRST, Base.FIRST, True)
        elif pickoff_base == "2":
            advance = RunnerAdvance(Base.SECOND, Base.SECOND, True)
        elif pickoff_base == "3":
            advance = RunnerAdvance(Base.THIRD, Base.THIRD, True)

        return self.handle_runners(state, [advance])


@dataclass(frozen=True)
class FieldersChoiceEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        # FC indicates an implict B-1
        advance = [RunnerAdvance(Base.BATTER, Base.FIRST, False)]
        return self.handle_runners(state, advance)


@dataclass(frozen=True)
class CatcherInterferenceEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        advance = [RunnerAdvance(Base.BATTER, Base.FIRST, False)]
        return self.handle_runners(state, advance)


@dataclass(frozen=True)
class PickedOffCaughtStealingEvent(CaughtStealingEvent):
    def _get_base(self, string):
        stolen_base = string[4]
        return stolen_base


@dataclass(frozen=True)
class HitByPitchEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        return state.force_advance(Base.BATTER)


@dataclass(frozen=True)
class HomeRunEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        advance = RunnerAdvance(Base.BATTER, Base.HOME, False)
        return self.handle_runners(state, [advance])


@dataclass(frozen=True)
class ErrorEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        advance = RunnerAdvance(Base.BATTER, Base.FIRST, False)
        return self.handle_runners(state, [advance])


@dataclass(frozen=True)
class DefensiveIndifferenceEvent(Event):
    ...


@dataclass(frozen=True)
class OtherAdvanceEvent(Event):
    ...


@dataclass(frozen=True)
class WildPitchEvent(Event):
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
class IntentionalWalkEvent(WalkEvent):
    ...


@dataclass(frozen=True)
class SecondaryErrorEvent(Event):
    ...


@dataclass(frozen=True)
class PassedBallEvent(Event):
    ...
