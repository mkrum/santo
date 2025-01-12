import re
from typing import List, Set, Optional
from itertools import product
from dataclasses import dataclass, field

from santo.utils import Base, Modifier, Position
from santo.game import GameState
from santo.updates import RunnerAdvance

@dataclass(frozen=True)
class Event:
    def advances(self) -> [RunnerAdvance]:
        ...

@dataclass(frozen=True)
class Item:
    event: Event
    modifiers: List[Modifier] = field(default_factory=lambda: [])
    advancements: List[RunnerAdvance] = field(default_factory=lambda: [])

    def get_runners(self) -> List[str]:
        return self.advancements

    def handle_runners(
        self, state, other_runners: List[RunnerAdvance] = None
    ) -> GameState:

        new_state = state

        runners = self.advancements + self.event.advances()

        # Runners make sure runners are sorted, can get messed up sometimes
        runners = sorted(runners, key=lambda k: k.from_base.value, reverse=True)

        if other_runners:
            runners = simplify_runner_advances(runners + other_runners)

        for advance_runners in runners:
            new_state = advance_runners(new_state)

        return new_state

    def __call__(self, state: GameState) -> GameState:
        """
        By Default, an event will just look at the notation for advancing
        runners, and and advance those runners.
        """
        return self.handle_runners(state)

    
def force_advance(state, base: Base) -> GameState:
    # If there is a runner on the base we are trying to move to, first move
    # that runner
    if state.bases[base.next_base()]:
        state = force_advance(state, base.next_base())

    advance = RunnerAdvance(base, base.next_base(), False)
    state = advance(state)
    return state


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
class IdentityEvent(Event):
    def advances(self) -> [RunnerAdvance]:
        return []


@dataclass(frozen=True)
class OutEvent(Event):
    positions: List[Position]
    player: Base

    def advances(self):
       return [RunnerAdvance(self.player, self.player.next_base(), is_out=True)]

@dataclass(frozen=True)
class MultiOutEvent(Event):
    outs: List[OutEvent]

    def advances(self):

        advances = sum([o.advances() for o in self.outs])

        # If the batter is not listed as being out, it is implicitily a single
        if not any([r.from_base == Base.BATTER for r in advances]):
            advances.append(RunnerAdvance(Base.BATTER, Base.FIRST, False))

        return advances

@dataclass(frozen=True)
class StrikeOutEvent(Event):

    def advances(self):
        # But, a K is not always an out! A batter can advance to fist on a wild
        # pitch. This will need to be explicitly stated. So, this should get overwritten if that is the case
        return RunnerAdvance(Base.BATTER, Base.FIRST, False)

#@dataclass(frozen=True)
#class StrikeOutEvent(Event):
#    def _handle_plus(self):
#        other_event_str = self.raw_string.split("+")[1]
#
#        if other_event_str[:2] == "CS":
#            other_event = CaughtStealingEvent.from_string(other_event_str)
#        elif other_event_str[:2] == "SB":
#            other_event = StolenBaseEvent.from_string(other_event_str)
#        elif other_event_str[:2] == "WP":
#            other_event = WildPitchEvent.from_string(other_event_str)
#        elif other_event_str[:2] == "PB":
#            other_event = PassedBallEvent.from_string(other_event_str)
#        elif other_event_str[:2] == "OA":
#            other_event = OtherAdvanceEvent.from_string(other_event_str)
#        elif other_event_str[:4] == "POCS":
#            other_event = PickedOffCaughtStealingEvent.from_string(other_event_str)
#        elif other_event_str[:2] == "PO":
#            other_event = PickedOffEvent.from_string(other_event_str)
#        elif other_event_str[:2] == "DI":
#            other_event = DefensiveIndifferenceEvent.from_string(other_event_str)
#        elif other_event_str[0] == "E":
#            other_event = SecondaryErrorEvent.from_string(other_event_str)
#        else:
#            # I am using an "assert False" here, since I think we are only
#            # catching Assertion errors to differentiate between parsing
#            # errors and programmatic errors.
#            assert False, f"unknown event {other_event}"
#
#        return other_event



@dataclass(frozen=True)
class WalkEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        new_state = state

        if len(self.raw_string) > 1 and self.raw_string[1] == "+":
            other_event = self._handle_plus()
            new_state = other_event(new_state)
            runners = other_event.get_runners()

            if any([r.from_base == Base.BATTER for r in runners]):
                return new_state
            else:
                return force_advance(new_state, Base.BATTER)
        else:
            new_state = self.handle_runners(new_state)

            if any([r.from_base == Base.BATTER for r in self.get_runners()]):
                return new_state
            else:
                return force_advance(new_state, Base.BATTER)

@dataclass(frozen=True)
class HitEvent(Event):
    fielded: Optional[Position] = None

@dataclass(frozen=True)
class SingleEvent(HitEvent):
    def advances(self):
        return [RunnerAdvance(Base.BATTER, Base.FIRST, False)]

@dataclass(frozen=True)
class DoubleEvent(HitEvent):
    def advances(self):
        return [RunnerAdvance(Base.BATTER, Base.SECOND, False)]

@dataclass(frozen=True)
class TripleEvent(HitEvent):
    def advances(self):
        return [RunnerAdvance(Base.BATTER, Base.THIRD, False)]

@dataclass(frozen=True)
class StolenBaseEvent(Event):

    stolen_base: Base

    def advances(self): 
        return RunnerAdvance(self.stolen_base.last_base, self.stolen_base, False)

@dataclass(frozen=True)
class CaughtStealingEvent(Event):
    stolen_base: Base

    def advances(self): 
        return RunnerAdvance(self.stolen_base.last_base, self.stolen_base, True)

    #def _get_base(self, string):
    #    stolen_base = string[2]
    #    return stolen_base

    #def __call__(self, state: GameState) -> GameState:
    #    cs_strings = list(filter(lambda x: "CS" in x, self.raw_string.split(";")))

    #    new_state = state
    #    advances = []
    #    for cs_string in cs_strings:
    #        stolen_base = self._get_base(cs_string)

    #        modifer_strings = re.findall(r"\((.*?)\)", cs_string)
    #        is_out = True

    #        if any([("E" in m) for m in modifer_strings]):
    #            additional_out_string = re.findall(r"\((\d*?)\)", self.raw_string)
    #            is_out = len(additional_out_string) != 0

    #        if stolen_base == "2":
    #            advances.append(RunnerAdvance(Base.FIRST, Base.SECOND, is_out))
    #        elif stolen_base == "3":
    #            advances.append(RunnerAdvance(Base.SECOND, Base.THIRD, is_out))
    #        elif stolen_base == "H":
    #            advances.append(RunnerAdvance(Base.THIRD, Base.HOME, is_out))

    #    new_state = self.handle_runners(new_state, advances)
    #    return new_state



@dataclass(frozen=True)
class PickedOffEvent(Event):

    pickoff_base: Base

    def advances(self): 
        return [RunnerAdvance(self.pickoff_base, self.pickoff_base, is_out=True)]

    #def __call__(self, state: GameState) -> GameState:
    #    pickoff_base = self.raw_string[2]

    #    play_string = self.raw_string.split("/")[0].split(".")[0]

    #    if "E" in play_string:
    #        return self.handle_runners(state)
    #    elif pickoff_base == "1":
    #        advance = RunnerAdvance(Base.FIRST, Base.FIRST, True)
    #    elif pickoff_base == "2":
    #        advance = RunnerAdvance(Base.SECOND, Base.SECOND, True)
    #    elif pickoff_base == "3":
    #        advance = RunnerAdvance(Base.THIRD, Base.THIRD, True)

    #    return self.handle_runners(state, [advance])


@dataclass(frozen=True)
class FieldersChoiceEvent(Event):

    def advances(self): 
        return [RunnerAdvance(Base.BATTER, Base.FIRST, False)]

@dataclass(frozen=True)
class CatcherInterferenceEvent(Event):

    def advances(self): 
        return [RunnerAdvance(Base.BATTER, Base.FIRST, False)]

@dataclass(frozen=True)
class PickedOffCaughtStealingEvent(CaughtStealingEvent):
    def _get_base(self, string):
        stolen_base = string[4]
        return stolen_base


@dataclass(frozen=True)
class HitByPitchEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        return force_advance(state, Base.BATTER)


@dataclass(frozen=True)
class HomeRunEvent(Event):
    def __call__(self, state: GameState) -> GameState:
        advance = RunnerAdvance(Base.BATTER, Base.HOME, False)
        return self.handle_runners(state, [advance])


@dataclass(frozen=True)
class ErrorEvent(Event):

    position: Position = None

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
