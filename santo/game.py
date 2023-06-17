
import re
from dataclasses import dataclass, replace
from pyrsistent import PMap, pmap

EMPTY_BASES = pmap({1: False, 2: False, 3: False})

FIELDERS = list(range(1, 10))


@dataclass(frozen=True)
class GameState:
    inning: int = 0
    outs: int = 0

    home_team_up: bool = False
    away_team_up: bool = True

    home_team_score: int = 0
    away_team_score: int = 0
    
    bases: PMap = EMPTY_BASES

    def add_out(self) -> "GameState":
        current_outs = self.outs

        assert current_outs < 3
        assert current_outs >= 0

        new_outs = current_outs + 1

        return replace(self, outs=new_outs)

    def inning_is_over(self) -> bool:
        return self.outs == 3

    def end_inning(self) -> "GameState":
        home_team_up = not self.home_team_up
        away_team_up = not self.away_team_up
        
        # Fully reset the state, only carry over the score
        return GameState(
            home_team_score=self.home_team_score,
            away_team_score=self.away_team_score,
            home_team_up=home_team_up,
            away_team_up=away_team_up,
        )

    def _add_run(self, is_home_team: bool) -> "GameState":
        ...
        home_team_score = self.home_team_score
        away_team_score = self.away_team_score

        if is_home_team:
            home_team_score += 1
        else:
            away_team_score += 1

        return replace(self, home_team_score=home_team_score, away_team_score=away_team_score)

    def add_home_team_score(self) -> 'GameState':
        return self._add_run(is_home_team=True)

    def add_away_team_score(self) -> 'GameState':
        return self._add_run(is_home_team=False)

    def add_runner(self, base_idx: int) -> "GameState":
        new_bases = self.bases.set(base_idx, True)
        return replace(self, bases=new_bases)

@dataclass(frozen=True)
class IdentityEvent:
    def __call__(self, state: GameState) -> GameState:
        return state
        
@dataclass(frozen=True)
class Event:

    def __add__(self, other_event): 
        return UnionEvent(self, other_event)

    def __call__(self, state: GameState) -> GameState:
        ...
    
    @classmethod
    def from_string(cls, play_str: str):

        events_strs = play_str.split("/")

        event_str = events_strs[0]
        modifier_strs = events_strs[1:]
        
        # Nothing happens
        event = IdentityEvent()
        
        # Out
        if re.match(r"^[1-9]+(\/.*)?", play_str):
            event = OutEvent()
        
        # Out, with runner specified
        elif re.match(r"^[1-9]+\([B,1-3]\)(\/.*)?", play_str):
            event = OutEvent()

        # double play
        elif re.match(r"^[1-9]+\([B,1-3]\)[1-9](\/.*)?", play_str):
            event = OutEvent() + OutEvent()

        # Some kind of interference
        elif re.match(r"C\/E[1-9](\.[B,1-3][-,X][B,1-3])?", play_str):
            event = HitEvent(bags=1)
        
        # Singles, Doubles, Tripples
        elif re.match(r"S[1-9]?(\/.*)?", play_str):
            event = HitEvent(bags=1)

        elif re.match(r"D[1-9]?(\/.*)?", play_str):
            event = HitEvent(bags=2)

        elif re.match(r"T[1-9]?(\/.*)?", play_str):
            event = HitEvent(bags=3)

        # Ground Rule Double
        elif re.match(r"DGR(\/.*)?", play_str):
            event = HitEvent(bags=2)

        # Runner reaches on error
        elif re.match(r"E[1-9](\/.*)?", play_str):
            event = HitEvent(bags=3)
        
        # Fielders Choice
        elif re.match(r"FC[1-9](\/.*)?", play_str):
            event = HitEvent(bags=1)

        # Error on foul ball
        elif re.match(r"FLE(\/.*)?", play_str):
            event = HitEvent(bags=1)
        
        # Some kind of home run
        elif re.match(r"H(R)?(\/.*)?", play_str):
            event = HitEvent(bags=1)

        elif re.match(r"HP(\/.*)?", play_str):
            event = OutEvent()

        elif re.match(r"K(\+.*)?(\/.*)?", play_str):
            event = OutEvent()
        
        # No Play, used for subs sometimes
        elif play_str == "NP":
            return IdentityEvent()

        # Intentional Walk
        elif re.match(r"I(W)?(\+.*)?(\/.*)?", play_str):
            event = HitEvent(bags=1)

        # Normal Walk
        elif re.match(r"W(\+.*)?(\/.*)?", play_str):
            event = HitEvent(bags=1)

        elif re.match(r"W(\+.*)?(\/.*)?", play_str):
            event = HitEvent(bags=1)
        
        # Balk
        elif re.match(r"BK(\.[B,1-3][-,X][B,1-3])*;?", play_str):
            event = HitEvent(bags=1)

        # Defensive Indifference, no attempt to try to stop the player from
        # stealing
        elif re.match(r"DI(\.[B,1-3][-,X][B,1-3])*;?", play_str):
            event = HitEvent(bags=1)

        # Defensive Indifference, no attempt to try to stop the player from
        # stealing
        elif re.match(r"OA(\.[B,1-3][-,X][B,1-3])*;?", play_str):
            event = HitEvent(bags=1)

        # Passed Ball
        elif re.match(r"PB(\.[B,1-3][-,X][B,1-3])*;?", play_str):
            event = HitEvent(bags=1)

        # Wild pitch
        elif re.match(r"WP(\.[B,1-3][-,X][B,1-3])*;?", play_str):
            event = HitEvent(bags=1)

        # Caught Stealing
        elif re.match(r"CS[2,3,H]\(.*\)(\.[B,1-3][-,X][B,1-3])*;?", play_str):
            event = HitEvent(bags=1)

        # Picked Off
        elif re.match(r"PO[2,3,H]\(.*\)(\.[B,1-3][-,X][B,1-3])*;?", play_str):
            event = HitEvent(bags=1)

        # Picked Off, runner charged with caught stealing
        elif re.match(r"POCS[2,3,H]\(.*\)(\.[B,1-3][-,X][B,1-3])*;?", play_str):
            event = HitEvent(bags=1)

        else:
            print("Didn't " + play_str)

        return event

@dataclass(frozen=True)
class OutEvent:
    def __call__(self, state: GameState) -> GameState:
        return state.add_out()

@dataclass(frozen=True)
class HitEvent:

    bags: int

    def __call__(self, state: GameState) -> GameState:
        return state.add_runner(self.bags)

@dataclass(frozen=True)
class UnionEvent:

    event_one: Event
    event_two: Event

    def __call__(self, state: GameState) -> GameState: 
        return event_two(event_one(state))
    

def parse(state: GameState, event_str: str) -> GameState:

    event = Event.from_string(event_str)
    state = event(state)

    if state.inning_is_over():
        state = state.end_inning()

    return state
