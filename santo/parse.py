
from santo.events import *
from santo.game import GameState
import re

def parse_event_string(play_str: str) -> Event:
    events_strs = play_str.split("/")

    event_str = events_strs[0]
    modifier_strs = events_strs[1:]

    # Nothing happens
    event = IdentityEvent()

    # Out
    if re.match(r"^[1-9]+(\/.*)?", play_str):
        event_type = OutEvent

    # Out, with runner specified
    elif re.match(r"^[1-9]+\([B,1-3]\)(\/.*)?", play_str):
        event_type = OutEvent

    # double play
    elif re.match(r"^[1-9]+\([B,1-3]\)[1-9](\/.*)?", play_str):
        event_type = OutEvent

    # Some kind of interference
    elif re.match(r"C\/E[1-9](\.[B,1-3][-,X][B,1-3])?", play_str):
        event_type = HitEvent

    # Singles, Doubles, Tripples
    elif re.match(r"S[1-9]?(\/.*)?", play_str):
        event_type = HitEvent

    elif re.match(r"D[1-9]?(\/.*)?", play_str):
        event_type = HitEvent

    elif re.match(r"T[1-9]?(\/.*)?", play_str):
        event_type = HitEvent

    # Ground Rule Double
    elif re.match(r"DGR(\/.*)?", play_str):
        event_type = GroundRuleDoubleEvent

    # Runner reaches on error
    elif re.match(r"E[1-9](\/.*)?", play_str):
        event_type = ErrorEvent

    # Fielders Choice
    elif re.match(r"FC[1-9]?(\/.*)?", play_str):
        event_type = FieldersChoiceEvent

    # Error on foul ball
    elif re.match(r"FLE(\/.*)?", play_str):
        event_type = FoulBallErrorEvent

    # Some kind of home run
    elif re.match(r"H(R)?(\/.*)?", play_str):
        event_type = HomeRunEvent
    
    # Hit By Pitch
    elif re.match(r"HP(\/.*)?", play_str):
        event_type = HitByPitchEvent

    elif re.match(r"K(\+.*)?(\/.*)?", play_str):
        event_type = StrikeOutEvent

    # No Play, used for subs sometimes
    elif play_str == "NP":
        event_type = IdentityEvent

    # Intentional Walk
    elif re.match(r"I(W)?(\+.*)?(\/.*)?", play_str):
        event_type = IntentionalWalkEvent

    # Normal Walk
    elif re.match(r"W(\+.*)?(\/.*)?", play_str):
        event_type = WalkEvent

    # Balk
    elif re.match(r"BK(\.[B,1-3][-,X][B,1-3])*;?", play_str):
        event_type = BalkEvent

    # Defensive Indifference, no attempt to try to stop the player from
    # stealing
    elif re.match(r"DI(\.[B,1-3][-,X][B,1-3])*;?", play_str):
        event_type = DefensiveIndifferenceEvent
    
    # Runner Advance not covered by other things
    elif re.match(r"OA(\.[B,1-3][-,X][B,1-3])*;?", play_str):
        event_type = RunnersAdvanceEvent

    # Passed Ball
    elif re.match(r"PB(\.[B,1-3][-,X][B,1-3])*;?", play_str):
        event_type = PassedBallEvent

    # Wild pitch
    elif re.match(r"WP(\.[B,1-3][-,X][B,1-3])*;?", play_str):
        event_type = WildPitchEvent

    # Caught Stealing
    elif re.match(r"CS[2,3,H]\(.*\)(\.[B,1-3][-,X][B,1-3])*;?", play_str):
        event_type = CaughtStealingEvent

    # Picked Off
    elif re.match(r"PO[1,2,3,H]\(.*\)(\.[B,1-3][-,X][B,1-3])*;?", play_str):
        event_type = PickedOffEvent

    # Picked Off, runner charged with caught stealing
    elif re.match(r"POCS[2,3,H]\(.*\)(\.[B,1-3][-,X][B,1-3])*;?", play_str):
        event_type = PickedOffCaughtStealingEvent

    else:
        raise ValueError(f"Syntax error parsing {play_str}")

    event = event_type.from_string(play_str)
    return event


def parse(state: GameState, event_str: str) -> GameState:
    event = parse_event_string(event_str)
    state = event(state)

    if state.inning_is_over():
        state = state.end_inning()

    return state
