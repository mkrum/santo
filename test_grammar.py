
from typing import Any, List, Optional

from dataclasses import dataclass, field

from lark import Lark, Transformer

from santo.utils import Base, Position, HitLocation, ModifierCode
from santo.updates import RunnerAdvance

from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class Modifier:
    modifier: ModifierCode
    location: Optional[HitLocation] = None
    base: Optional[Base] = None
    player: Optional[Position] = None

@dataclass
class Event:
    event: Any
    modifiers: List[Modifier] = None
    advancements: List[RunnerAdvance] = None

class PlayTransformer(Transformer):

    def HIT_LOCATION(self, items):
        location = items[0]
        return getattr(HitLocation, location)

    def MODIFIERS(self, items):
        modifier = items[0]
        return getattr(ModifierCode, modifier)

    def modifier(self, items):
        code = items[0] 

        location = None
        base = None
        player = None
        if len(items) > 1:
            additional_info = items[1]
            if isinstance(additional_info, Base):
                base = additional_info
            elif isinstance(additional_info, Position):
                player = additional_info
            elif isinstance(additional_info, HitLocation):
                location = additional_info
            else:
                throw(f"Not sure how to handle this modifer info: {additional_info}")

        assert len(items) <= 2
        return Modifier(code, location=location, base=base, player=player)

    def base(self, items):
        return items[0]

    def position(self, items):
        return getattr(Position, items[0].type)

    def succesful_advancement(self, items):
        return RunnerAdvance(*items, is_out=False, explicit=True)

    def unsuccesful_advancement(self, items):
        return RunnerAdvance(*items, is_out=True, explicit=True)

    def advancement(self, items):
        return items[0]

    def advancements(self, items):
        return items

    def start(self, items):
        event = items[0]

        modifiers = None
        advancements = None

        for i in items[1:]:
            if isinstance(i, list) and isinstance(i[0], RunnerAdvance):
                advancements = i
            else:
                modifiers = i

        return Event(event, modifiers, advancements)

    def event(self, items):
        print(items)
        return items[0]

    def event(self, items):

# Add bases
for base in Base:
    setattr(PlayTransformer, base.name, lambda self, tok: base)

# Define the grammar
grammar = open("grammar.lark", "r").read()
parser = Lark(grammar, start='start', parser='lalr', debug=True)
transformer = PlayTransformer()

#strings = ["6", "6413", "64(1)3", "64(1)3(B)9", "4(1)3/G4/GDP", "8(B)84(2)/LDP/L8", "54(B)/BG25/SH.1-2", "1(B)16(2)63(1)/LTP/L1", "T9/F9LD.2-H", "S4/G34.2-H(E4/TH)(UR)(NR);1-3;B-2", "E6/G6.3-H(RBI);2-3;B-1", "S/L9S.3-H;2X3(5/INT);1-2", "K.1-2(WP)"]
strings = ["E6/G6.3-H(RBI);2-3;B-1"]

for s in strings:
    out = parser.parse(s)
    #print(out)
    data = transformer.transform(out)
    print(data)
    exit()
    #children = out.children

    #event_data = children[0].children

    #modifiers = None
    #advancements = None
    #for c in children[1:]:
    #    if c.data == 'advancements':
    #        advancements = c.children
    #    if c.data == 'modifiers':
    #        modifiers = c.children

    #breakpoint()
    #advances = parse_advances(advancements)
