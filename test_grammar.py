
from typing import Any

from dataclasses import dataclass, field

from lark import Lark

from santo.utils import Base, Position
from santo.updates import RunnerAdvance


from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class Advancement:
    from_base: str
    to_base: str
    success: bool
    modifiers: Optional[List[str]] = field(default_factory=list)

@dataclass
class Modifier:
    type: str
    location: Optional[str] = None
    throw_base: Optional[str] = None
    relay_base: Optional[str] = None
    error_position: Optional[str] = None

@dataclass
class Event:
    type: str
    details: Optional[Union[str, List[str]]] = None

@dataclass
class Play:
    event: Event
    modifiers: List[Modifier] = field(default_factory=list)
    advancements: List[Advancement] = field(default_factory=list)

from lark import Transformer

class PlayTransformer(Transformer):

    def event(self, items):
        return Event(type=items[0], details=items[1:] if len(items) > 1 else None)

    def modifier(self, items):
        if items[0] in {"TH", "R"}:  # Special modifiers
            return Modifier(type=items[0], throw_base=items[1] if len(items) > 1 else None)
        elif items[0].startswith("E"):
            return Modifier(type=items[0], error_position=items[1] if len(items) > 1 else None)
        else:
            return Modifier(type=items[0], location=items[1] if len(items) > 1 else None)

    def BATTER(self, tok):
        return Base.BATTER

    def FIRST_BASE(self, tok):
        return Base.FIRST

    def SECOND_BASE(self, tok):
        return Base.SECOND

    def THIRD_BASE(self, tok):
        return Base.THIRD

    def HOME(self, tok):
        return Base.HOME

    def base(self, items):
        return items[0]

    def SHORTSTOP(self, tok):
        return Position.SHORTSTOP

    def position(self, items):
        return items[0]

    def succesful_advancement(self, items):
        return RunnerAdvance(*items, is_out=False, explicit=True)

    def unsuccesful_advancement(self, items):
        return RunnerAdvance(*items, is_out=True, explicit=True)

    def advancement(self, items):
        return items[0]

    def advancements(self, items):
        return items

# Define the grammar
grammar = open("grammar.lark", "r").read()
parser = Lark(grammar, start='start', parser='lalr', debug=True)
transformer = PlayTransformer()

#strings = ["6", "6413", "64(1)3", "64(1)3(B)9", "4(1)3/G4/GDP", "8(B)84(2)/LDP/L8", "54(B)/BG25/SH.1-2", "1(B)16(2)63(1)/LTP/L1", "T9/F9LD.2-H", "S4/G34.2-H(E4/TH)(UR)(NR);1-3;B-2", "E6/G6.3-H(RBI);2-3;B-1", "S/L9S.3-H;2X3(5/INT);1-2", "K.1-2(WP)"]
strings = ["E6/G6.3-H(RBI);2-3;B-1"]

for s in strings:
    out = parser.parse(s)
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
