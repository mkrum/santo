from typing import Any, List, Optional

from dataclasses import dataclass, field

from lark import Lark, Transformer

from santo.utils import Base, Position, HitLocation, ModifierCode
from santo.updates import RunnerAdvance
from santo.events import Event

from dataclasses import dataclass, field
from typing import List, Optional, Union

from santo.grammar import PlayTransformer

# Define the grammar
grammar = open("grammar.lark", "r").read()
parser = Lark(grammar, start='start', parser='lalr', debug=True)
transformer = PlayTransformer()

strings = ["6", "6413", "64(1)3", "64(1)3(B)9", "4(1)3/G4/GDP", "8(B)84(2)/LDP/L8", "54(B)/BG25/SH.1-2", "1(B)16(2)63(1)/LTP/L1", "T9/F9LD.2-H", "S4/G34.2-H(E4/TH)(UR)(NR);1-3;B-2", "E6/G6.3-H(RBI);2-3;B-1", "S/L9S.3-H;2X3(5/INT);1-2", "K.1-2(WP)"]
#strings = ["4(1)3/G4/LDP"]
strings= ["E6/G6.3-H(RBI);2-3;B-1"]

#MultiOutPlay(outs=[OutEvent([Position.SECOND], Base.FIRST), OutEvent([Position.FIRST], BATTER.BATTER)], modifiers=Modifier(

for s in strings:
    print(s)
    out = parser.parse(s)
    #print(out)
    data = transformer.transform(out)
    print(data.event)
    print(data.modifiers)
    print(data.advancements)
