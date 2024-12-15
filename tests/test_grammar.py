from typing import Any, List, Optional

from dataclasses import dataclass, field

from lark import Lark, Transformer

from santo.utils import Base, Position, HitLocation, ModifierCode, Modifier
from santo.updates import RunnerAdvance
from santo.events import Event

from dataclasses import dataclass, field
from typing import List, Optional, Union

from santo.grammar import PlayTransformer
from santo.events import SingleEvent, OutEvent, MultiOutEvent, Item

# Define the grammar
grammar = open("grammar.lark", "r").read()
parser = Lark(grammar, start='start', parser='lalr', debug=True)

transformer = PlayTransformer()

strings = [
    "6",
    "6413",
    "64(1)3",
    "64(1)3(3)9",
    "4(1)3/G4/GDP",
    "S4/G34.2-H;1-3;B-2",
    #"S4/G34.2-H(E4/TH)(UR)(NR);1-3;B-2",
    #"1(B)16(2)63(1)/LTP/L1",
    #"T9/F9LD.2-H",
    #"E6/G6.3-H(RBI);2-3;B-1",
    #"S/L9S.3-H;2X3(5/INT);1-2",
    #"K.1-2(WP)"
]

parsed = [
    Item(OutEvent([Position(6)], Base.BATTER)),
    Item(OutEvent([Position(6), Position(4), Position(1), Position(3)], Base.BATTER)),
    Item(MultiOutEvent(
        [
            OutEvent([Position(6), Position(4)], Base.FIRST),
            OutEvent([Position(3)], Base.BATTER)
        ]
    )),
    Item(MultiOutEvent(
        [
            OutEvent([Position(6), Position(4)], Base.FIRST),
            OutEvent([Position(3)], Base.THIRD),
            OutEvent([Position(9)], Base.BATTER)
        ]
    )),
    Item(MultiOutEvent(
        [
            OutEvent([Position(4)], Base.FIRST),
            OutEvent([Position(3)], Base.BATTER),
        ]),
        [
            Modifier(modifier=ModifierCode.G, location=getattr(HitLocation, "4")),
            Modifier(modifier=ModifierCode.GDP),
        ]
    ),
    Item(SingleEvent(fielded=Position(4)),
        [
            Modifier(modifier=ModifierCode.G, location=getattr(HitLocation, "34")),
        ],
        [
            RunnerAdvance(Base.SECOND, Base.HOME, False, True),
            RunnerAdvance(Base.FIRST, Base.THIRD, False, True),
            RunnerAdvance(Base.BATTER, Base.SECOND, False, True),
        ]
    ),
]

for s, p in zip(strings, parsed):
    out = parser.parse(s)
    data = transformer.transform(out)

    if data != p:

        print(data.event == p.event)
        print(data.event)
        print(p.event)

        print(data.modifiers == p.modifiers)
        print(data.modifiers)
        print(p.modifiers)

        print(data.advancements == p.advancements)
        print(data.advancements)
        print(p.advancements)

    assert data == p
