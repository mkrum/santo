from typing import Any, List, Optional

from dataclasses import dataclass, field

from lark import Lark, Transformer

from santo.utils import Base, Position, HitLocation, ModifierCode, Modifier
from santo.updates import RunnerAdvance
from santo.events import Event

from dataclasses import dataclass, field
from typing import List, Optional, Union

from santo.grammar import PlayTransformer
from santo.events import *

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
    "S4/G34.2-H(E4/TH)(UR)(NR);1-3;B-2",
    "1(B)16(2)63(1)/LTP/L1",
    "T9/F9LD.2-H",
    "K.1-2(WP)",
    "S/L9S.3-H;2X3(54/INT);1-2",
    #"E6/G6.3-H(RBI);2-3;B-1",
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
    Item(SingleEvent(fielded=Position(4)),
        [
            Modifier(modifier=ModifierCode.G, location=getattr(HitLocation, "34")),
        ],
        [
            RunnerAdvance(Base.SECOND, Base.HOME, False, True,
                          modifications=[
                              Modifier(ModifierCode.E, player=Position(4)),
                              Modifier(ModifierCode.TH),
                              Modifier(ModifierCode.UR),
                              Modifier(ModifierCode.NR),
                          ]
                          ),
            RunnerAdvance(Base.FIRST, Base.THIRD, False, True),
            RunnerAdvance(Base.BATTER, Base.SECOND, False, True),
        ]
    ),
    #"1(B)16(2)63(1)/LTP/L1",
    Item(MultiOutEvent(
           [
            OutEvent([Position(1)], Base.BATTER),
            OutEvent([Position(1), Position(6)], Base.SECOND),
            OutEvent([Position(6), Position(3)], Base.FIRST),
           ]
           ),
        [
            Modifier(modifier=ModifierCode.LTP),
            Modifier(modifier=ModifierCode.L, location=getattr(HitLocation, "1")),
        ],
    ),
    #"T9/F9LD.2-H"
    Item(
        TripleEvent(fielded=Position(9)),
        [Modifier(modifier=ModifierCode.F, location=getattr(HitLocation, "9LD"))],
        [RunnerAdvance(Base.SECOND, Base.HOME, False, True)]
    ),
    # "K.1-2(WP)"
    Item(
        StrikeOutEvent(),
        [],
        [RunnerAdvance(Base.FIRST, Base.SECOND, False, True, modifications=[Modifier(ModifierCode.WP)]) ]
    ),
    # "S/L9S.3-H;2X3(54/INT);1-2",
    Item(
        SingleEvent(),
        [Modifier(ModifierCode.L, location=getattr(HitLocation, "9S"))],
        [
            RunnerAdvance(Base.THIRD, Base.HOME, False, True),
            RunnerAdvance(Base.SECOND, Base.THIRD, True, True, modifications=[Modifier(modifier=None, player=[Position(5), Position(4)]), Modifier(modifier=ModifierCode.INT)]),
            RunnerAdvance(Base.FIRST, Base.SECOND, False, True),
        ]
    )
]

for s, p in zip(strings, parsed):
    out = parser.parse(s)
    data = transformer.transform(out)

    if data != p:
        print(s)

        print(data.event == p.event)
        print(data.event)
        print(p.event)

        print(data.modifiers == p.modifiers)
        print(data.modifiers)
        print(p.modifiers)

        print(data.advancements == p.advancements)
        for r in data.advancements:
            print(r)
        print('-' * 80)
        for r in p.advancements:
            print(r)

    assert data == p
