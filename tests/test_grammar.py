from typing import Any, List, Optional

from dataclasses import dataclass, field

from lark import Lark, Transformer

from santo.utils import Base, Position, HitLocation, ModifierCode
from santo.updates import RunnerAdvance
from santo.events import Event

from dataclasses import dataclass, field
from typing import List, Optional, Union

from santo.grammar import PlayTransformer, OutEvent, Item

# Define the grammar
grammar = open("grammar.lark", "r").read()
parser = Lark(grammar, start='start', parser='lalr', debug=True)

transformer = PlayTransformer()

strings = [
    "6",
    "6413"
]
parsed = [
    Item(OutEvent([Position(6)], Base.BATTER)),
    Item(OutEvent([Position(6), Position(4), Position(1), Position(3)], Base.BATTER)),
]

for s, p in zip(strings, parsed):
    out = parser.parse(s)
    data = transformer.transform(out)
    print(data)
    print(p)
    assert data == p
