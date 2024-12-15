
from typing import List, Optional, Any
from dataclasses import dataclass, field

from lark import Lark, Transformer

from santo.utils import Base, Position, HitLocation, ModifierCode, Modifier
from santo.updates import RunnerAdvance
from santo.events import *



class PlayTransformer(Transformer):

    def HIT_LOCATION(self, items):
        location = items
        return getattr(HitLocation, location)

    def MODIFIERS(self, items):
        modifier = items
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

    def hit(self, items):
        assert len(items) == 1
        return items[0]

    def single(self, items):
        return SingleEvent(items[0]) if len(items) > 0 else SingleEvent()

    def base(self, items):
        return getattr(Base, items[0].type)

    def position(self, items):
        return getattr(Position, items[0].type)

    def succesful_advancement(self, items):
        return dict(from_base=items[0], to_base=items[1], is_out=False, explicit=True)

    def unsuccesful_advancement(self, items):
        return dict(from_base=items[0], to_base=items[1], is_out=True, explicit=True)

    def advancement(self, items):
        args = items[0]
        modifications = items[1:]
        return RunnerAdvance(**args, modifications=modifications)

    def advancements(self, items):
        return items

    def start(self, items):
        event = items[0]

        modifiers = []
        advancements = []

        for i in items[1:]:
            if isinstance(i, list) and isinstance(i[0], RunnerAdvance):
                advancements = i
            else:
                modifiers.append(i)

        return Item(event, modifiers, advancements)

    def event(self, items):
        assert len(items) == 1
        return items[0]

    def out(self, items):
        assert len(items) == 1
        return items[0]

    def advance_modifier(self, items):
        return items

    def assisted_out(self, items):
        return OutEvent(items, Base.BATTER)

    def unassisted_out(self, items):
        return OutEvent(items, Base.BATTER)

    def unspecified_out(self, items):
        return OutEvent(items[0].positions, Base.BATTER)

    def specified_out(self, items):
        old_out = items[0].positions
        batter_out = items[1]
        return OutEvent(old_out, batter_out)

    def double_play(self, items):
        return MultiOutEvent(items)

    def triple_play(self, items):
        return MultiOutEvent(items)
