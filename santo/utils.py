
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

import functools


def partial(fn):
    def wrapped_fn(*args, **kwargs):
        return functools.partial(fn, *args, **kwargs)

    return wrapped_fn


def load_game_log(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, header=None)
    data = data.iloc[:, :11]
    data.columns = [
        "Date",
        "GameNumber",
        "Weekday",
        "VisitingTeam",
        "VisitingLeague",
        "VisitingTeamGameNumber",
        "HomeTeam",
        "HomeLeague",
        "HomeTeamGameNumber",
        "VisitingScore",
        "HomeScore",
    ]
    return data

class Position(Enum):
    PITCHER: int = 1
    CATCHER: int = 2
    FIRST_BASEMAN: int = 3
    SECOND_BASEMAN: int = 4
    THIRD_BASEMAN: int = 5
    SHORTSTOP: int = 6
    LEFT_FIELDER: int = 7
    CENTER_FIELDER: int = 8
    RIGHT_FIELDER: int = 9
    UNKNOWN: int = 10

class Base(Enum):
    BATTER: int = 0
    FIRST: int = 1
    SECOND: int = 2
    THIRD: int = 3
    HOME: int = 4

    @classmethod
    def from_int(cls, integer: int) -> "Base":
        names = ["BATTER", "FIRST", "SECOND", "THIRD", "HOME"]
        return cls[names[integer]]

    @classmethod
    def from_short_string(cls, short_string: str) -> "Base":
        possible = ["B", "1", "2", "3", "H"]
        assert short_string in possible
        return cls.from_int(possible.index(short_string))

    def next_base(self):
        # No next base for Home
        assert self.value != 4
        return self.__class__.from_int(self.value + 1)

    def last_base(self):
        # No last base for batter
        assert self.value != 0
        return self.__class__.from_int(self.value - 1)

    def __ge__(self, other):
        return self.value >= other.value

    def __gt__(self, other):
        return self.value > other.value

HitLocation= Enum(
    "HitLocation",
    [
        "78XD",
        "8XD",
        "89XD",
        "7LD",
        "7D",
        "78D",
        "8D",
        "89D",
        "9D",
        "9LD",
        "7L",
        "7",
        "78",
        "8",
        "89",
        "9",
        "9L",
        "7LS",
        "7S",
        "78S",
        "8S",
        "89S",
        "9S",
        "9LS",
        "5D",
        "56D",
        "6D",
        "6MD",
        "4MD",
        "4D",
        "34D",
        "3D",
        "5",
        "56",
        "6",
        "6M",
        "4M",
        "4",
        "34",
        "3",
    ],
)

ModifierCode = Enum(
    "ModifierCode",
    ["AP",
     "E",
     "UR",
     "NR",
     "TUR",
     "BP",
     "BG",
     "BGDP",
     "BINT",
     "BL",
     "BOOT",
     "BPDP",
     "BR",
     "C",
     "COUB",
     "COUF",
     "COUR",
     "DP",
     "F",
     "FDP",
     "FINT",
     "FL",
     "FO",
     "G",
     "GDP",
     "GTP",
     "IF",
     "INT",
     "IPHR",
     "L",
     "LDP",
     "LTP",
     "MREV",
     "NDP",
     "OBS",
     "P",
     "PASS",
     "RINT",
     "SF",
     "SH",
     "TH",
     "TP",
     "UINT",
     "UREV"
])

HitLocation= Enum(
    "HitLocation",
    [
        "2F",
        "25F",
        "25",
        "1S",
        "23",
        "23F",
        "5S",
        "58S",
        "15",
        "1",
        "13",
        "34S",
        "6S",
        "6MS",
        "4MS",
        "4S",
        "3S",
        "5F",
        "5",
        "56",
        "6",
        "6M",
        "4M",
        "4",
        "34",
        "3",
        "3F",
        "5DF",
        "5D",
        "58D",
        "6D",
        "6MD",
        "4MD",
        "4D",
        "34D",
        "3D",
        "3DF",
        "7LSF",
        "7LS",
        "7S",
        "78S",
        "8S",
        "89S",
        "9S",
        "9LS",
        "9LSF",
        "7LF",
        "7L",
        "7",
        "78",
        "8",
        "89",
        "9",
        "9L",
        "9LF",
        "7LDF",
        "7LD",
        "7D",
        "78D",
        "8D",
        "89D",
        "9D",
        "9LD",
        "9LDF",
        "78XD",
        "8XD",
        "89XD"
    ],
)

@dataclass(frozen=True)
class Modifier:
    modifier: ModifierCode
    location: Optional[HitLocation] = None
    base: Optional[Base] = None
    player: Optional[Position] = None
