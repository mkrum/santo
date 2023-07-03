import pandas as pd
from enum import Enum


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


BattingZone = Enum(
    "BattingZone",
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
