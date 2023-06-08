import sys
from typing import List

import dataclasses
from dataclasses import dataclass


@dataclass
class Entry:
    type_name: str

    @classmethod
    def from_entry(cls, data_list: List[str]) -> "Entry":
        types = [field.type for field in dataclasses.fields(cls)]
        init_list = [t(d) for (t, d) in zip(types, data_list)]
        return cls(*init_list)


@dataclass
class CommentaryEntry(Entry):
    com: str


@dataclass
class PlayEntry(Entry):
    inning: int
    is_home_team: bool
    player_at_player: str
    count: str
    pitches: str
    play: str


@dataclass
class IdEntry(Entry):
    id_: str


@dataclass
class VersionEntry(Entry):
    version: int


@dataclass
class StartEntry(Entry):
    player_id: str
    player_name: str
    is_home_team: bool
    batting_order: int
    fielding_position: int


@dataclass
class SubEntry(Entry):
    player_id: str
    player_name: str
    is_home_team: bool
    batting_order: int
    fielding_position: int


@dataclass
class InfoEntry(Entry):
    key: str
    value: str


@dataclass
class DataEntry(Entry):
    data_type: str
    player_id: str
    value: str


@dataclass
class Game:
    entries: List[Entry]

    def _get_entries(self, data_type: Entry) -> List[Entry]:
        return list(filter(lambda x: isinstance(x, data_type), self.entries))

    def get_home_team(self) -> str:
        infos = self._get_entries(InfoEntry)
        home_team = list(filter(lambda info: info.key == "hometeam", infos))
        assert len(home_team) == 1
        return home_team[0].value

    def get_away_team(self) -> str:
        infos = self._get_entries(InfoEntry)
        away_team = list(filter(lambda info: info.key == "visteam", infos))
        assert len(away_team) == 1
        return away_team[0].value

    def box_score(self):
        total_plays = self._get_entries(PlayEntry)
        for inning in range(1, 10):
            plays = list(filter(lambda x: x.inning == inning, total_plays))

            home_team_plays = []
            away_team_plays = []
            for p in plays:
                if p.is_home_team:
                    home_team_plays.append(p)
                else:
                    away_team_plays.append(p)
            breakpoint()

def load_evn(file_name: str) -> List[Game]:
    with open(file_name, "r") as input_file:
        raw_data = input_file.read()

    data_lines = raw_data.splitlines()
    data = list(map(lambda x: list(x.split(",")), data_lines))
    entries = load_entries(data)
    return split_into_games(entries)


def load_entries(data: List[List[str]]) -> List[Entry]:
    entries = []
    for d in data:
        type_name = d[0]

        data_type = None
        if type_name == "id":
            data_type = IdEntry
        elif type_name == "play":
            data_type = PlayEntry
        elif type_name == "start":
            data_type = StartEntry
        elif type_name == "sub":
            data_type = SubEntry
        elif type_name == "version":
            data_type = VersionEntry
        elif type_name == "info":
            data_type = InfoEntry
        elif type_name == "com":
            data_type = CommentaryEntry
        elif type_name == "data":
            data_type = CommentaryEntry

        entries.append(data_type.from_entry(d))

    return entries


def split_into_games(entries: List[Entry]) -> List[Game]:
    game_buffer = []
    games = []
    for e in entries:
        if isinstance(e, IdEntry) and len(game_buffer) > 0:
            games.append(Game(game_buffer))
            game_buffer = []
        game_buffer.append(e)

    if len(game_buffer) > 0:
        games.append(Game(game_buffer))

    return games
