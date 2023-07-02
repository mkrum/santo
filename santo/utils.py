
import pandas as pd

def load_game_log(path: str) -> pd.DataFrame: 
    data = pd.read_csv(path, header=None)
    data = data.iloc[:, :11]
    data.columns = ["Date", "GameNumber", "Weekday", "VisitingTeam", "VisitingLeague", "VisitingTeamGameNumber", "HomeTeam", "HomeLeague", "HomeTeamGameNumber", "VisitingScore", "HomeScore"]
    return data

BATTING_ZONES = [
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
    "3"
]
