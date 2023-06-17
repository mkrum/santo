
import pandas as pd

def load_game_log(path: str) -> pd.DataFrame: 
    data = pd.read_csv(path, header=None)
    data = data.iloc[:, :11]
    data.columns = ["Date", "GameNumber", "Weekday", "VisitingTeam", "VisitingLeague", "VisitingTeamGameNumber", "HomeTeam", "HomeLeague", "HomeTeamGameNumber", "VisitingScore", "HomeScore"]
    return data

