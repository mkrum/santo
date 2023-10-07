import argparse
import logging
import glob

import santo.data
from santo.utils import load_game_log
from santo.tests import check_correctness

logging.getLogger().setLevel(logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game_id")

    args = parser.parse_args()

    year = int(args.game_id[3:7])
    print(year)

    data = load_game_log(f"./data/{year}/gl{year}.txt")
    all_files = glob.glob(f"./data/{year}/*.EV*")
    all_games = []
    for f in all_files:
        all_games += santo.data.load_evn(f)

    game = next(filter(lambda x: x.game_id == args.game_id, all_games))
    check_correctness(data, game)
