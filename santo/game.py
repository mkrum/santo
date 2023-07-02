import re
from dataclasses import dataclass, replace
from pyrsistent import PMap, pmap

EMPTY_BASES = pmap({1: False, 2: False, 3: False})

@dataclass(frozen=True)
class GameState:
    inning: int = 0
    outs: int = 0

    home_team_up: bool = False

    score: PMap = pmap({"home": 0, "away": 0})
    bases: PMap = EMPTY_BASES

    def at_bat(self):
        if self.home_team_up:
            return "home"
        else:
            return "away"

    def add_out(self) -> "GameState":
        current_outs = self.outs

        assert current_outs < 3
        assert current_outs >= 0

        new_outs = current_outs + 1

        return replace(self, outs=new_outs)

    def inning_is_over(self) -> bool:
        return self.outs == 3

    def end_inning(self) -> "GameState":
        home_team_up = not self.home_team_up

        # Fully reset the state, only carry over the score
        return GameState(
            score=self.score,
            home_team_up=home_team_up,
        )

    def add_runs(self, runs: int) -> "GameState":
        at_bat = self.at_bat()
        score = self.score
        new_score = self.score.get(at_bat, runs)
        new_score_map = self.score.set(at_bat, new_score)
        return replace(
            self, score=new_score_map
        )

    def empty_bases(self) -> 'GameState':
        return replace(self, bases=EMPTY_BASES)

    def add_home_team_score(self) -> "GameState":
        return self._add_run(is_home_team=True)

    def add_away_team_score(self) -> "GameState":
        return self._add_run(is_home_team=False)

    def add_runner(self, base_idx: int) -> "GameState":
        new_bases = self.bases.set(base_idx, True)
        return replace(self, bases=new_bases)

    def get_num_runners(self):
        return sum([int(self.bases[i]) for i in [1, 2, 3]])

    def home_run(self) -> "GameState":
        num_runners = self.get_num_runners()
        new_state = self.add_runs(num_runners + 1)
        new_state = new_state.empty_bases()
        return new_state

    def force_advance(self) -> "GameState":
        new_bases = self.bases
        
        new_runs = 0
        if not new_bases[1]:
            new_bases = new_bases.set(1, True)
        elif new_bases[1] and not (new_bases[2] or new_bases[3]):
            new_bases = new_bases.set(1, True)
            new_bases = new_bases.set(2, True)
        elif new_bases[1] and new_bases[2] and not new_bases[3]:
            new_bases = new_bases.set(1, True)
            new_bases = new_bases.set(2, True)
            new_bases = new_bases.set(3, True)
        elif new_bases[1] and new_bases[2] and new_bases[3]:
            new_runs += 1
        
        new_state = self.add_runs(new_runs)
        new_state = replace(new_state, bases=new_bases)
        return new_state

    def move_runners(self, move_string) -> "GameState":
        new_state = self
        if move_string[1] != '-':
            if move_string[0] != "B":
                new_state = new_state.add_out()
        else:
            new_bases = self.bases

            from_base = move_string[0]
            to_base = move_string[2]

            new_bases = new_bases.set(from_base, False)

            if to_base == 'H':
                new_state = new_state.add_runs(1)
            else:
                new_bases = new_bases.set(to_base, False)

            new_state = replace(new_state, bases=new_bases)
        return new_state
