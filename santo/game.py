import re
from dataclasses import dataclass, replace
from typing import List

from pyrsistent import PMap, pmap

from santo.utils import Base

EMPTY_BASES = pmap({b: False for b in Base})


@dataclass(frozen=True)
class GameState:
    inning: int = 1
    outs: int = 0
    manfred: bool = True
    _is_over: bool = False

    home_team_up: bool = False

    score: PMap = pmap({"home": 0, "away": 0})
    bases: PMap = EMPTY_BASES

    def at_bat(self):
        if self.home_team_up:
            return "home"
        else:
            return "away"

    @property
    def away_team_up(self):
        return not self.home_team_up

    def add_out(self, player: Base) -> "GameState":
        current_outs = self.outs

        assert current_outs < 3, f"Tried to set outs to {current_outs}"
        assert current_outs >= 0, f"Tried to set outs to {current_outs}"

        new_outs = current_outs + 1

        new_state = self.remove_runner(player)

        return replace(new_state, outs=new_outs)

    def inning_is_over(self) -> bool:
        return self.outs == 3

    def end_inning(self) -> "GameState":
        new_inning = self.inning

        if self.home_team_up:
            new_inning += 1

        home_team_up = not self.home_team_up

        _is_over = False
        if new_inning < 9:
            _is_over = False
        elif home_team_up and self.score["home"] > self.score["away"]:
            _is_over = True
        elif (
            new_inning > 9
            and (not home_team_up)
            and self.score["away"] > self.score["home"]
        ):
            _is_over = True

        # Fully reset the state, only carry over the score
        new_state = GameState(
            score=self.score,
            home_team_up=home_team_up,
            inning=new_inning,
            manfred=self.manfred,
            _is_over=_is_over,
        )

        # If its post 2020, we add an player to second base to start extra
        # innings
        if self.manfred and new_inning > 9:
            new_state = new_state.add_runner(Base.SECOND)

        return new_state

    @property
    def is_over(self) -> bool:
        """
        This logic is a little complicated since we need to handle both walk
        offs and games ending naturally. _is_over handles games ending naturally
        at the inning break. We need to check for both of these seperately since
        there can be amiguity as to the ordering of events. For example a state
        with a zero out home run and a state that had a home run in the previous
        inning.
        """
        if self._is_over:
            return True

        # Only home team can walk it off
        if (
            self.inning >= 9
            and self.home_team_up
            and self.score["home"] > self.score["away"]
        ):
            return True

        return False

    def add_runs(self, runs: int) -> "GameState":
        at_bat = self.at_bat()
        score = self.score
        new_score = self.score.get(at_bat) + runs
        new_score_map = self.score.set(at_bat, new_score)
        return replace(self, score=new_score_map)

    def empty_bases(self) -> "GameState":
        return replace(self, bases=EMPTY_BASES)

    def add_home_team_score(self) -> "GameState":
        return self._add_run(is_home_team=True)

    def add_away_team_score(self) -> "GameState":
        return self._add_run(is_home_team=False)

    def add_runner(self, base_idx: int) -> "GameState":
        new_bases = self.bases.set(base_idx, True)
        return replace(self, bases=new_bases)

    def get_num_runners(self):
        return sum([int(self.bases[i]) for i in list(Base)])

    def home_run(self) -> "GameState":
        num_runners = self.get_num_runners()
        new_state = self.add_runs(num_runners + 1)
        new_state = new_state.empty_bases()
        return new_state

    def force_advance(self, base: Base) -> "GameState":
        # If there is a runner on the base we are trying to move to, first move
        # that runner

        new_state = self
        if self.bases[base.next_base()]:
            new_state = self.force_advance(base.next_base())

        new_state = new_state.remove_runner(base)
        new_state = new_state.add_runner(base.next_base())
        return new_state

    def remove_runner(self, base: Base) -> "GameState":
        new_bases = self.bases

        # There is always a batter, don't remove
        if base == Base.BATTER:
            return self

        # Can't remove a runner that doesn't exist
        assert new_bases.get(base), f"Tried to remove a runner on {base}"

        new_bases = new_bases.set(base, False)

        return replace(self, bases=new_bases)

    def add_runner(self, base: Base) -> "GameState":
        new_bases = self.bases
        new_state = self

        # Can't add a runner to a base with someone on it
        assert not new_bases.get(base), f"Tried to remove a runner from {base}"

        if base != Base.HOME:
            new_bases = new_bases.set(base, True)
        else:
            # Moving a runner to home is the same as scoring a run
            new_state = new_state.add_runs(1)

        return replace(new_state, bases=new_bases)

    def get_base_runners(self) -> List[Base]:
        base_runners = [Base.BATTER]
        for b in Base:
            if self.bases[b]:
                base_runners.append(b)

        return base_runners

    def is_tie(self) -> bool:
        return self.score["home"] == self.score["away"]
