# TTP states related classes and basic functions
from copy import copy
from typing import Union

import numpy as np
import xxhash

from lib.ttp_instance import TTPInstance


class State:
    def __init__(self, games_left: np.array(2, bool), forbidden_opponents: np.array(1, int), rounds: np.array(1, int),
                 positions: np.array(1, int), possible_away_streaks: np.array(1, int),
                 possible_home_stands: np.array(1, int)):
        self.games_left = games_left
        self.forbidden_opponents = forbidden_opponents
        self.rounds = rounds
        self.positions = positions
        self.possible_away_streaks = possible_away_streaks
        self.possible_home_stands = possible_home_stands

    def __hash__(self):
        # return hash((self.games_left, self.forbidden_opponents, self.rounds, self.positions,
        # self.possible_away_streaks, self.possible_home_stands))
        return xxhash.xxh32_intdigest(self.games_left, seed=xxhash.xxh32_intdigest(self.forbidden_opponents, seed=xxhash.xxh32_intdigest(self.rounds, seed=xxhash.xxh32_intdigest(self.positions, seed=xxhash.xxh32_intdigest(self.possible_away_streaks, seed=xxhash.xxh32_intdigest(self.possible_home_stands))))))

    def __eq__(self, other):
        return (np.array_equal(self.games_left, other.games_left) and np.array_equal(self.forbidden_opponents,
                                                                                     other.forbidden_opponents)
                and np.array_equal(self.rounds, other.rounds) and np.array_equal(self.positions, other.positions)
                and np.array_equal(self.possible_away_streaks, other.possible_away_streaks) and np.array_equal(
                    self.possible_home_stands, other.possible_home_stands))


class Node:
    def __init__(self):
        self.layer: int = 0
        self.shortest_path_length: int = 0
        self.heuristic_estimate: int = np.iinfo(np.int64).max
        self.games_left: int = 0
        self.state: Union[State, None] = None
        self.heuristic_estimates: np.array(1, int) = []
        self.noise: float = 0.0
        self.solution: np.array(1, (int, int)) = []
        self.number_of_away_games_left: np.array(1, int) = []
        self.number_of_home_games_left: np.array(1, int) = []
        self.away_games_left_by_team: np.array(2, int) = []
        self.home_games_left_by_team: np.array(2, int) = []
        self.teams_away_streak_limit_hit_last_round: int = 0
        self.teams_away_streak_limit_hit_current_round: int = 0
        self.teams_home_stand_limit_hit_last_round: int = 0
        self.teams_home_stand_limit_hit_current_round: int = 0

    def __hash__(self):
        return hash((self.layer, self.shortest_path_length, self.heuristic_estimate, self.games_left, self.state,
                     self.heuristic_estimates, self.noise, self.solution, self.number_of_away_games_left,
                     self.number_of_home_games_left, self.away_games_left_by_team, self.home_games_left_by_team,
                     self.teams_away_streak_limit_hit_last_round, self.teams_away_streak_limit_hit_current_round,
                     self.teams_home_stand_limit_hit_last_round, self.teams_home_stand_limit_hit_current_round))

    def __eq__(self, other):
        return (self.shortest_path_length + self.heuristic_estimate + self.noise, self.shortest_path_length) == (
            other.shortest_path_length + other.heuristic_estimate + other.noise, other.shortest_path_length)

    def __gt__(self, other):
        return (self.shortest_path_length + self.heuristic_estimate + self.noise, self.shortest_path_length) > (
            other.shortest_path_length + other.heuristic_estimate + other.noise, other.shortest_path_length)

    def __lt__(self, other):
        return (self.shortest_path_length + self.heuristic_estimate + self.noise, self.shortest_path_length) < (
            other.shortest_path_length + other.heuristic_estimate + other.noise, other.shortest_path_length)


# a state transitions copies the existing states and makes corresponding updates determined by the game being played
def update_state(ttp_instance: TTPInstance, state: State, away_team: int, home_team: int,
                 home_team_number_of_away_games_left: int, away_team_number_of_home_games_left: int):
    games_left = copy(state.games_left)
    games_left[away_team - 1][home_team - 1] = False

    rounds = copy(state.rounds)
    rounds[away_team - 1] += 1
    rounds[home_team - 1] += 1

    positions = copy(state.positions)
    positions[away_team - 1] = home_team
    positions[home_team - 1] = home_team

    possible_away_streaks = copy(state.possible_away_streaks)
    possible_away_streaks[away_team - 1] -= 1
    possible_away_streaks[home_team - 1] = min(ttp_instance.streak_limit, home_team_number_of_away_games_left)

    possible_home_stands = copy(state.possible_home_stands)
    possible_home_stands[away_team - 1] = min(ttp_instance.streak_limit, away_team_number_of_home_games_left)
    possible_home_stands[home_team - 1] -= 1

    forbidden_opponents = copy(state.forbidden_opponents)
    if ttp_instance.no_repeat:
        if games_left[home_team - 1][away_team - 1]:
            forbidden_opponents[away_team - 1] = home_team
            forbidden_opponents[home_team - 1] = away_team
        else:
            forbidden_opponents[away_team - 1] = -1
            forbidden_opponents[home_team - 1] = -1

        for team in range(1, ttp_instance.n + 1):
            if team != away_team and team != home_team and (
                    forbidden_opponents[team - 1] == away_team or forbidden_opponents[team - 1] == home_team):
                forbidden_opponents[team - 1] = -1

    return State(games_left, forbidden_opponents, rounds, positions, possible_away_streaks, possible_home_stands)
