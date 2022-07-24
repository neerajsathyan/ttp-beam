# TTP states related classes and basic functions
from typing import Union

import numpy as np


class State:
    def __init__(self, games_left: np.array(2, bool), forbidden_opponents: np.array(1, int), rounds: np.array(1, int),
                 positions: np.array(1, int), possible_away_streaks: np.array(1, int),
                 possible_home_stands: np.array(1, int)):
        self.games_let = games_left
        self.forbidden_opponents = forbidden_opponents
        self.rounds = rounds
        self.positions = positions
        self.possible_away_streaks = possible_away_streaks
        self.possible_home_stands = possible_home_stands

    def __hash__(self):
        return hash((self.games_let, self.forbidden_opponents, self.rounds, self.positions, self.possible_away_streaks,
                     self.possible_home_stands))

    def __eq__(self, other):
        return (self.games_let, self.forbidden_opponents, self.rounds, self.positions, self.possible_away_streaks,
                self.possible_home_stands) == (
                   other.games_let, other.forbidden_opponents, other.rounds, other.positions,
                   other.possible_away_streaks,
                   other.possible_home_stands)


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
        return (self.layer, self.shortest_path_length, self.heuristic_estimate, self.games_left, self.state,
                self.heuristic_estimates, self.noise, self.solution, self.number_of_away_games_left,
                self.number_of_home_games_left, self.away_games_left_by_team, self.home_games_left_by_team,
                self.teams_away_streak_limit_hit_last_round, self.teams_away_streak_limit_hit_current_round,
                self.teams_home_stand_limit_hit_last_round, self.teams_home_stand_limit_hit_current_round) == (
               other.layer, other.shortest_path_length, other.heuristic_estimate, other.games_left, other.state,
               other.heuristic_estimates, other.noise, other.solution, other.number_of_away_games_left,
               other.number_of_home_games_left, other.away_games_left_by_team, other.home_games_left_by_team,
               other.teams_away_streak_limit_hit_last_round, other.teams_away_streak_limit_hit_current_round,
               other.teams_home_stand_limit_hit_last_round, other.teams_home_stand_limit_hit_current_round)
