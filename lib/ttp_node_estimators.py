from math import ceil

import numpy as np

from lib.ttp_instance import TTPInstance
from lib.ttp_util import mask_teams_left


# the remaining number of home games imply a minimum of vehicles (away streaks) for feasibility...
# Checked
def min_vehicles_by_home_games(ttp_instance: TTPInstance, team: int, teams_left: np.array(1, int),
                               number_of_home_games_left: int, position: int, streak: int):
    if team == position:
        return max(ceil((streak + number_of_home_games_left) / ttp_instance.streak_limit) - 1, 0)
    else:
        return ceil(number_of_home_games_left / ttp_instance.streak_limit)


# .. so do the remaining away games
# Checked
def min_vehicles_by_away_games(ttp_instance: TTPInstance, team: int, teams_left: np.array(1, int),
                               number_of_home_games_left: int, position: int, streak: int):
    if team == position:
        return ceil(len(teams_left) / ttp_instance.streak_limit)
    else:
        return ceil((streak + len(teams_left)) / ttp_instance.streak_limit)


# the number of home games left imply a maximum number of vehicles
# Checked
def max_vehicles_by_home_games(ttp_instance: TTPInstance, team: int, teams_left: np.array(1, int),
                               number_of_home_games_left: int, position: int, streak: int):
    return 1 + number_of_home_games_left


# if the streak limit > 1, it is never optimal to have more the one streak with length one, since they can be merged
# Checked
def max_vehicles_by_away_games(ttp_instance: TTPInstance, team: int, teams_left: np.array(1, int),
                               number_of_home_games_left: int, position: int, streak: int):
    if team == position:
        return ceil(len(teams_left) / min(ttp_instance.streak_limit, 2))
    else:
        if streak == 1 and ttp_instance.streak_limit > 1:
            return 1 + ceil((len(teams_left) - 1) / 2)
        else:
            return 1 + ceil(len(teams_left) / min(ttp_instance.streak_limit, 2))


# the minimum number and maximum number of vehicles needed for solving the CVRPH problem for a team, inferred by the
# feasibility and optimality considerations the current streak counts as a vehicle
# Checked
def min_max_vehicles(ttp_instance: TTPInstance, team: int, teams_left: np.array(1, int), number_of_home_games_left: int,
                     position: int, streak: int):
    minimum_vehicles_needed = max(
        min_vehicles_by_home_games(ttp_instance, team, teams_left, number_of_home_games_left, position, streak),
        min_vehicles_by_away_games(ttp_instance, team, teams_left, number_of_home_games_left, position, streak))
    maximum_vehicles_allowed = min(
        max_vehicles_by_home_games(ttp_instance, team, teams_left, number_of_home_games_left, position, streak),
        max_vehicles_by_away_games(ttp_instance, team, teams_left, number_of_home_games_left, position, streak))
    maximum_vehicles_allowed = max(maximum_vehicles_allowed, minimum_vehicles_needed)

    return minimum_vehicles_needed, maximum_vehicles_allowed


# this is similar to min max vehicles, but for the precalculated lower bound values we do not count the vehicles (
# i.e. streaks) but the home stands, where the return home at the end also counts as home stand
# Checked
def min_max_home_stands(ttp_instance: TTPInstance, team: int, teams_left: np.array(1, int),
                        number_of_home_games_left: int, position: int, streak: int):
    minimum_vehicles_needed, maximum_vehicles_allowed = min_max_vehicles(ttp_instance, team, teams_left,
                                                                         number_of_home_games_left, position, streak)

    if team == position:
        return minimum_vehicles_needed + 1, maximum_vehicles_allowed + 1
    else:
        return minimum_vehicles_needed, maximum_vehicles_allowed


# CVRP or TSP
# heuristic estimates based on precalculated exact cvrp bounds
# Checked
def heuristic_estimate(ttp_instance: TTPInstance, team: int, teams_left: np.array(1, int),
                       number_of_home_games_left: int, position: int, streak: int, bounds_by_state: np.array(4, int),
                       heuristic_estimates_cache: None):
    if len(teams_left) == 0:
        return ttp_instance.d[position - 1][team - 1]

    if team == position:
        streak = 0
    if streak == ttp_instance.streak_limit:
        return ttp_instance.d[position - 1][team - 1] + bounds_by_state[team - 1][mask_teams_left(team, teams_left) - 1][team - 1][0]
    else:
        return min(ttp_instance.d[position - 1][team - 1] +
                   bounds_by_state[team - 1][mask_teams_left(team, teams_left) - 1][team - 1][0],
                   bounds_by_state[team - 1][mask_teams_left(team, teams_left) - 1][position - 1][streak])


# heuristic estimates based on precalculated exact cvrph bounds
# Checked
def heuristic_estimate_cvrph(ttp_instance: TTPInstance, team: int, teams_left: np.array(1, int),
                       number_of_home_games_left: int, position: int, streak: int, bounds_by_state: np.array(5, int),
                       heuristic_estimates_cache: None):
    if len(teams_left) == 0:
        return ttp_instance.d[position - 1][team - 1]

    if team == position:
        away_streak = 0
    else:
        away_streak = streak

    # we hit the away streak limit and have to return home, adding a detour
    if team != position and streak == ttp_instance.streak_limit:
        detour = ttp_instance.d[position - 1][team - 1]
        away_streak = 0
        streak = 0
        position = team
    else:
        detour = 0

    minimum_home_stands_needed, maximum_home_stands_allowed = min_max_home_stands(ttp_instance, team, teams_left,
                                                                                  number_of_home_games_left, position,
                                                                                  streak)

    best_bound_direct = np.amin(
        bounds_by_state[team - 1][mask_teams_left(team, teams_left) - 1][position - 1][away_streak][
        minimum_home_stands_needed - 1:maximum_home_stands_allowed])
    if team != position:
        minimum_home_stands_needed_detour, maximum_home_stands_allowed_detour = min_max_home_stands(ttp_instance, team,
                                                                                                    teams_left,
                                                                                                    number_of_home_games_left,
                                                                                                    team, 0)
        best_bound_via_home = ttp_instance.d[position - 1][team - 1] + np.amin(
            bounds_by_state[team - 1][mask_teams_left(team, teams_left) - 1][team - 1][0][
            minimum_home_stands_needed_detour - 1:maximum_home_stands_allowed_detour])
    else:
        best_bound_via_home = np.iinfo(np.int32).max

    return detour + min(best_bound_direct, best_bound_via_home)
