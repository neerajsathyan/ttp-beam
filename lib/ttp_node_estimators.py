import numpy as np

from lib.ttp_instance import TTPInstance
from lib.ttp_util import mask_teams_left


def heuristic_estimate(ttp_instance: TTPInstance, team: int, teams_left: np.array(1,int), number_of_home_games_left: int, position: int, streak: int, bounds_by_state: np.array(4, int), heuristic_estimates_cache: None):
    streak2 = streak
    if len(teams_left) == 0:
        return ttp_instance.d[position-1][team-1]

    if team == position:
        streak2 = 0
    if streak2 == ttp_instance.streak_limit:
        return ttp_instance.d[position - 1][team - 1] + bounds_by_state[team - 1][mask_teams_left(team, teams_left) - 1][team - 1][0]
    else:
        return min(ttp_instance.d[position-1][team-1] + bounds_by_state[team-1][mask_teams_left(team, teams_left)-1][team-1][0], bounds_by_state[team-1][mask_teams_left(team, teams_left)-1][position-1][streak])
