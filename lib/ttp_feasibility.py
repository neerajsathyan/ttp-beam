from lib.ttp_instance import TTPInstance
from lib.ttp_states import State, Node


# check whether game can be played for which both teams have to be in the same round and the at-most/no-repeat
# constraints must not be violated
def game_allowed(ttp_instance: TTPInstance, state: State, away_team: int, home_team: int):
    return state.rounds[away_team - 1] == state.rounds[home_team - 1] and state.possible_away_streaks[
        away_team - 1] > 0 and state.possible_home_stands[home_team - 1] > 0 and (not ttp_instance.no_repeat or (
            state.forbidden_opponents[away_team - 1] != home_team and state.forbidden_opponents[
        home_team - 1] != away_team))


def delta_at_most_check(ttp_instance, node, away_team, home_team, away_team_home_games_left, away_team_away_games_left,
                        home_team_home_games_left, home_team_away_games_left):
        pass

# incrementally check whether playing (away_team, home_team) would result into a state without a feasible completion
# for certain according to our checks
def delta_feasibility_check(ttp_instance: TTPInstance, node: Node, away_team: int, home_team: int,
                            dead_teams_check: bool):
    away_team_home_games_left = node.number_of_home_games_left[away_team-1]
    away_team_away_games_left = node.number_of_away_games_left[away_team-1]-1
    home_team_home_games_left = node.number_of_home_games_left[home_team-1]-1
    home_team_away_games_left = node.number_of_away_games_left[home_team-1]

    if not delta_at_most_check(ttp_instance, node, away_team, home_team, away_team_home_games_left, away_team_away_games_left, home_team_home_games_left, home_team_away_games_left):
        return False

    if ttp_instance.no_repeat and not delta_no_repeat_check(ttp_instance, node, away_team, home_team, away_team_home_games_left, away_team_away_games_left, home_team_home_games_left, home_team_away_games_left):
        return False

    if dead_teams_check and not delta_check_dead_team(ttp_instance, node, away_team, home_team):
        return False

    return True
