# TTP feasibility related checks for beam search nodes
from math import ceil


from lib.ttp_instance import TTPInstance
from lib.ttp_states import State, Node, update_state


# check whether game can be played for which both teams have to be in the same round and the at-most/no-repeat
# constraints must not be violated
def game_allowed(ttp_instance: TTPInstance, state: State, away_team: int, home_team: int):
    return state.rounds[away_team - 1] == state.rounds[home_team - 1] and state.possible_away_streaks[
        away_team - 1] > 0 and state.possible_home_stands[home_team - 1] > 0 and (not ttp_instance.no_repeat or (
            state.forbidden_opponents[away_team - 1] != home_team and state.forbidden_opponents[
        home_team - 1] != away_team))


def delta_at_most_check(ttp_instance: TTPInstance, node: Node, away_team: int, home_team: int,
                        away_team_home_games_left: int, away_team_away_games_left: int,
                        home_team_home_games_left: int, home_team_away_games_left: int):
    if home_team_home_games_left * ttp_instance.streak_limit + ttp_instance.streak_limit < home_team_away_games_left:
        return False
    if away_team_away_games_left * ttp_instance.streak_limit + ttp_instance.streak_limit < away_team_home_games_left:
        return False

    return True


# if after playing (away_team, home_team) only two games are left for one of the teams, that are against the same
# opponent, we would violate no-repeat for sure
def delta_no_repeat_check(ttp_instance: TTPInstance, node: Node, away_team: int, home_team: int,
                          away_team_home_games_left: int, away_team_away_games_left: int,
                          home_team_home_games_left: int, home_team_away_games_left: int):
    if (away_team_home_games_left != 1 or away_team_away_games_left != 1) and (
            home_team_home_games_left != 1 or home_team_away_games_left != 1):
        return True

    if away_team_home_games_left == 1 and away_team_away_games_left == 1:
        away_games = node.away_games_left_by_team[away_team - 1]
        home_games = node.home_games_left_by_team[away_team - 1]
        if away_games[0] == home_team:
            remaining_away_opponent = away_games[1]
        else:
            remaining_away_opponent = away_games[0]
        remaining_home_opponent = home_games[0]
        if remaining_away_opponent == remaining_home_opponent:
            return False

    if home_team_home_games_left == 1 and home_team_away_games_left == 1:
        away_games = node.away_games_left_by_team[home_team - 1]
        home_games = node.home_games_left_by_team[home_team - 1]
        if home_games[0] == away_team:
            remaining_home_opponent = home_games[1]
        else:
            remaining_home_opponent = home_games[0]
        remaining_away_opponent = away_games[0]
        if remaining_away_opponent == remaining_home_opponent:
            return False

    return True


# infer whether there cannot be a dead team after game (away_team, home_team)
def delta_infer_no_dead_team(ttp_instance: TTPInstance, layer: int, state: State, team: int, team_home_games_left: int,
                             team_away_games_left: int, teams_away_streak_limit_hit_last_round: int,
                             teams_away_streak_limit_hit_current_round: int, teams_home_stand_limit_hit_last_round: int,
                             teams_home_stand_limit_hit_current_round: int):
    worst_case_forbidden_games = 0

    games_per_round = ttp_instance.n / 2
    current_round = ceil(layer / games_per_round)
    games_played_in_this_round = layer % games_per_round

    if current_round > state.rounds[team - 1]:
        worst_case_forbidden_games += min(2 * games_played_in_this_round, team_home_games_left)
        worst_case_forbidden_games += min(2 * games_played_in_this_round, team_away_games_left)

    if current_round == state.rounds[team - 1]:
        if state.positions[team - 1] != team:
            if state.possible_away_streaks[team - 1] == 0:
                worst_case_forbidden_games += team_away_games_left
                worst_case_forbidden_games += min(team_home_games_left, teams_away_streak_limit_hit_current_round - 1)
            else:
                worst_case_forbidden_games += min(ttp_instance.n - 1,
                                                  teams_away_streak_limit_hit_current_round + teams_home_stand_limit_hit_current_round)
        else:
            if state.possible_home_stands[team - 1] == 0:
                worst_case_forbidden_games += team_home_games_left
                worst_case_forbidden_games += min(team_away_games_left, teams_home_stand_limit_hit_current_round - 1)
            else:
                worst_case_forbidden_games += min(ttp_instance.n - 1,
                                                  teams_away_streak_limit_hit_current_round + teams_home_stand_limit_hit_current_round)
    else:
        if state.positions[team - 1] != team:
            if state.possible_away_streaks[team - 1] == 0:
                worst_case_forbidden_games += team_away_games_left
                worst_case_forbidden_games += min(team_home_games_left, teams_away_streak_limit_hit_last_round - 1)
            else:
                worst_case_forbidden_games += min(ttp_instance.n - 1,
                                                  teams_away_streak_limit_hit_last_round + teams_home_stand_limit_hit_last_round)
        else:
            if state.possible_home_stands[team - 1] == 0:
                worst_case_forbidden_games += team_home_games_left
                worst_case_forbidden_games += min(team_away_games_left, teams_home_stand_limit_hit_last_round - 1)
            else:
                worst_case_forbidden_games += min(ttp_instance.n - 1,
                                                  teams_away_streak_limit_hit_last_round + teams_home_stand_limit_hit_last_round)

    if state.forbidden_opponents[team - 1] != -1:
        worst_case_forbidden_games += 1

    return worst_case_forbidden_games < (team_away_games_left + team_home_games_left)


# the incremental variant of game_allowed, where we check whether playing the game (away_team, home_team) would be
# allowed
def delta_game_allowed(ttp_instance: TTPInstance, state: State, team: int, opponent: int, away_team: int,
                       home_team: int):
    if state.rounds[team - 1] < state.rounds[opponent - 1]:
        return False
    elif state.rounds[team - 1] == state.rounds[opponent - 1]:
        return game_allowed(ttp_instance, state, away_team, home_team)
    else:
        if team == away_team:
            return state.possible_away_streaks[team - 1] > 0
        else:
            return state.possible_home_stands[team - 1] > 0


# checks whether there would be a team left after game (away_team, home_team), that has no permitted games,
# a dead team, O(n^2)
def delta_check_dead_team(ttp_instance: TTPInstance, node: Node, away_team: int, home_team: int):
    state = update_state(ttp_instance, node.state, away_team, home_team, node.number_of_away_games_left[home_team - 1],
                         node.number_of_home_games_left[away_team - 1])

    games_per_round = ttp_instance.n / 2
    current_round = ceil((node.layer + 1) / games_per_round)
    games_played_in_this_round = (node.layer + 1) % games_per_round

    teams_away_streak_limit_hit_last_round = node.teams_away_streak_limit_hit_last_round
    teams_away_streak_limit_hit_current_round = node.teams_away_streak_limit_hit_current_round
    teams_home_stand_limit_hit_last_round = node.teams_home_stand_limit_hit_last_round
    teams_home_stand_limit_hit_current_round = node.teams_home_stand_limit_hit_current_round

    if games_played_in_this_round == 1:
        teams_home_stand_limit_hit_last_round = teams_home_stand_limit_hit_current_round
        teams_away_streak_limit_hit_last_round = teams_away_streak_limit_hit_current_round
        teams_home_stand_limit_hit_current_round = 0
        teams_away_streak_limit_hit_current_round = 0

    if node.state.possible_home_stands[away_team - 1] == 0:
        teams_home_stand_limit_hit_last_round -= 1
    if state.possible_away_streaks[away_team - 1] == 0:
        teams_away_streak_limit_hit_current_round += 1
    if state.possible_home_stands[away_team - 1] == 0:
        teams_home_stand_limit_hit_current_round += 1
    if node.state.possible_away_streaks[home_team - 1] == 0:
        teams_away_streak_limit_hit_last_round -= 1
    if state.possible_home_stands[home_team - 1] == 0:
        teams_home_stand_limit_hit_current_round += 1
    if state.possible_away_streaks[home_team - 1] == 0:
        teams_away_streak_limit_hit_current_round += 1

    for team in range(1, ttp_instance.n + 1):
        if team == away_team:
            team_away_games_left = node.number_of_away_games_left[team - 1] - 1
        else:
            team_away_games_left = node.number_of_away_games_left[team - 1]

        if team == home_team:
            team_home_games_left = node.number_of_home_games_left[team - 1] - 1
        else:
            team_home_games_left = node.number_of_home_games_left[team - 1]

        team_games_left = team_home_games_left + team_away_games_left

        if team_games_left == 0:
            continue

        if delta_infer_no_dead_team(ttp_instance, node.layer + 1, state, team, team_home_games_left,
                                    team_away_games_left, teams_away_streak_limit_hit_last_round,
                                    teams_away_streak_limit_hit_current_round, teams_home_stand_limit_hit_last_round,
                                    teams_home_stand_limit_hit_current_round):
            continue

        witness_found = False
        for opponent in node.away_games_left_by_team[team - 1]:
            if state.games_left[team - 1][opponent - 1] and delta_game_allowed(ttp_instance, state, team, opponent,
                                                                               team, opponent):
                witness_found = True
                break
        if not witness_found:
            for opponent in node.home_games_left_by_team[team - 1]:
                if state.games_left[opponent - 1][team - 1] and delta_game_allowed(ttp_instance, state, team, opponent,
                                                                                   opponent, team):
                    witness_found = True
                    break
        if not witness_found:
            # print("no witness for %d" % team)
            return False

    return True


# incrementally check whether playing (away_team, home_team) would result into a state without a feasible completion
# for certain according to our checks
def delta_feasibility_check(ttp_instance: TTPInstance, node: Node, away_team: int, home_team: int,
                            dead_teams_check: bool):
    away_team_home_games_left = node.number_of_home_games_left[away_team - 1]
    away_team_away_games_left = node.number_of_away_games_left[away_team - 1] - 1
    home_team_home_games_left = node.number_of_home_games_left[home_team - 1] - 1
    home_team_away_games_left = node.number_of_away_games_left[home_team - 1]

    if not delta_at_most_check(ttp_instance, node, away_team, home_team, away_team_home_games_left,
                               away_team_away_games_left, home_team_home_games_left, home_team_away_games_left):
        return False

    if ttp_instance.no_repeat and not delta_no_repeat_check(ttp_instance, node, away_team, home_team,
                                                            away_team_home_games_left, away_team_away_games_left,
                                                            home_team_home_games_left, home_team_away_games_left):
        return False

    if dead_teams_check and not delta_check_dead_team(ttp_instance, node, away_team, home_team):
        return False

    return True
