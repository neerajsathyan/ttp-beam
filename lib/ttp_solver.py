from copy import copy
from math import ceil
from typing import Union, Dict, Tuple, List

import enum
import numpy as np

from lib import heapdict
from lib.ttp_feasibility import game_allowed, delta_feasibility_check
from lib.ttp_instance import TTPInstance
from lib.ttp_node_estimators import heuristic_estimate, heuristic_estimate_cvrph
from lib.ttp_states import Node, State, update_state


class Statistics:
    def __init__(self):
        self.feasibility_checks_failed: int = 0
        self.optimality_checks_failed: int = 0
        self.construction_time: float = 0
        self.heuristic_estimates_cache_size: int = 0


class GameResult(enum.Enum):
    played = 1
    disallowed = 2
    infeasible = 3
    suboptimal = 4


def root_state(ttp_instance: TTPInstance):
    games_left = np.ones((ttp_instance.n, ttp_instance.n), bool)
    games_left[np.diag_indices(ttp_instance.n)] = False
    forbidden_opponents = -np.ones(ttp_instance.n, dtype=int)
    rounds = np.zeros(ttp_instance.n, dtype=int)
    positions = np.arange(1, ttp_instance.n + 1)
    possible_away_streaks = ttp_instance.streak_limit * np.ones(ttp_instance.n, dtype=int)
    possible_home_stands = ttp_instance.streak_limit * np.ones(ttp_instance.n, dtype=int)
    return State(games_left, forbidden_opponents, rounds, positions, possible_away_streaks, possible_home_stands)


def terminal_state(ttp_instance: TTPInstance):
    games_left = np.zeros((ttp_instance.n, ttp_instance.n), bool)
    forbidden_opponents = -np.ones(ttp_instance.n, dtype=int)
    rounds = np.ones(ttp_instance.n, dtype=int) * (2 * ttp_instance.n - 1)
    positions = np.arange(1, ttp_instance.n + 1)
    possible_away_streaks = np.zeros(ttp_instance.n, dtype=int)
    possible_home_stands = np.zeros(ttp_instance.n, dtype=int)
    return State(games_left, forbidden_opponents, rounds, positions, possible_away_streaks, possible_home_stands)


def go_home(ttp_instance: TTPInstance, node: Node, terminal: Node):
    weight = np.sum(
        map(lambda x: ttp_instance.d[node.state.positions[x - 1] - 1][x - 1], np.arange(1, ttp_instance.n + 1)))
    if node.shortest_path_length + weight < terminal.shortest_path_length:
        terminal.shortest_path_length = node.shortest_path_length + weight
        terminal.solution = copy(node.solution)
    return terminal


def next_team(ttp_instance: TTPInstance, node: Node, teams_permutation: []):
    return np.argmin(
        map(lambda x: (node.state.rounds[x - 1], teams_permutation[x - 1]), np.arange(1, ttp_instance.n + 1))) + 1


# Optional Gaussian noise for f-values..
def noise_for_guidance_value(sigma: float, layer: int, first_k_layers_noisy: int):
    if first_k_layers_noisy == -1 or layer <= first_k_layers_noisy:
        return np.random.randn() * sigma
    else:
        return 0.0


# incrementally checks whether playing (away_team, home_team) would result into a node that is admissible for the
# beam by its f-value, otherwise it is not expanded
def delta_optimality_check(ttp_instance: TTPInstance, node: Node, away_team: int, home_team: int, beam: heapdict,
                           beam_width: int,
                           bounds_by_state: Union[List[List[List[List[int]]]], List[List[List[List[List[int]]]]], None],
                           heuristic_estimates_cache: Union[
                               Dict[Tuple[int, int, int, int], int], int, Dict[
                                   Tuple[int, int, int, int, int, int], int], None], noise: float):
    weight = ttp_instance.d[node.state.positions[away_team - 1] - 1][home_team - 1] + \
             ttp_instance.d[node.state.positions[home_team - 1] - 1][home_team - 1]
    away_team_away_teams_left = copy(node.away_games_left_by_team[away_team - 1])
    away_team_away_teams_left = np.delete(away_team_away_teams_left,
                                          np.searchsorted(away_team_away_teams_left, home_team))
    home_team_away_teams_left = node.away_games_left_by_team[home_team - 1]
    away_team_number_of_home_games_left = node.number_of_home_games_left[away_team - 1]
    home_team_number_of_home_games_left = node.number_of_home_games_left[home_team - 1] - 1

    heuristic_estimate_delta = -(node.heuristic_estimates[away_team - 1] + node.heuristic_estimates[home_team - 1])
    heuristic_estimate_delta += heuristic_estimate(ttp_instance, away_team, away_team_away_teams_left,
                                                   away_team_number_of_home_games_left, home_team,
                                                   ttp_instance.streak_limit - (
                                                           node.state.possible_away_streaks[away_team - 1] - 1),
                                                   bounds_by_state, heuristic_estimates_cache)
    heuristic_estimate_delta += heuristic_estimate(ttp_instance, home_team, home_team_away_teams_left,
                                                   home_team_number_of_home_games_left, home_team,
                                                   ttp_instance.streak_limit - (
                                                           node.state.possible_home_stands[home_team - 1] - 1),
                                                   bounds_by_state, heuristic_estimates_cache)

    if len(beam) >= beam_width:
        worst_state, worst_node = beam.peekitem()
        if worst_node.shortest_path_length + worst_node.heuristic_estimate + worst_node.noise <= node.shortest_path_length + weight + node.heuristic_estimate + heuristic_estimate_delta + noise:
            return False

    return True


def calc_streak_limit_hits(ttp_instance: TTPInstance, node: Node, state: State, old_node: Node, away_team: int,
                           home_team: int):
    games_per_round = ttp_instance.n / 2
    current_round = ceil(node.layer / games_per_round)
    games_played_in_this_round = node.layer % games_per_round

    node.teams_home_stand_limit_hit_last_round = old_node.teams_home_stand_limit_hit_last_round
    node.teams_away_streak_limit_hit_current_round = old_node.teams_away_streak_limit_hit_current_round
    node.teams_away_streak_limit_hit_last_round = old_node.teams_away_streak_limit_hit_last_round
    node.teams_home_stand_limit_hit_current_round = old_node.teams_home_stand_limit_hit_current_round

    if games_played_in_this_round == 1:
        node.teams_home_stand_limit_hit_last_round = node.teams_home_stand_limit_hit_current_round
        node.teams_away_streak_limit_hit_last_round = node.teams_away_streak_limit_hit_current_round
        node.teams_home_stand_limit_hit_current_round = 0
        node.teams_away_streak_limit_hit_current_round = 0

    if old_node.state.possible_home_stands[away_team - 1] == 0:
        node.teams_home_stand_limit_hit_last_round -= 1
    if node.state.possible_away_streaks[away_team - 1] == 0:
        node.teams_away_streak_limit_hit_current_round += 1
    if node.state.possible_home_stands[away_team - 1] == 0:
        node.teams_home_stand_limit_hit_current_round += 1
    if old_node.state.possible_away_streaks[home_team - 1] == 0:
        node.teams_away_streak_limit_hit_last_round -= 1
    if node.state.possible_home_stands[home_team - 1] == 0:
        node.teams_home_stand_limit_hit_current_round += 1
    if node.state.possible_away_streaks[home_team - 1] == 0:
        node.teams_away_streak_limit_hit_current_round += 1

    assert node.teams_home_stand_limit_hit_last_round >= 0
    assert node.teams_away_streak_limit_hit_current_round >= 0
    assert node.teams_away_streak_limit_hit_current_round >= 0
    assert node.teams_home_stand_limit_hit_last_round >= 0


# perform a state transition by playing game (away_team, home_team)
def play_game(ttp_instance: TTPInstance, node: Node, away_team: int, home_team: int,
              bounds_by_state: Union[List[List[List[List[int]]]], List[List[List[List[List[int]]]]], None],
              heuristic_estimates_cache: Union[
                  Dict[Tuple[int, int, int, int], int], int, Dict[
                      Tuple[int, int, int, int, int, int], int], None], noise: float):
    new_node = Node()
    new_node.layer = node.layer + 1
    weight = ttp_instance.d[node.state.positions[away_team - 1] - 1][home_team - 1] + \
             ttp_instance.d[node.state.positions[home_team - 1] - 1][home_team - 1]
    new_node.shortest_path_length = node.shortest_path_length + weight
    new_node.games_left = node.games_left - 1
    new_node.heuristic_estimates = copy(node.heuristic_estimates)
    new_node.solution = copy(node.solution)
    new_node.number_of_home_games_left = copy(node.number_of_home_games_left)
    new_node.number_of_away_games_left = copy(node.number_of_away_games_left)
    new_node.away_games_left_by_team = copy(node.away_games_left_by_team)
    new_node.home_games_left_by_team = copy(node.home_games_left_by_team)
    new_node.away_games_left_by_team[away_team - 1] = copy(node.away_games_left_by_team[away_team - 1])
    new_node.home_games_left_by_team[home_team - 1] = copy(node.home_games_left_by_team[home_team - 1])
    new_node.number_of_home_games_left[home_team - 1] -= 1
    new_node.number_of_away_games_left[away_team - 1] -= 1
    new_node.away_games_left_by_team[away_team - 1] = np.delete(new_node.away_games_left_by_team[away_team - 1],
                                                                np.searchsorted(
                                                                    new_node.away_games_left_by_team[away_team - 1],
                                                                    home_team))
    new_node.home_games_left_by_team[home_team - 1] = np.delete(new_node.home_games_left_by_team[home_team - 1],
                                                                np.searchsorted(
                                                                    new_node.home_games_left_by_team[home_team - 1],
                                                                    away_team))
    new_node.state = update_state(ttp_instance, node.state, away_team, home_team,
                                  new_node.number_of_away_games_left[home_team - 1],
                                  new_node.number_of_home_games_left[away_team - 1])

    calc_streak_limit_hits(ttp_instance, new_node, new_node.state, node, away_team, home_team)

    new_node.solution.append((away_team, home_team))

    new_node.heuristic_estimates[away_team - 1] = heuristic_estimate(ttp_instance, away_team,
                                                                     new_node.away_games_left_by_team[away_team - 1],
                                                                     new_node.number_of_home_games_left[away_team - 1],
                                                                     new_node.state.positions[away_team - 1],
                                                                     ttp_instance.streak_limit -
                                                                     new_node.state.possible_away_streaks[
                                                                         away_team - 1], bounds_by_state,
                                                                     heuristic_estimates_cache)
    new_node.heuristic_estimates[home_team - 1] = heuristic_estimate(ttp_instance, home_team,
                                                                     new_node.away_games_left_by_team[home_team - 1],
                                                                     new_node.number_of_home_games_left[home_team - 1],
                                                                     new_node.state.positions[home_team - 1],
                                                                     ttp_instance.streak_limit -
                                                                     new_node.state.possible_home_stands[home_team - 1],
                                                                     bounds_by_state, heuristic_estimates_cache)

    new_node.heuristic_estimate = np.sum(new_node.heuristic_estimates)
    new_node.noise = noise
    return new_node, weight


# def haskey(pq: PriorityQueue, key: State):
#     for i in range(len(pq.queue)):
#         if pq.queue[i][1] == key:
#             return True, i
#     return False, -1


def check_state_exists(state: State, beam_dict: Dict[State, Tuple[Node, int, int]]):
    for state2 in beam_dict:
        if np.array_equal(state.games_left, state2.games_left) and np.array_equal(state.forbidden_opponents,
                                                                                  state2.forbidden_opponents) \
                and np.array_equal(state.rounds, state2.rounds) and np.array_equal(state.positions, state2.positions) \
                and np.array_equal(state.possible_away_streaks, state2.possible_away_streaks) \
                and np.array_equal(state.possible_home_stands, state2.possible_home_stands):
            return True
    return False


# conditionally incorporate node into beam or update existing node with same state, if shortest path to it has been
# found
def incorporate(parent: Node, new_node: Node, beam: heapdict, beam_width: int):
    # if haskey(beam, new_node.state)
    # if haskey(nodes, new_node.state)

    if new_node.state in beam.d.keys():
        # if check_state_exists(new_node.state, beam.d):
        existing_node = beam.get(new_node.state)
        if new_node < existing_node:
            beam[new_node.state] = new_node
            # beam.queue.pop(ind)
            # beam.put((new_node, new_node.state))
            # beam.queue[ind][0] = new_node
    else:
        if len(beam) < beam_width:
            beam[new_node.state] = new_node
        else:
            worst_state, worst_node = beam.peekitem()
            if new_node < worst_node:
                beam[new_node.state] = new_node
                beam.popitem()


# conditionally play (away_team, home_team), if it is currently allowed and would not result into an infeasible state
# (according to our checks) and suboptimal node given our current beam
def conditionally_play_game(ttp_instance: TTPInstance, node: Node, away_team: int, home_team: int, beam: heapdict,
                            beam_width: int, bounds_by_state: Union[
            List[List[List[List[int]]]], List[List[List[List[List[int]]]]], None],
                            heuristic_estimates_cache: Union[
                                Dict[Tuple[int, int, int, int], int], int, Dict[
                                    Tuple[int, int, int, int, int, int], int], None], dead_teams_check: bool,
                            noise: float):
    if game_allowed(ttp_instance, node.state, away_team, home_team):
        if delta_feasibility_check(ttp_instance, node, away_team, home_team, dead_teams_check):
            if delta_optimality_check(ttp_instance, node, away_team, home_team, beam, beam_width, bounds_by_state,
                                      heuristic_estimates_cache, noise):
                new_node, weight = play_game(ttp_instance, node, away_team, home_team, bounds_by_state,
                                             heuristic_estimates_cache, noise)
                incorporate(node, new_node, beam, beam_width)
                return GameResult.played
            else:
                return GameResult.suboptimal
        else:
            return GameResult.infeasible
    else:
        return GameResult.disallowed


def update(stats: Statistics, game_result: GameResult):
    if game_result == GameResult.infeasible:
        stats.feasibility_checks_failed += 1
    elif game_result == GameResult.suboptimal:
        stats.optimality_checks_failed += 1


def solution_to_rounds_matrix(ttp_instance: TTPInstance, solution: List[Tuple[int, int]]):
    copied_solution = copy(solution)
    rounds = 2 * ttp_instance.n - 2
    rounds_matrix = np.zeros((rounds, ttp_instance.n), dtype=int)
    for round in range(1, rounds + 1):
        for i in range(1, (ttp_instance.n / 2) + 1):
            game = solution.pop()
            rounds_matrix[round - 1][game[0]] = -game[1]
            rounds_matrix[round - 1][game[1]] = game[0]
    return rounds_matrix


def construct(ttp_instance: TTPInstance,
              bounds_by_state: Union[List[List[List[List[int]]]], List[List[List[List[List[int]]]]], None],
              beam_width: int, dead_teams_check: bool, randomized_team_order: bool, sigma_rel: float,
              heuristic_estimates_cache: Union[
                  Dict[Tuple[int, int, int, int], int], int, Dict[Tuple[int, int, int, int, int, int], int], None],
              first_k_layers_noisy: int = -1):
    root = Node()
    root.shortest_path_length = 0
    root.games_left = ttp_instance.n * (ttp_instance.n - 1)
    root.heuristic_estimates = np.zeros(ttp_instance.n, dtype=int)
    for team in range(1, ttp_instance.n + 1):
        teams_left = np.delete(np.arange(1, ttp_instance.n + 1), team - 1)
        if np.ndim(bounds_by_state) == 4:
            root.heuristic_estimates[team - 1] = heuristic_estimate(ttp_instance, team, teams_left, ttp_instance.n - 1,
                                                                    team, 0, bounds_by_state, heuristic_estimates_cache)
        else:
            root.heuristic_estimates[team - 1] = heuristic_estimate_cvrph(ttp_instance, team, teams_left,
                                                                          ttp_instance.n - 1,
                                                                          team, 0, bounds_by_state,
                                                                          heuristic_estimates_cache)
        root.away_games_left_by_team.append([])
        root.home_games_left_by_team.append([])
        root.away_games_left_by_team[team - 1] = teams_left
        root.home_games_left_by_team[team - 1] = teams_left
    root.heuristic_estimate = np.sum(root.heuristic_estimates)
    root.state = root_state(ttp_instance)
    root.number_of_home_games_left = np.ones(ttp_instance.n, dtype=int) * (ttp_instance.n - 1)
    root.number_of_away_games_left = np.ones(ttp_instance.n, dtype=int) * (ttp_instance.n - 1)

    sigma = root.heuristic_estimate * sigma_rel

    last_layer = root.games_left

    terminal = Node()
    terminal.shortest_path_length = np.iinfo(np.int64).max
    terminal.heuristic_estimates = np.zeros(ttp_instance.n, dtype=int)
    terminal.heuristic_estimate = 0
    terminal.state = terminal_state(ttp_instance)
    terminal.number_of_home_games_left = np.zeros(ttp_instance.n, dtype=int)
    terminal.number_of_away_games_left = np.zeros(ttp_instance.n, dtype=int)

    stats = Statistics()

    if randomized_team_order:
        teams_permutation = np.random.permutation(ttp_instance.n) + 1
    else:
        teams_permutation = np.arange(1, ttp_instance.n + 1)

    print("root heuristic estimate: %d" % root.heuristic_estimate)

    Q = [root]
    '''Variant of Queue that retrieves open entries in priority order (lowest first).

        Entries are typically tuples of the form:  (priority number, data).
        '''
    beam = heapdict.heapdict()

    for layer in range(0, last_layer + 1):
        print("layer %d size %d" % (layer, len(Q)))

        for node in Q:
            if node.games_left == 0:
                go_home(ttp_instance, node, terminal)
            else:
                team = next_team(ttp_instance, node, teams_permutation)

                for opponent in node.away_games_left_by_team[team]:
                    noise = noise_for_guidance_value(sigma, layer + 1, first_k_layers_noisy)
                    game_result = conditionally_play_game(ttp_instance, node, team, opponent, beam, beam_width,
                                                          bounds_by_state, heuristic_estimates_cache, dead_teams_check,
                                                          noise)
                    update(stats, game_result)

                for opponent in node.home_games_left_by_team[team]:
                    noise = noise_for_guidance_value(sigma, layer + 1, first_k_layers_noisy)
                    game_result = conditionally_play_game(ttp_instance, node, opponent, team, beam, beam_width,
                                                          bounds_by_state, heuristic_estimates_cache, dead_teams_check,
                                                          noise)
                    update(stats, game_result)

        Q = sorted(map(lambda x: x[0], beam.heap))
        beam = heapdict.heapdict()

    stats.heuristic_estimates_cache_size = len(heuristic_estimates_cache)
    print("feasibility checks failed %d" % stats.feasibility_checks_failed)
    print("optimality checks failed %d" % stats.optimality_checks_failed)
    print("shortest path length %d" % terminal.shortest_path_length)
    print("heuristic estimates cache size %d" % stats.heuristic_estimates_cache_size)

    if len(terminal.solution) == 0:
        print("no feasible solution found")
    else:
        print(solution_to_rounds_matrix(ttp_instance, terminal.solution))
    return terminal, stats
