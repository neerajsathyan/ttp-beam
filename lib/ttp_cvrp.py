# TTP CVRP(H) bounds precalculation module by recursive state space traversal and subsequent bottom up sweep
import numpy as np

import lib.ttp_instance as TTPInstance
import lib.ttp_util as TTPUtil

import copy
from typing import Generic, TypeVar, Union, Dict
from collections import deque
from queue import Queue

T = TypeVar("T")


class State:
    def __init__(self, teams_left: set, position: int, streak: int):
        self.teams_left = frozenset(teams_left)
        self.position = position
        self.streak = streak

    def __hash__(self):
        return hash((self.teams_left, self.position, self.streak))

    def __eq__(self, other):
        return (self.teams_left, self.position, self.streak) == (other.teams_left, other.position, other.streak)


class Arc(Generic[T]):
    def __init__(self, destination: T, weight: int) -> None:
        self.destination: T = destination
        self.weight: int = weight

    def __hash__(self):
        return hash((self.destination, self.weight))

    def __eq__(self, other):
        return (self.destination, self.weight) == (other.destination, other.weight)


class Node:
    def __init__(self):
        self.layer: int = 0
        self.shortest_path_length: int = 0
        self.lower_bound: int = 0
        self.constrained_lower_bounds: Union[[]:int, None] = None
        self.parent: Union[Node, None] = None
        self.forward_arcs: Union[[]:Arc[Node], None] = []
        self.state: Union[State, None] = None

    def __hash__(self):
        return hash((
                    self.layer, self.shortest_path_length, self.lower_bound, self.constrained_lower_bounds, self.parent,
                    self.forward_arcs, self.state))

    def __eq__(self, other):
        return (self.layer, self.shortest_path_length, self.lower_bound, self.constrained_lower_bounds, self.parent,
                self.forward_arcs, self.state) == (
               other.layer, other.shortest_path_length, other.lower_bound, other.constrained_lower_bounds, other.parent,
               other.forward_arcs, other.state)


# def hash(state: State, h: int):
#     return hash(state.teams_left, hash(state.position, hash(state.streak)))


def isequal(a: State, b: State):
    return (a.teams_left == b.teams_left) and (a.position == b.position) and (a.streak == b.streak)


def move_to_team(ttp_instance: TTPInstance.TTPInstance, node: Node, to_team: int):
    new_node = Node()
    weight = ttp_instance.d[node.state.position - 1][to_team - 1]
    new_node.shortest_path_length = node.shortest_path_length + weight
    teams_left = set(copy.copy(node.state.teams_left))
    teams_left.remove(to_team)
    new_node.state = State(teams_left, to_team, node.state.streak + 1)
    return new_node, weight


def move_to_team_and_home(ttp_instance: TTPInstance.TTPInstance, node: Node, to_team: int, home: int):
    new_node = Node()
    weight = ttp_instance.d[node.state.position - 1][to_team - 1] + ttp_instance.d[to_team - 1][home - 1]
    new_node.shortest_path_length = node.shortest_path_length + weight
    teams_left = set(copy.copy(node.state.teams_left))
    teams_left.remove(to_team)
    new_node.state = State(teams_left, home, 0)
    return new_node, weight


def check_state_exists(state: State, nodes: Dict[State, Node]):
    for state2 in nodes:
        if np.array_equal(state.teams_left,
                          state2.teams_left) and state.streak == state2.streak and state.position == state2.position:
            return True
    return False


def incorporate(parent: Node, new_node: Node, weight: int, nodes, nodes_by_layers, Q):
    new_node.layer = parent.layer + 1
    # if check_state_exists(new_node.state, nodes):
    if new_node.state in nodes.keys():
        existing_node = nodes[new_node.state]
        if new_node.shortest_path_length < existing_node.shortest_path_length:
            existing_node.shortest_path_length = new_node.shortest_path_length
            existing_node.parent = parent
        parent.forward_arcs.append(Arc[Node](existing_node, weight))
    else:
        nodes[new_node.state] = new_node
        nodes_by_layers[new_node.layer].append(new_node)
        parent.forward_arcs.append(Arc[Node](new_node, weight))
        Q.put(new_node)


def construct(team: int, ttp_instance: TTPInstance.TTPInstance, streak_limit: int):
    root = Node()
    root.shortest_path_length = 0
    teams_left = set(np.arange(1, ttp_instance.n + 1))
    teams_left.remove(team)
    root.state = State(teams_left, team, 0)

    terminal = Node()
    terminal.shortest_path_length = np.iinfo(np.int32).max
    terminal.lower_bound = 0
    terminal.constrained_lower_bounds = np.ones(ttp_instance.n, int) * np.iinfo(np.int32).max
    terminal.constrained_lower_bounds[0] = 0
    terminal.state = State(set(), team, 0)
    terminal.layer = ttp_instance.n - 1

    Q = Queue()
    Q.put(root)

    nodes = dict()
    nodes[root.state] = root
    nodes[terminal.state] = terminal

    nodes_by_layers = {i: deque() for i in range(ttp_instance.n)}
    nodes_by_layers[0].append(root)
    nodes_by_layers[ttp_instance.n - 1].append(terminal)

    transitions = 0

    while not Q.empty():
        node = Q.get()
        for to_team in node.state.teams_left:
            if len(node.state.teams_left) > 1 and node.state.streak < streak_limit - 1:
                new_node, weight = move_to_team(ttp_instance, node, to_team)
                transitions += 1
                incorporate(node, new_node, weight, nodes, nodes_by_layers, Q)

            new_node, weight = move_to_team_and_home(ttp_instance, node, to_team, team)
            transitions += 1
            incorporate(node, new_node, weight, nodes, nodes_by_layers, Q)

    print("%d transitions\n" % transitions)
    print("%d nodes\n" % len(nodes))

    return nodes, nodes_by_layers, terminal.shortest_path_length


def calculate_bounds_for_teams(ttp_instance: TTPInstance.TTPInstance, bounds_by_state: np.array(4, int)):
    root_bound_sum = 0
    for team in range(1, ttp_instance.n + 1):
        print("calculating team %d\n" % team)
        nodes, nodes_by_layers, shortest_path = construct(team, ttp_instance, ttp_instance.streak_limit)

        root_bound_sum += shortest_path

        for i in range(ttp_instance.n - 2, -1, -1):
            for node in nodes_by_layers[i]:
                node.lower_bound = min(map(lambda x: x.destination.lower_bound + x.weight, node.forward_arcs))

        for node_state, node in nodes.items():
            bounds_by_state[team - 1][TTPUtil.mask_teams_left(team, node.state.teams_left) - 1][
                node.state.position - 1][
                node.state.streak] = node.lower_bound

    return root_bound_sum


def find_min_for_each_element_array(array1, array2):
    for index, bound in np.ndenumerate(array1):
        # each element in array1 is compared with their respective indexes in array2[*][i]
        min_ele = bound
        for i in range(len(array2)):
            if min_ele > array2[i][index[0]]:
                min_ele = array2[i][index[0]]
        array1[index[0]] = min_ele


def calculate_bounds_for_teams_cvrph(ttp_instance: TTPInstance.TTPInstance, bounds_by_state: np.array(5, int)):
    print("CVRPH")
    root_bound_sum = 0
    for team in range(1, ttp_instance.n + 1):
        print("calculating team %d\n" % team)
        nodes, nodes_by_layers, shortest_path = construct(team, ttp_instance, ttp_instance.streak_limit)

        root_bound_sum += shortest_path

        for i in range(ttp_instance.n - 2, -1, -1):
            for node in nodes_by_layers[i]:
                node.constrained_lower_bounds = np.ones(ttp_instance.n, dtype=int) * np.iinfo(np.int32).max
                if node.state.position == team:
                    # node.constrained_lower_bounds[1:] = min(node.constrained_lower_bounds[1:],
                    #                                         map(lambda x: [sum_with_potential_infinity(
                    #                                             x.destination.constrained_lower_bounds[i], x.weight)
                    #                                             for i in range(0, ttp_instance.n - 1)],
                    #                                             node.forward_arcs))
                    mapped_array = list(map(lambda x: [sum_with_potential_infinity(x.destination.constrained_lower_bounds[i], x.weight) for i in range(ttp_instance.n-1)], node.forward_arcs))
                    find_min_for_each_element_array(node.constrained_lower_bounds[1:], mapped_array)
                else:
                    # node.constrained_lower_bounds = min(node.constrained_lower_bounds,
                    #                                     map(lambda x: [sum_with_potential_infinity(
                    #                                         x.destination.constrained_lower_bounds[i], x.weight)
                    #                                         for i in range(0, ttp_instance.n)],
                    #                                         node.forward_arcs))
                    mapped_array = list(map(
                        lambda x: [sum_with_potential_infinity(x.destination.constrained_lower_bounds[i], x.weight) for
                                   i in range(ttp_instance.n)], node.forward_arcs))
                    find_min_for_each_element_array(node.constrained_lower_bounds, mapped_array)

        for node_state, node in nodes.items():
            bounds_by_state[team - 1][TTPUtil.mask_teams_left(team, node.state.teams_left) - 1][
                node.state.position - 1][
                node.state.streak][:] = node.constrained_lower_bounds
    return root_bound_sum


def sum_with_potential_infinity(a: int, b: int):
    if a is None:
        return a
    elif b is None:
        return b
    elif a is None and b is None:
        return 0
    elif a == np.iinfo(np.int32).max or b == np.iinfo(np.int32).max:
        return np.iinfo(np.int32).max
    else:
        return a + b
