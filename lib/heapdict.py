from typing import Dict, Tuple

import numpy as np

from lib.ttp_states import State, Node

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping


def doc(s):
    if hasattr(s, '__call__'):
        s = s.__doc__

    def f(g):
        g.__doc__ = s
        return g

    return f


# def check_state_exists(state: State, beam_dict: Dict[State, Tuple[Node, int, int]]):
#     for state2 in beam_dict:
#         if np.array_equal(state.games_left, state2.games_left) and np.array_equal(state.forbidden_opponents,
#                                                                                   state2.forbidden_opponents) \
#                 and np.array_equal(state.rounds, state2.rounds) and np.array_equal(state.positions,
#                                                                                    state2.positions) \
#                 and np.array_equal(state.possible_away_streaks, state2.possible_away_streaks) \
#                 and np.array_equal(state.possible_home_stands, state2.possible_home_stands):
#             return True
#     return False


class heapdict(MutableMapping):
    __marker = object()

    def __init__(self, *args, **kw):
        self.heap = []
        self.d = {}
        self.update(*args, **kw)

    @doc(dict.clear)
    def clear(self):
        del self.heap[:]
        self.d.clear()

    @doc(dict.__setitem__)
    def __setitem__(self, key, value):
        if key in self.d:
            # if check_state_exists(key, self.d):
            self.pop(key)
        wrapper = [value, key, len(self)]
        self.d[key] = wrapper
        self.heap.append(wrapper)
        self._increase_key(len(self.heap) - 1)

    def _max_heapify(self, i):
        n = len(self.heap)
        h = self.heap
        while True:
            # calculate the offset of the left child
            l = (i << 1) + 1
            # calculate the offset of the right child
            r = (i + 1) << 1
            if l < n and h[i][0] < h[l][0]:
                max = l
            else:
                max = i
            if r < n and h[max][0] < h[r][0]:
                max = r

            if max == i:
                break

            self._swap(i, max)
            i = max

    def _increase_key(self, i):
        while i:
            # calculate the offset of the parent
            parent = (i - 1) >> 1
            if self.heap[parent][0] > self.heap[i][0]:
                break
            self._swap(i, parent)
            i = parent

    def _swap(self, i, j):
        h = self.heap
        h[i], h[j] = h[j], h[i]
        h[i][2] = i
        h[j][2] = j

    @doc(dict.__delitem__)
    def __delitem__(self, key):
        wrapper = self.d[key]
        while wrapper[2]:
            # calculate the offset of the parent
            parentpos = (wrapper[2] - 1) >> 1
            parent = self.heap[parentpos]
            self._swap(wrapper[2], parent[2])
        self.popitem()

    @doc(dict.__getitem__)
    def __getitem__(self, key):
        return self.d[key][0]

    @doc(dict.__iter__)
    def __iter__(self):
        return iter(self.d)

    def popitem(self):
        """D.popitem() -> (k, v), remove and return the (key, value) pair with lowest\nvalue; but raise KeyError if D is empty."""
        wrapper = self.heap[0]
        if len(self.heap) == 1:
            self.heap.pop()
        else:
            self.heap[0] = self.heap.pop()
            self.heap[0][2] = 0
            self._max_heapify(0)
        del self.d[wrapper[1]]
        return wrapper[1], wrapper[0]

    @doc(dict.__len__)
    def __len__(self):
        return len(self.d)

    def peekitem(self):
        """D.peekitem() -> (k, v), return the (key, value) pair with lowest value;\n but raise KeyError if D is empty."""
        return (self.heap[0][1], self.heap[0][0])


del doc
__all__ = ['heapdict']
