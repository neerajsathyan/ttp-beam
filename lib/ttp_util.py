# TTP Utility Functions
import bz2
import pickle

import numpy


def save_numpy_pickle(numpy_pickle_file, bounds_by_state):
    sfile = bz2.BZ2File(numpy_pickle_file, "w")
    pickle.dump(numpy.array(bounds_by_state), sfile, protocol=4)
    sfile.close()


def load_numpy_pickle(numpy_pickle_file):
    sfile = bz2.BZ2File(numpy_pickle_file, "r")
    bounds_by_state = pickle.load(sfile)
    sfile.close()
    return bounds_by_state


def mask_teams_left(team, teams_left):
    set_mask = 0
    for away_team in teams_left:
        if away_team < team:
            set_mask += 2 ** (away_team - 1)
        else:
            set_mask += 2 ** (away_team - 2)
    return set_mask + 1
