#!/usr/bin/python
import sys

import numpy as np

import lib.ttp_cvrp as TTPCVRP
import lib.ttp_util as TTPUtil
import lib.ttp_instance as TTPInstance


def main():
    if len(sys.argv) != 5:
        print("Usage: $PROGRAM_FILE <instance-file> <streak-limit> <numpy-pickle-file> <cvrp-h-bounds>")
        sys.exit(1)

    instance_file = sys.argv[1]
    streak_limit = int(sys.argv[2])
    numpy_pickle_file = sys.argv[3]
    ttp_instance = TTPInstance.TTPInstance(instance_file, streak_limit, True)
    cvrp_h_bounds = bool(int(sys.argv[4]))

    if cvrp_h_bounds:
        bounds_by_state = np.zeros(
            (ttp_instance.n, 2 ** (ttp_instance.n - 1), ttp_instance.n, streak_limit, ttp_instance.n), dtype=int)
        root_bound_sum = TTPCVRP.calculate_bounds_for_teams_cvrph(ttp_instance, bounds_by_state)
    else:
        bounds_by_state = np.zeros((ttp_instance.n, 2 ** (ttp_instance.n - 1), ttp_instance.n, streak_limit), dtype=int)
        root_bound_sum = TTPCVRP.calculate_bounds_for_teams(ttp_instance, bounds_by_state)

    print("bounds sum %d\n" % np.sum(bounds_by_state))
    print("root bound %d\n" % root_bound_sum)

    TTPUtil.save_numpy_pickle(numpy_pickle_file, bounds_by_state)


if __name__ == "__main__":
    main()
