#!/usr/bin/python
import sys
import csv
from hashlib import md5
import uuid

import numpy as np
import time
# import resource

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
    bounds = sys.argv[4]

    if bounds == "cvrph":
        bounds_by_state = np.zeros(
            (ttp_instance.n, 2 ** (ttp_instance.n - 1), ttp_instance.n, streak_limit, ttp_instance.n), dtype=int)
        root_bound_sum = TTPCVRP.calculate_bounds_for_teams_cvrph(ttp_instance, bounds_by_state)
    elif bounds == "cvrp":
        bounds_by_state = np.zeros((ttp_instance.n, 2 ** (ttp_instance.n - 1), ttp_instance.n, streak_limit), dtype=int)
        root_bound_sum = TTPCVRP.calculate_bounds_for_teams(ttp_instance, bounds_by_state)
    else:
        bounds_by_state = np.zeros((ttp_instance.n, 2 ** (ttp_instance.n - 1), ttp_instance.n), dtype=int)
        root_bound_sum = TTPCVRP.calculate_bounds_for_teams_tsp(ttp_instance, bounds_by_state)

    print("bounds sum %d\n" % np.sum(bounds_by_state))
    print("root bound %d\n" % root_bound_sum)

    TTPUtil.save_numpy_pickle(numpy_pickle_file, bounds_by_state)
    return np.sum(bounds_by_state), root_bound_sum


if __name__ == "__main__":
    time_start = time.perf_counter()
    bounds_by_state_sum, root_bound_sum = main()
    time_elapsed = (time.perf_counter() - time_start)
    # memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    print("%5.8f secs" % time_elapsed)

    # Collect all data to a csv file
    f = open('data/lower_bounds/results.csv', 'a')
    writer = csv.writer(f)
    typ = sys.argv[4].upper()
    writer.writerow([md5(sys.argv[1].encode()).hexdigest()+"_"+str(uuid.uuid4()), sys.argv[1], int(sys.argv[2]), sys.argv[3], typ, bounds_by_state_sum, root_bound_sum, time_elapsed])
    f.close()

