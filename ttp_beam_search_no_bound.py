#!/usr/bin/python
# ttp_beam_search.jl
# performs beam search for a given TTP instance with either exact CVRP or CVRPH bounds guidance
# bounds have to precalculated via ttp_bounds_precalculation.jl saved as pickled numpy array
import csv
import sys
import time
import uuid
from hashlib import md5

from lib.ttp_instance import TTPInstance
from lib.ttp_solver import construct
from lib.ttp_util import load_numpy_pickle


def main():
    if len(sys.argv) != 10:
        print("Usage: $PROGRAM_FILE <instance-file> <streak-limit> <no-repeat> <cvrp-bounds-file> <beam-width> "
              "<dead_teams_check> <randomized-team-order> <relative-sigma> <first-k-layers-noisy>")
        sys.exit(1)

    instance_file = sys.argv[1]
    streak_limit = int(sys.argv[2])
    no_repeat = bool(int(sys.argv[3]))
    cvrp_bounds_file = sys.argv[4]
    beam_width = int(sys.argv[5])
    dead_teams_check = bool(int(sys.argv[6]))
    randomized_team_order = bool(int(sys.argv[7]))
    relative_sigma = float(sys.argv[8])
    first_k_layers_noisy = int(sys.argv[9])
    ttp_instance = TTPInstance(instance_file, streak_limit, no_repeat)

    print("Loading bounds file %s\n" % cvrp_bounds_file)
    bounds_by_state = load_numpy_pickle(cvrp_bounds_file)

    # This is a zero heuristic bound beam search, so bounds_by_state all will be 0..
    bounds_by_state = 0*bounds_by_state

    print("Solving TTP instance %s with beam search, beam width %d, streak limit %d, no repeaters set to %s\n" % (
    instance_file, beam_width, streak_limit, no_repeat))
    # construction_time = @elapsed terminal, stats = TTPSolver.construct(ttp_instance, bounds_by_state, beam_width,
    # dead_teams_check, randomized_team_order, relative_sigma, nothing, first_k_layers_noisy)
    terminal, stats = construct(ttp_instance, bounds_by_state, beam_width, dead_teams_check, randomized_team_order,
                                relative_sigma, None, cvrp_bounds_file, first_k_layers_noisy)

    # @printf("[CSV] %s;%d;%d;%d;%.02f;%d;%d;%f;%d\n", TTPUtil.basename(instance_file, ".txt"), ttp_instance.n, beam_width, terminal.shortest_path_length, construction_time, dead_teams_check, randomized_team_order, relative_sigma, first_k_layers_noisy)
    return terminal, ttp_instance


if __name__ == "__main__":
    time_start = time.perf_counter()
    terminal, ttp_instance = main()
    time_elapsed = (time.perf_counter() - time_start)
    print("%5.8f secs" % time_elapsed)

    # Collect all data to a csv file
    # insts/NL/nl4.txt 3 1 data/nl4_cvrph.pkl.bz2 10000 1 1 0.001 -1
    f = open('data/beam_search/bs_results_no_bounds.csv', 'a')
    writer = csv.writer(f)
    typ = '0'
    # if 'cvrph' in sys.argv[4]:
    #     typ = 'CVRPH'
    # elif 'cvrp' in sys.argv[4]:
    #     typ = 'CVRP'
    writer.writerow([md5(sys.argv[1].encode()).hexdigest() + "_" + str(uuid.uuid4()), typ, sys.argv[1], ttp_instance.n, ttp_instance.streak_limit, ttp_instance.no_repeat, sys.argv[4], int(sys.argv[5]), bool(int(sys.argv[6])), bool(int(sys.argv[7])), float(sys.argv[8]), int(sys.argv[9]), terminal.shortest_path_length, time_elapsed])
    f.close()
