#!/usr/bin/python
# ttp_beam_search.jl
# performs beam search for a given TTP instance with either exact CVRP or CVRPH bounds guidance
# bounds have to precalculated via ttp_bounds_precalculation.jl saved as pickled numpy array
import sys

from lib.ttp_instance import TTPInstance
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

    print("Solving TTP instance %s with beam search, beam width %d, streak limit %d, no repeaters set to %s\n" % (instance_file, beam_width, streak_limit, no_repeat))
    #construction_time = @elapsed terminal, stats = TTPSolver.construct(ttp_instance, bounds_by_state, beam_width, dead_teams_check, randomized_team_order, relative_sigma, nothing, first_k_layers_noisy)

    #@printf("[CSV] %s;%d;%d;%d;%.02f;%d;%d;%f;%d\n", TTPUtil.basename(instance_file, ".txt"), ttp_instance.n, beam_width, terminal.shortest_path_length, construction_time, dead_teams_check, randomized_team_order, relative_sigma, first_k_layers_noisy)


if __name__ == "__main__":
    main()

