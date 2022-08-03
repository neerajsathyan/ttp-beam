import subprocess
import sys

with open('nl_lb_args.txt') as f:
    for line in f:
        args = line.split()
        sys.stdout = open('ttp_lower_bound_precalc.txt', 'a')
        instance_file = args[0]
        streak_limit = args[1]
        pickle_file = args[2]
        cvrp_bound = args[3]
        for exp in range(10):
            subprocess.check_call(['python', 'ttp_lower_bound_precalc.py', instance_file, streak_limit, pickle_file, cvrp_bound], stdout=sys.stdout, stderr=subprocess.STDOUT)


# for scriptInstance in [1, 2, 3]:
#     sys.stdout = open('result%s.txt' % scriptInstance, 'w')
#     subprocess.check_call(['python', 'slave.py'], stdout=sys.stdout, stderr=subprocess.STDOUT)
