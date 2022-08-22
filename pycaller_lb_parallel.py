# insts/NL/nl4.txt 3 data/nl4_sl3_cvrp.pkl.bz2 cvrp
import subprocess
import sys
import glob
import os
from string import digits
import itertools
import multiprocessing


# A function which will process a tuple of parameters
def func(params):
    file = params[0]
    sys.stdout = open('ttp_lb.txt', 'w')
    print("Running Instance: " + file)
    inst = file.split('.')
    instance = inst[0].translate({ord(k): None for k in digits})
    team_size = inst[0].replace(instance, '')
    streak_limit = params[1]
    h_type = params[2]
    for exp in range(1):
        subprocess.check_call(
            ['python', 'ttp_lower_bound_precalc.py', "insts/" + instance.upper() + "/" + file, str(streak_limit),
             "data/" + instance + team_size + "_sl" + str(streak_limit) + "_" + h_type + ".pkl.bz2", h_type],
            stdout=sys.stdout, stderr=subprocess.STDOUT)


if __name__ == '__main__':
    cd = os.getcwd()
    os.chdir('insts/CIRC')

    sl = [1, 2, 3]
    typ = ["cvrp", "cvrph"]
    fileList = glob.glob("*.txt")

    os.chdir(cd)

    paramlist = list(itertools.product(fileList, sl, typ))
    pool = multiprocessing.Pool(6)

    res = pool.map(func, paramlist)
