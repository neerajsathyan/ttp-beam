import subprocess
import sys
import glob
import os
from string import digits

cd = os.getcwd()
os.chdir('data')

beam_widths = [1, 10, 100, 1000, 10000, 15000]
relative_sigmas = [0, 0.1, 0.01, 0.001, 0.0001]
first_k_layers_noisy = ["-1", "half", "full"]

fileList = glob.glob("*.bz2")
os.chdir(cd)
for file in fileList:
    print("Running Instance: " + file)
    sys.stdout = open('ttp_beam_search.txt', 'a')
    inst = file.split('_')
    instance = inst[0].translate({ord(k): None for k in digits})
    team_size = inst[0].replace(instance, '')
    instance_file = 'insts/' + instance.upper() + "/" + instance + team_size + ".txt"
    streak_limit = inst[1].replace('sl', '')
    typ = inst[2].split('.')[0]
    for bw in beam_widths:
        for rs in relative_sigmas:
            for noise in first_k_layers_noisy:
                for exp in range(1):
                    ns = noise
                    if noise == "half":
                        ns = team_size
                    elif noise == "full":
                        ns = str(int(team_size) * (int(team_size) - 1))
                    print("Running Experiment Iteration: %d\tbw: %d\trs: %d\tnoise: %s" % ((exp + 1), bw, rs, ns))
                    subprocess.check_call(
                        ['python', 'ttp_beam_search.py', instance_file, streak_limit, '1', 'data/'+file, str(bw), '1', '1',
                         str(rs), ns], stdout=sys.stdout, stderr=subprocess.STDOUT)

