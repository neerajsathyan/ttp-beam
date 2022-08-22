import sys
import os
import random

import numpy as np


def distance(cord1, cord2, p):
    return round(((abs(cord2[0] - cord1[0]) ** p) + (abs(cord2[1] - cord1[1]) ** p)) ** (1/p))


if __name__ == '__main__':
    team_size = int(sys.argv[1])
    sample_size = int(sys.argv[2])
    sampled_teams = []
    os.chdir('random_insts')
    for sample in range(sample_size):
        # Randomly generate team points in a 1000x1000 grid
        teams = []
        for team in range(1, team_size + 1):
            # for this team add a random unique (x,y)
            x_cord = random.randrange(0, 1000)
            y_cord = random.randrange(0, 1000)
            teams.append((x_cord, y_cord))
        # Now create a distance matrix and save those instances as files
        matrix = np.zeros((team_size, team_size), dtype=int)
        matrix2 = np.zeros((team_size, team_size), dtype=int)
        for i, team1 in enumerate(teams):
            for j, team2 in enumerate(teams):
                if i != j:
                    matrix[i][j] = distance(team1, team2, 1)
                    matrix2[i][j] = distance(team1, team2, 2)
        np.savetxt('rand'+str(team_size)+'_manhattan_'+str(sample+1)+'.txt', matrix, delimiter=' ', newline='\n', fmt="%d")
        np.savetxt('rand' + str(team_size) + '_euclidean_' + str(sample + 1) + '.txt', matrix2, delimiter=' ',
                   newline='\n', fmt="%d")
