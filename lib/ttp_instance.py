# TTP instance module
import numpy as np


class TTPInstance:
    def __init__(self, file_name: str, streak_limit: int, no_repeat: bool):
        self.d = np.loadtxt(file_name, int)
        self.n = np.size(self.d, 1)
        self.streak_limit = streak_limit
        self.no_repeat = no_repeat
