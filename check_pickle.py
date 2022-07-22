import numpy as np
import lib.ttp_util as TTPUtil

b1 = TTPUtil.load_numpy_pickle("data/nl4_cvrp.pkl.bz2")
b2 = TTPUtil.load_numpy_pickle("data/frauhner/nl4_cvrp.pkl.bz2")

print(np.array_equal(b1, b2))
