from lib import heapdict
import numpy as np

hd = heapdict.heapdict()

hd['pen'] = 3

hd['notebook'] = 1

hd['bagpack'] = 4

hd['lunchbox'] = 2

print(hd.get('bagpack'))
print(hd.peekitem())
while hd:
    print(hd.popitem())

print(np.delete(np.arange(1, 5), 2 - 1))
print(np.random.permutation(4)+1)

rounds = [2,1,1,2]
teams_permutation = [2,4,1,3]

ty = list(map(lambda x: (rounds[x - 1], teams_permutation[x - 1]), np.arange(1, 4 + 1)))

print(np.argmin(
        list(map(lambda x: (rounds[x - 1], teams_permutation[x - 1]), np.arange(1, 4 + 1)))))
gh = [(0,2), (0,3), (0,1), (0,4)]
print(ty)
print(min(ty))
print(ty.index(min(ty)))
#
# 1
# ('notebook', 1)
# ('notebook', 1)
# ('lunchbox', 2)
# ('pen', 3)
# ('bagpack', 4)
