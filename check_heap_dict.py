from lib import heapdict

hd = heapdict.heapdict()

hd['pen'] = 3

hd['notebook'] = 1

hd['bagpack'] = 4

hd['lunchbox'] = 2

print(hd.get('bagpack'))
print(hd.peekitem())
while hd:
    print(hd.popitem())


#
# 1
# ('notebook', 1)
# ('notebook', 1)
# ('lunchbox', 2)
# ('pen', 3)
# ('bagpack', 4)
