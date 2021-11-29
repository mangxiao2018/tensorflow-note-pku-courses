import numpy as np
# 加上种子seed=1后，每次随机出来的数据是一样的，如果不加，每次都不一样
rdm = np.random.RandomState(seed=1)
a = rdm.rand()
b = rdm.rand(2, 3)
print("a:", a)
print("b:", b)
