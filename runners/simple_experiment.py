import numpy as np

from data_featured.dataset_featured import DatasetFeatures
from probability_distribution.probability_distribution import ProbabilityDistribution
from game.game import Game

data = DatasetFeatures.from_normal_distribution_independent()

# print(data.data)

prb = ProbabilityDistribution(data)

# distcretization =


# prb.plot()

dim = 2

x0s = np.linspace(0,1,5)
x1s = np.linspace(0,1,5)
x0, x1 = np.meshgrid(x0s, x1s)

# get actions
mesh = np.c_[x0.ravel(), x1.ravel()]
print(mesh)
actions = range(len(mesh))

# get utilities
print('generating utilities')
utils = [mesh[a][0] for a in actions]
print('utils')
print(type(utils))
print(utils)
# get distributions
# for a in actions:
#     print(prb.prb(a)[0])
dist = [prb.prb(mesh[a])[0] for a in actions]
print('actions')
print(type(actions))
print(actions)
print('dist')
print(type(dist))
print(dist)


g = Game(actions, utils, dist, 0.1, 0)
g.solve()
decisions = np.array(g.thetas).reshape(x0.shape)
print(decisions)

# X = np.mgrid[0:1:5j]
# pos = np.vstack([X.ravel(), X.ravel()])
# pos = np.meshgrid(X, X)
# print(X)
# print(pos)


