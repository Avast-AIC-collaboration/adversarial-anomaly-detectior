import numpy as np
import matplotlib.pyplot as plt

from math import log
from data_featured.dataset_featured import DatasetFeatures
from probability_distribution.probability_distribution import ProbabilityDistribution
from game.game import Game


data = DatasetFeatures.from_normal_distribution_independent()

# print(data.data)

prb = ProbabilityDistribution(data)
# prb.prb.


fig, ax = plt.subplots(2,2)

# distcretization =


prb.plot(ax[0,0])

dim = 2
discretize=4
x0s = np.linspace(0,1,discretize)
x1s = np.linspace(0,1,discretize)
x0, x1 = np.meshgrid(x0s, x1s)

# get actions
mesh = np.c_[x0.ravel(), x1.ravel()]
# print(mesh)
actions = range(len(mesh))

# get utilities
# print('generating utilities')

# utils = [ 10 for a in actions]

# utils = [mesh[a][0] for a in actions]
utils = [0.1 for a in actions]


# print('utils')
# print(type(utils))
# print(utils)
# get distributions
# for a in actions:
#     print(prb.prb(a)[0])
dist = [prb.prb(mesh[a])[0] for a in actions]
# print('actions')
# print(type(actions))
# print(actions)
# print('dist')
# print(type(dist))
# print(dist)


g = Game(actions, utils, dist, 0.1, 0)
g.solve()
decisions = np.array(g.thetas).reshape(x0.shape)
# print(decisions)

# cax = ax[1].imshow(decisions, extent=[0, 1, 0, 1])
cax = ax[0,1].imshow(decisions, cmap=plt.get_cmap('Greys'), extent=[0, 1, 0, 1], vmin=0, vmax=1, interpolation='nearest')
ax[0,1].set_xlim([0, 1])
ax[0,1].set_ylim([0, 1])

# attacker utilities
# print(type(utils[0]))
# print(type(g.thetas[0]))
attack_utils = np.array([a*(1-b) for a,b in zip(utils, g.thetas)]).reshape(x0.shape)
# print(attack_utils)

cax = ax[1,1].imshow(attack_utils, cmap=plt.cm.gist_earth_r, extent=[0, 1, 0, 1], vmin=0, vmax=1, interpolation='nearest')
ax[1,1].set_xlim([0, 1])
ax[1,1].set_ylim([0, 1])


# original utilities
cax = ax[1,0].imshow(np.array(utils).reshape(x0.shape), cmap=plt.cm.gist_earth_r, extent=[0, 1, 0, 1], vmin=0, vmax=1, interpolation='nearest')
ax[1,0].set_xlim([0, 1])
ax[1,0].set_ylim([0, 1])

plt.show()

# X = np.mgrid[0:1:5j]
# pos = np.vstack([X.ravel(), X.ravel()])
# pos = np.meshgrid(X, X)
# print(X)
# print(pos)


