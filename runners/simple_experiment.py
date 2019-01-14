import numpy as np
import matplotlib.pyplot as plt

from math import log
from data_featured.dataset_featured import DatasetFeatures
from probability_distribution.probability_distribution import ProbabilityDistribution
from game.game import Game, UtilityFunctions

# DEFINE CONSTANTS

def prepare_action_and_mesh(prb, discretize):
    spaces = np.linspace(0, 1, discretize)
    x = np.meshgrid(*[spaces] * prb.prb.d)
    mesh = np.column_stack((xsub.ravel() for xsub in x))
    actions = range(int(discretize ** prb.prb.d))
    return actions, mesh, x

def solve_simple_game(data, discretize=50, FPrate=0.2, discount=0, plot=True):

    # get data

    # turn data into distribution
    prb = ProbabilityDistribution(data, 1. / discretize)

    # prepare inputs for the game
    util = UtilityFunctions.utilityMul

    actions, mesh, x = prepare_action_and_mesh(prb, discretize)

    g = Game(actions, util, mesh, prb.getPrb, FPrate, discount)
    value = g.solve()
    decisions = np.array(g.thetas).reshape(x[0].shape)

    if plot:
        fig, ax = plt.subplots(2, 3)
        prb.plot(ax[0, 0])

        # Defender strategy
        ax[0, 1].set_title('Defender probabilities')
        ax[0, 1].set_xlabel(data.features[0])
        ax[0, 1].set_ylabel(data.features[1])
        cax = ax[0, 1].imshow(decisions, cmap=plt.get_cmap('Greys'), extent=[0, 1, 0, 1], vmin=0, vmax=1,
            interpolation='nearest')
        ax[0, 1].set_xlim([0, 1])
        ax[0, 1].set_ylim([0, 1])

        #  Defender false positive
        ax[0, 2].set_title('Real false-positives')
        ax[0, 2].set_xlabel(data.features[0])
        ax[0, 2].set_ylabel(data.features[1])
        cax = ax[0, 2].imshow(np.array([prb.getPrb(mesh[a]) * g.thetas[a] for a in actions]).reshape(x[0].shape),
            cmap=plt.get_cmap('Greys'), extent=[0, 1, 0, 1])
        ax[0, 2].set_xlim([0, 1])
        ax[0, 2].set_ylim([0, 1])

        # Attacker utilities
        ax[1, 1].set_title('Attacker detected utilities')
        ax[1, 1].set_xlabel(data.features[0])
        ax[1, 1].set_ylabel(data.features[1])

        attack_utils = np.array([util(mesh[a]) * (1 - p) for a, p in zip(actions, g.thetas)]).reshape(x[0].shape)
        cax = ax[1, 1].imshow(attack_utils, cmap=plt.cm.gist_earth_r, extent=[0, 1, 0, 1], vmin=0, vmax=1,
            interpolation='nearest')
        ax[1, 1].set_xlim([0, 1])
        ax[1, 1].set_ylim([0, 1])

        # original utilities
        ax[1, 0].set_title('Attacker undetected utilities')
        ax[1, 0].set_xlabel(data.features[0])
        ax[1, 0].set_ylabel(data.features[1])
        cax = ax[1, 0].imshow(np.array([util(mesh[a]) for a in actions]).reshape(x[0].shape), cmap=plt.cm.gist_earth_r,
            extent=[0, 1, 0, 1], vmin=0, vmax=1, interpolation='nearest')
        ax[1, 0].set_xlim([0, 1])
        ax[1, 0].set_ylim([0, 1])

        plt.show()
    return value

def simple_runner():
    discretize = 20
    FPrate = 0.1
    discount = 0
    data = DatasetFeatures.from_normal_distribution_independent()
    solve_simple_game(data, discretize, FPrate, discount, plot=True)

def plot_FP_vs_utility():
    discretize = 20
    FPrate = 0.1
    discount = 0
    data = DatasetFeatures.from_normal_distribution_independent()

    vals = dict()
    for FPrate in np.linspace(0, 1, 50):
        val = solve_simple_game(data, discretize, FPrate, discount, plot=False)
        vals.update({FPrate:val})

    plt.plot(vals.keys(), vals.values())
    plt.title('FP vs utility')
    plt.xlabel('false-positive-rate')
    plt.ylabel('attacker utility')
    plt.show()

if __name__ == "__main__":
    import sys
