import numpy as np
from scipy import stats
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from math import log
from data_featured.dataset_featured import DatasetFeatures
from probability_distribution.probability_distribution import ProbabilityDistribution
from game.game import Game, UtilityFunctions

# DEFINE CONSTANTS

def prepare_action_and_mesh(prb, discretize):
    spaces = [np.linspace(prb.data_featured.mins[d], prb.data_featured.maxs[d], discretize) for d in range(prb.prb.d)]
    x = np.meshgrid(*spaces)
    # x = np.mgrid(*spaces)

    mesh = np.stack([xsub.ravel() for xsub in x], axis=-1)
    print(mesh)
    actions = range(int(discretize ** prb.prb.d))
    return actions, mesh, x

def solve_simple_game(data:DatasetFeatures, discretize=50, FPrate=0.1, discount=0, plot=True):

    # turn data into distribution
    prb = ProbabilityDistribution(data, discretize)

    # prepare inputs for the game
    # util = UtilityFunctions.utilityUniform
    util = UtilityFunctions.utility1
    # util = UtilityFunctions.utilityMul

    actions, mesh, x = prepare_action_and_mesh(prb, discretize)

    g = Game(actions, util, mesh, prb.getPrb, FPrate, discount)
    value = g.solve()
    decisions = np.array(g.thetas).reshape(x[0].shape)

    for a in actions:
        print(mesh[a])

    if plot:
        fig, ax = plt.subplots(2, 3)
        prb.plot(ax[0, 0])
        plot_defender_strategy(ax[0,1], data, decisions)
        plot_defender_fp(actions, ax[0,2], data, g, mesh, prb, x)
        plot_defender_dist(actions, ax[1,2], data, mesh, prb, x)
        plot_attacker_utils(actions, ax[1,1], data, g, mesh, util, x)
        plot_attacker_utils_orig(actions, ax[1,0], data, mesh, util, x)

        plt.show()
    return value


def plot_attacker_utils_orig(actions, ax, data, mesh, util, x):
    if len(data.features) == 1:
        ax.set_title('Attacker utilities undet')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel('utility')
        ax.plot(actions, np.array([util(mesh[a]) for a in actions]).reshape(x[0].shape))

    if len(data.features) == 2:
        ax.set_title('Attacker utilities undet')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel(data.features[1])
        values = np.array([util(mesh[a]) for a in actions]).reshape(x[0].shape)
        vmin = values.min()
        vmax = values.max()
        ax.imshow(values, cmap=plt.cm.gist_earth_r,
            extent=np.ndarray.flatten(data.limits), vmin=vmin, vmax=vmax, interpolation='nearest',
            origin='lower')
        ax.set_xlim(data.limits[0])
        ax.set_ylim(data.limits[1])


def plot_attacker_utils(actions, ax, data, g, mesh, util, x):
    if len(data.features) == 1:
        ax.set_title('Attacker utilities')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel('uitlity')
        attack_utils = np.array([util(mesh[a]) * (1 - p) for a, p in zip(actions, g.thetas)]).reshape(x[0].shape)
        cax = ax.plot(actions, attack_utils)

    if len(data.features) == 2:
        ax.set_title('Attacker utilities')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel(data.features[1])
        values = np.array([util(mesh[a]) for a in actions]).reshape(x[0].shape)
        vmin = values.min()
        vmax = values.max()
        attack_utils = np.array([util(mesh[a]) * (1 - p) for a, p in zip(actions, g.thetas)]).reshape(x[0].shape)
        cax = ax.imshow(attack_utils, cmap=plt.cm.gist_earth_r, extent=np.ndarray.flatten(data.limits),
            vmin=vmin, vmax=vmax, interpolation='nearest', origin='lower')
        ax.set_xlim(data.limits[0])
        ax.set_ylim(data.limits[1])


def plot_defender_fp(actions, ax, data, g, mesh, prb, x):
    if len(data.features) == 1:
        ax.set_title('Real false-positives')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel('probability')
        ax.plot([prb.getPrb(mesh[a]) * g.thetas[a] for a in actions])

    if len(data.features) == 2:
        ax.set_title('Real false-positives')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel(data.features[1])
        values = np.array([prb.getPrb(mesh[a]) * g.thetas[a] for a in actions]).reshape(x[0].shape)
        ax.imshow(values,
            cmap=plt.get_cmap('Greys'), extent=np.ndarray.flatten(data.limits), origin='lower')
        ax.set_xlim(data.limits[0])
        ax.set_ylim(data.limits[1])

def plot_defender_dist(actions, ax, data, mesh, prb, x):
    if len(data.features) == 1:
        ax.set_title('data distribution false-positives')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel('probability')
        ax.plot([prb.getPrb(mesh[a]) for a in actions])

    if len(data.features) == 2:
        ax.set_title('Real false-positives')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel(data.features[1])
        values = np.array([prb.getPrb(mesh[a]) for a in actions]).reshape(x[0].shape)
        ax.imshow(values,
            cmap=plt.get_cmap('Greys'), extent=np.ndarray.flatten(data.limits), origin='lower')
        ax.set_xlim(data.limits[0])
        ax.set_ylim(data.limits[1])

def plot_defender_strategy(ax, data, decisions):
    if len(data.features) ==1:
        ax.set_title('Defender probabilities')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel('probability')
        cax = ax.plot(decisions)

    if len(data.features) == 2:
        ax.set_title('Defender probabilities')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel(data.features[1])
        values = decisions
        ax.imshow(decisions, cmap=plt.get_cmap('Greys'), extent=np.ndarray.flatten(data.limits),
            vmin=0, vmax=1, interpolation='nearest', origin='lower')
        ax.set_xlim(data.limits[0])
        ax.set_ylim(data.limits[1])


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



def aggregate(df):
    n = 50
    list_df = [df[i:i+n] for i in range(0, df.shape[0]-n-1)]
    # print(list_df[0])

    means = []
    stds = []

    for line in list_df:
        means.append(line.agg('mean')['value'])
        stds.append(line.agg('std')['value'])
    return pd.DataFrame({'mean': means, 'std':stds})

####################
# Main
####################
if __name__ == '__main__':
    print('Adversarial Anomaly Detector.')
    print()

    # Parse the parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Amount of verbosity. This shows more info about the results.', action='count', required=False, default=1)
    parser.add_argument('-e', '--debug', help='Amount of debugging. This shows inner information about the flows.', action='store', required=False, type=int, default=0)
    parser.add_argument('-p', '--plot', help='Plot the data in an active window.', action='store_true', required=False)
    parser.add_argument('-a', '--alg', help='Algorithm [Simple]', action='store_true', required=False)
    parser.add_argument('-d', '--data', help='Where to get the data [Generate|File]', action='store', required=False)
    parser.add_argument('--dist', help='Where to get the data [Generate|File]', action='store', required=False)
    args = parser.parse_args()

    if args.alg is False:
        args.alg = 'simple'
        # args.data = 'generate'
        args.data = 'file'
        args.datafile = '/home/kori/data/projects/NAB/data/artificialWithAnomaly/art_daily_flatmiddle.csv'



    if args.data == 'generate':
        data = DatasetFeatures.from_normal_distribution_independent(features=['F1', 'F2'], mean=[0.5, 0.5], var=[0.1, 0.2], size=1000)
        # data = DatasetFeatures.from_normal_distribution_independent(features=['F1'], mean=[0.2], var=[0.1], size=1000)
        print(data.data)
    if args.data == 'file':
        df = pd.read_csv(args.datafile)
        df_agg = aggregate(df)
        # print(df_agg)
        # print(df_agg.columns)
        data = DatasetFeatures(df_agg, df_agg.columns)


    if args.alg == 'simple':
        solve_simple_game(data, FPrate=0.01, plot=True)
