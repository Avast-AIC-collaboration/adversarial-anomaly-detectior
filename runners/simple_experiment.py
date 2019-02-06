import numpy as np
from scipy import stats
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from math import log
from data_featured.dataset_featured import DatasetFeatures
from probability_distribution.probability_distribution import ProbabilityDistribution
from game.game import Game, UtilityFunctions
from nn.neuralNetwork import NN


# DEFINE CONSTANTS

def prepare_defender(prb, discretize):
    spaces = [np.linspace(prb.data_featured.mins[d], prb.data_featured.maxs[d], discretize) for d in range(prb.prb.d)]
    # print(spaces)
    x = np.meshgrid(*spaces)
    mesh = np.stack([xsub.ravel() for xsub in x], axis=-1)
    actions = range(int(discretize ** prb.prb.d))
    return actions, mesh, x

def prepare_attacker(prb, discretize, neg=False):
    step = (prb.data_featured.maxs - prb.data_featured.mins) / (discretize -1)
    if neg:
        spaces = [np.array(range(discretize)) * step[d] - step[d]*discretize/2 for d in range(prb.prb.d)]
    else:
        spaces = [np.array(range(discretize)) * step[d] for d in range(prb.prb.d)]
    # spaces = [np.array(range(discretize)) * step[d] for d in range(prb.prb.d)]
    # print(spaces)
    # spaces = [np.linspace(0, 1, discretize) for d in range(prb.prb.d)]
    x = np.meshgrid(*spaces)
    mesh = np.stack([xsub.ravel() for xsub in x], axis=-1)
    return mesh

def solve_simple_game(data:DatasetFeatures, discretize=10, FPrate=0.1, discount=0.9, att_type='replace', plot=True):

    # turn data into distribution
    prb = ProbabilityDistribution(data, discretize)

    # prepare inputs for the game
    # util = UtilityFunctions.utilityUniform
    # util = UtilityFunctions.utility1
    util = UtilityFunctions.utilityMul
    # util = UtilityFunctions.utilitySum

    actions, mesh, x = prepare_defender(prb, discretize)
    att_act = prepare_attacker(prb, discretize, neg=False)


    print('Solving game ....')
    g = Game(actions, util, mesh, prb.getPrb, FPrate, discount, att_act, att_type)
    value = g.solve()
    decisions = np.array(g.thetas).reshape(x[0].shape)

    if plot:
        print('Plotting data')
        fig, ax = plt.subplots(2, 3)
        prb.plot(ax[0,0])
        plot_defender_strategy(ax[0,1], data, decisions, discretize)
        plot_defender_fp(actions, ax[0,2], data, g, mesh, prb, x, discretize)
        plot_attacker_utils(actions, att_act, ax[1,2], data, g, mesh, util, x, att_type=att_type, dist=prb.getPrb)
        plot_attacker_utils_orig(actions, att_act, ax[1,0], data, mesh, util, x)
        plot_attacker_det_prb(actions, att_act, ax[1,1], data, g, mesh, util, x, att_type=att_type, dist=prb.getPrb)
        plt.tight_layout()

        plt.show()
    return value

def solve_simple_game_with_nn(data:DatasetFeatures, FPrate=0.1, discount=0.9, att_type='replace', plot=True):

    # prepare inputs for the game
    # util = UtilityFunctions.utilityUniform
    # util = UtilityFunctions.utility1
    util = UtilityFunctions.utilityMul
    # util = UtilityFunctions.utilitySum


    print('Solving game ....')
    neural = NN(data, util, FPrate, discount, att_type)
    neural.solve()
    return 0

def plot_attacker_utils_orig(actions, att_mesh, ax, data, mesh, util, x):
    print('Plotting attacker orig utility')
    if len(data.features) == 1:
        ax.set_title('Attacker utilities undet')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel('utility')
        ax.plot(att_mesh.flatten(), np.array([util(att_mesh[a]) for a in actions]).reshape(x[0].shape))

    if len(data.features) == 2:
        ax.set_title('Attacker utilities undet')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel(data.features[1])
        values = np.array([util(att_mesh[a]) for a in actions]).reshape(x[0].shape)
        vmin = values.min()
        vmax = values.max()

        f1max = max(att_mesh.T[0])
        f2max = max(att_mesh.T[1])
        ex = [0, f1max, 0, f2max]

        limits1 = [att_mesh.T[0].min(), att_mesh.T[0].max()]
        limits2 = [att_mesh.T[1].min(), att_mesh.T[1].max()]
        ex = limits1 + limits2

        ax.imshow(values, cmap=plt.cm.gist_earth_r,
            extent=ex, vmin=vmin, vmax=vmax, interpolation='nearest',
            origin='lower', aspect='auto')
        ax.set_xlim(limits1)
        ax.set_ylim(limits2)


def plot_attacker_utils(actions, att_mesh, ax, data, g, mesh, util, x, att_type, dist):
    print('Plotting attacker real utility.')
    if att_type == 'replace':
        values_orig = np.array([util(att_mesh[a]) for a in actions]).reshape(x[0].shape)
        vmin = values_orig.min()
        vmax = values_orig.max()
        values = [util(att_mesh[att_a]) * (1-g.thetas[att_a])  for att_a in actions]
        # values = [util(att_mesh[a]) * (1 - p) for a, p in zip(actions, g.thetas)]
    elif att_type == 'add':
      # values = [util(mesh[a]) * ( 1 - sum([dist(mesh[a]+mesh[a_]) * p for a_, p in zip(actions, g.thetas)]))  for a in actions]
        values_orig = np.array([util(att_mesh[a]) for a in actions]).reshape(x[0].shape)
        vmin = values_orig.min()
        vmax = values_orig.max()
        values = [util(att_mesh[att_a]) * (sum([dist(mesh[a] - att_mesh[att_a])
                        for a in actions])
                   - sum([dist(mesh[a] - att_mesh[att_a]) * (g.thetas[a]) for a in actions]))  for att_a in actions]
    attack_utils = np.array(values).reshape(x[0].shape)


    if len(data.features) == 1:
        ax.set_title('Attacker utilities')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel('utility')
        attack_utils = np.array(values).reshape(x[0].shape)
        ax.plot(att_mesh.flatten(), attack_utils)

    if len(data.features) == 2:
        ax.set_title('Attacker utilities')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel(data.features[1])

        limits1 = [att_mesh.T[0].min(), att_mesh.T[0].max()]
        limits2 = [att_mesh.T[1].min(), att_mesh.T[1].max()]
        ex = limits1 + limits2
        # ax.imshow(attack_utils, vmin=vmin, vmax=vmax, cmap=plt.cm.gist_earth_r, extent=ex, interpolation='nearest', origin='lower', aspect='auto')
        ax.imshow(attack_utils, cmap=plt.cm.gist_earth_r, extent=ex, interpolation='nearest', origin='lower', aspect='auto')
        ax.set_xlim(limits1)
        ax.set_ylim(limits2)

def plot_attacker_det_prb(actions, att_mesh, ax, data, g, mesh, util, x, att_type, dist):
    print('Plotting undetection probability.')
    if att_type == 'replace':
        values = [(1-g.thetas[att_a])  for att_a in actions]
        # values = [(1 - p) for a, p in zip(actions, g.thetas)]
    elif att_type == 'add':
        # values = [util(mesh[a]) * ( 1 - sum([dist(mesh[a]+mesh[a_]) * p for a_, p in zip(actions, g.thetas)]))  for a in actions]
        values = [(sum([dist(mesh[a] - att_mesh[att_a])
                                            for a in actions])
                                       - sum([dist(mesh[a] - att_mesh[att_a]) * (g.thetas[a]) for a in actions]))  for att_a in actions]
    attack_utils = np.array(values).reshape(x[0].shape)


    if len(data.features) == 1:
        ax.set_title('Attacker undet probability')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel('utility')
        attack_utils = np.array(values).reshape(x[0].shape)
        ax.plot(att_mesh.flatten(), attack_utils)

    if len(data.features) == 2:
        ax.set_title('Attacker undet probability')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel(data.features[1])

        limits1 = [att_mesh.T[0].min(), att_mesh.T[0].max()]
        limits2 = [att_mesh.T[1].min(), att_mesh.T[1].max()]
        ex = limits1 + limits2
        # ax.imshow(attack_utils, cmap=plt.cm.gist_earth_r, extent=np.ndarray.flatten(data.limits), vmin=vmin, vmax=vmax, interpolation='nearest', origin='lower')
        ax.imshow(attack_utils, cmap=plt.get_cmap('Greys'), extent=ex, interpolation='nearest', origin='lower', aspect='auto')
        ax.set_xlim(limits1)
        ax.set_ylim(limits2)

def plot_defender_fp(actions, ax, data, g, mesh, prb, x, discret):
    print('Plotting defender fpr')
    if len(data.features) == 1:
        ax.set_title('Real false-positives')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel('probability')
        xs = np.c_[np.linspace(data.mins[0],data.maxs[0],discret)]
        ax.plot(xs, [prb.getPrb(mesh[a]) * g.thetas[a] for a in actions])

    if len(data.features) == 2:
        ax.set_title('Real false-positives')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel(data.features[1])
        values = np.array([prb.getPrb(mesh[a]) * g.thetas[a] for a in actions]).reshape(x[0].shape)
        ax.imshow(values,
            cmap=plt.get_cmap('Greys'), extent=np.ndarray.flatten(data.limits), origin='lower', aspect='auto')
        ax.set_xlim(data.limits[0])
        ax.set_ylim(data.limits[1])

def plot_defender_dist(actions, ax, data, mesh, prb, x):
    print('Plotting defender dist.')
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
            cmap=plt.get_cmap('Greys'), extent=np.ndarray.flatten(data.limits), origin='lower', aspect='auto')
        ax.set_xlim(data.limits[0])
        ax.set_ylim(data.limits[1])

def plot_defender_strategy(ax, data:DatasetFeatures, decisions, discret):
    print('Plotting defender strategy')
    if len(data.features) ==1:
        ax.set_title('Defender probabilities')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel('probability')
        xs = np.c_[np.linspace(data.mins[0],data.maxs[0],discret)]
        ax.plot(xs, decisions)

    if len(data.features) == 2:
        ax.set_title('Defender probabilities')
        ax.set_xlabel(data.features[0])
        ax.set_ylabel(data.features[1])
        ax.imshow(decisions, cmap=plt.get_cmap('Greys'), extent=np.ndarray.flatten(data.limits),
            vmin=0, vmax=1, interpolation='nearest', origin='lower', aspect='auto')
        ax.set_xlim(data.limits[0])
        ax.set_ylim(data.limits[1])


def plot_FP_vs_utility():
    discretize = 20
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



def aggregate(df, fce):
    n = 50
    list_df = [df[i:i+n] for i in range(0, df.shape[0]-n-1)]
    # print(list_df[0])

    stats = dict()
    for f in fce:
        stats.update({f:list()})
    # means = []
    # stds = []

    for line in list_df:
        for f in fce:
            stats.get(f).append(line.agg(f)['value'])
    return pd.DataFrame({name:stats.get(name) for name in fce})
    # return pd.DataFrame({'mean': means, 'std':stds})

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
    parser.add_argument('-t', '--att_type', help='Attacker type [replace|add]', action='store', required=False)
    parser.add_argument('--dist', help='Where to get the data [Generate|File]', action='store', required=False)
    args = parser.parse_args()

    np.random.seed(42)

    if args.alg is False:
        args.alg = 'nn'
        # args.alg = 'linProg'
        # args.data = 'generate'
        # args.data = 'file_raw'
        args.data = 'file_featured'
        # args.datafile = '/home/kori/data/projects/NAB/data/artificialWithAnomaly/art_daily_flatmiddle.csv'
        args.datafile = '/home/kori/data/projects/avast-playground/adversarial-anomaly-detectior/data/data_5000_features_18_01_2019.pkl'
        args.att_type =  'replace'
        # args.att_type =  'add'


    if args.data == 'generate':
        # data = DatasetFeatures.from_normal_distribution_independent(features=['F1'], mean=[5], var=[1.0], size=1000)
        data = DatasetFeatures.from_normal_distribution_dependent(features=['Mean', 'Std'], mean=[5, 5.0], covar=[[1.5, 2.0],[2, 2. ]], size=1000)
        # data = DatasetFeatures.from_normal_distribution_independent(features=['F1', 'F2'], mean=[0.5, 0.5], var=[[0.1, 0.2],[.1, .1 ]], size=1000)
        # data = DatasetFeatures.from_normal_distribution_independent(features=['F1', 'F2'], mean=[0.5, 0.5], var=[0.1, 0.2], size=1000)
        # data = DatasetFeatures.from_normal_distribution_dependent_first(features=['F1'], mean=[0.5, 0.5], covar=[[1.0, 0.0],[20, 1. ]], size=1000)
        # print(data.data)
    elif args.data == 'file_raw':
        df = pd.read_csv(args.datafile)
        # df_agg = aggregate(df, ['mean'])
        df_agg = aggregate(df, ['mean','std'])
        # df_agg = aggregate(df, ['mean','sum'])
        # df_agg = aggregate(df, ['sum','std'])
        # df_agg = aggregate(df, ['mean','std','sum'])
        data = DatasetFeatures(df_agg, df_agg.columns)
    elif args.data == 'file_featured':
        df = pd.read_pickle(args.datafile)
        print(df.columns)
        # df = pd.read_csv(args.datafile)
        df['num_letters_norm'] = (df['num_letters'] - df['num_letters'].mean())/df['num_letters'].std()
        df['length_norm'] = (df['length'] - df['length'].mean())/df['length'].std()
        data = DatasetFeatures(pd.DataFrame(df[['entropy','length_norm']]), ['entropy', 'length_norm'])
        # data = DatasetFeatures(pd.DataFrame(df[['num_letters']), ['num_letters'])


    if args.alg == 'linProg':
        solve_simple_game(data, discretize=25, FPrate=0.10, att_type=args.att_type, discount=0, plot=True)
    elif args.alg == 'nn':
        solve_simple_game_with_nn(data, FPrate=0.1, att_type=args.att_type, discount=0, plot=True)

        # solve_simple_game(data, discretize=4, FPrate=0.01, att_type=args.att_type, discount=0, plot=False)
        # solve_simple_game(data, discretize=5, FPrate=0.01, att_type='replace', discount=0, plot=True)
