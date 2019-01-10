from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from data_featured.dataset_featured import DatasetFeatures


class ProbabilityDistribution:
    def __init__(self, data_featured:DatasetFeatures):
        self.data_featured = data_featured
        self.prb = gaussian_kde(data_featured.data.T)

    def getPrb(self, x):
        # todo: compute correcntly the probability mass function is epsilon area?
        pass


    def plot(self, ax):
        if self.prb.d == 1:
            xs = np.linspace(0,1,10)
            self.prb.covariance_factor = lambda : .25
            self.prb._compute_covariance()
            plt.plot(xs, self.prb(xs))
            plt.show()

        if self.prb.d == 2:
            X, Y = np.mgrid[0:1:10j, 0:1:10j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            # print(positions)
            Z = np.reshape(self.prb(positions).T, X.shape)
            Z = Z / Z.sum()
            print(('Z', Z.min(), Z.max(), Z.sum()))
            # print(Z)
            print(self.prb.integrate_box([0,0], [1,1]))

            cax = ax.imshow(np.rot90(Z), cmap=plt.get_cmap('Greys'), extent=[0, 1, 0, 1], vmin=0, vmax=1, interpolation='nearest')
            # ax.plot(self.data_featured.data[self.data_featured.features[0]], self.data_featured.data[self.data_featured.features[1]], 'k.', markersize=2)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            # plt.colorbar(cax, ticks=[0,1])

            # plt.show()

        else:
            print('Cannot plot distribution for dim>2.')
