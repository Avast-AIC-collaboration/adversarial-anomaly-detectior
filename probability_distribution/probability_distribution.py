from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from data_featured.dataset_featured import DatasetFeatures


class ProbabilityDistribution:
    def __init__(self, data_featured:DatasetFeatures, step=0.02):
        self.data_featured = data_featured
        self.prb = gaussian_kde(data_featured.data.T)
        self.step = step

    def getPrb(self, x):
        return self.prb.integrate_box(x, np.array([i+self.step for i in x]))


    def plot(self, ax):
        if self.prb.d == 1:
            xs = np.linspace(0,1,1/self.step)
            self.prb.covariance_factor = lambda : .25
            self.prb._compute_covariance()
            plt.plot(xs, self.prb(xs))
            plt.show()

        if self.prb.d == 2:
            X, Y = np.mgrid[0:1:(1./self.step)*1j, 0:1:(1./self.step)*1j]
            # positions = np.vstack([X.ravel(), Y.ravel()])
            positions = np.stack([X.ravel(), Y.ravel()], axis=-1)
            # print(positions)
            Z = np.array(list(map(self.getPrb, positions))).reshape(X.shape)
            Z = Z/Z.sum()
            print(('Z.sum()=', Z.sum()))
            # Z = np.reshape(self.prb(positions).T, X.shape)
            # print(('Z', Z.min(), Z.max(), Z.sum()))
            # Z = Z / Z.sum()
            # print(positions)
            # print(Z)
            # print(self.prb.integrate_box([0,0], [1,1]))

            cax = ax.imshow(np.rot90(Z), cmap=plt.get_cmap('Greys'), extent=[0, 1, 0, 1])
            # ax.plot(self.data_featured.data[self.data_featured.features[0]], self.data_featured.data[self.data_featured.features[1]], 'k.', markersize=2)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_title('Probability distribution')
            ax.set_xlabel(self.data_featured.features[0])
            ax.set_ylabel(self.data_featured.features[1])
            # plt.colorbar(cax, ticks=[0,1])

            # plt.show()

        else:
            print('Cannot plot distribution for dim>2.')
