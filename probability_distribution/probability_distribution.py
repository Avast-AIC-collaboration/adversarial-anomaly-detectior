from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from data_featured.dataset_featured import DatasetFeatures


class ProbabilityDistribution:
    def __init__(self, data_featured:DatasetFeatures):
        self.data_featured = data_featured
        self.prb = gaussian_kde(data_featured.data.T)


    def plot(self):
        if self.prb.d == 1:
            xs = np.linspace(0,1,10)
            self.prb.covariance_factor = lambda : .25
            self.prb._compute_covariance()
            plt.plot(xs, self.prb(xs))
            plt.show()

        if self.prb.d == 2:
            X, Y = np.mgrid[0:1:100j, 0:1:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(self.prb(positions).T, X.shape)

            fig, ax = plt.subplots()
            ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
                extent=[0, 1, 0, 1])
            ax.plot(self.data_featured.data[self.data_featured.features[0]], self.data_featured.data[self.data_featured.features[1]], 'k.', markersize=2)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            plt.show()

        else:
            print('Cannot plot distribution for dim>2.')
