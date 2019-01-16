from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from data_featured.dataset_featured import DatasetFeatures


class ProbabilityDistribution:
    def __init__(self, data_featured:DatasetFeatures, bins=10):
        self.data_featured = data_featured
        self.prb = gaussian_kde(data_featured.data.T)
        self.bins = bins
        self.prb_cache = dict()

    def getPrb(self, x):
        x_tup = tuple([round(elem, 5) for elem in x])
        if x_tup in self.prb_cache.keys():
            return self.prb_cache[x_tup]
        else:
            p0 = x
            p1 = [v + self.data_featured.feature_size[i]/(self.bins-1) for i,v in enumerate(x)]
            val = self.prb.integrate_box(p0, np.array(p1), maxpts=50)
            self.prb_cache.update({x_tup:val})
            # print((p0, p1, val))
            return val


    def plot(self, ax):
        if self.prb.d == 1:
            xs = np.c_[np.linspace(self.data_featured.mins[0],self.data_featured.maxs[0],self.bins)]
            print(xs)
            self.prb.covariance_factor = lambda : .25
            self.prb._compute_covariance()
            ax.plot(xs, np.array(list(map(self.getPrb, xs))))
            ax.set_title('Probability distribution')
            ax.set_xlabel(self.data_featured.features[0])
            ax.set_ylabel('probability')

        if self.prb.d == 2:
            X, Y = np.mgrid[self.data_featured.mins[0]:self.data_featured.maxs[0]:(self.bins)*1j,
                    self.data_featured.mins[1]:self.data_featured.maxs[1]:(self.bins)*1j]
            # positions = np.vstack([X.ravel(), Y.ravel()])

            positions = np.stack([X.ravel(), Y.ravel()], axis=-1)
            # print(positions)
            Z = np.array(list(map(self.getPrb, positions))).reshape(X.shape)
            # print(Z.sum())
            Z = Z/Z.sum()
            # print(Z)
            # print(('Z.sum()=', Z.sum()))

            # print(np.ndarray.flatten(self.data_featured.limits.T))
            ax.imshow(np.rot90(Z), cmap=plt.get_cmap('Greys'), extent=np.ndarray.flatten(self.data_featured.limits), aspect='auto')
            ax.plot(self.data_featured.data[self.data_featured.features[0]], self.data_featured.data[self.data_featured.features[1]], 'k.', markersize=2)

            ax.set_xlim(self.data_featured.limits[0])
            ax.set_ylim(self.data_featured.limits[1])
            ax.set_title('Probability distribution')
            ax.set_xlabel(self.data_featured.features[0])
            ax.set_ylabel(self.data_featured.features[1])

            # plt.show()

        else:
            print('Cannot plot distribution for dim>2.')
