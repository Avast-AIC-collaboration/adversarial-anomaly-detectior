import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class DatasetFeatures:

    def __init__(self, data:pd.DataFrame, features):
        self.data = data
        self.features = features


        mmin = data.min()
        mmax = data.max()
        feature_size = mmax - mmin

        margin = 0.4
        # margin = 1.4
        # margin = 0.0
        self.mins = data.min() - feature_size*margin
        self.maxs = data.max() + feature_size*margin
        self.limits = np.c_[self.mins, self.maxs]
        self.feature_size = self.maxs - self.mins
        #
        # print('creating datasetFeatured:')
        # print(self.data)
        # print(self.mins)
        # print(self.maxs)
        # print(self.limits)
        # print(self.feature_size)
        # print('done')

    def plot(self):
        fig, ax = plt.subplots(len(self.features))
        for i in range(len(self.features)):
            ax[i].hist(self.data[self.features[i]], bins=20)
            ax[i].set_xlabel(self.features[i])
        plt.show()

    @classmethod
    def from_file(cls, file_name, force=False):
        pass

    @classmethod
    def from_normal_distribution_independent(cls, features=['F1', 'F2'], mean=[0.3, 0.5], var=[0.1, 0.2], size=1000, force=False):
        data = pd.DataFrame()
        np.random.seed(seed=42)

        for l, m, v in zip(features, mean, var):
            data[l] = np.random.normal(m, v, size)

        res = cls(data, features=features)
        return res

    @classmethod
    def from_normal_distribution_dependent(cls, features, mean, covar, size, force=False):
        data = pd.DataFrame()
        np.random.seed(seed=42)

        X = np.random.multivariate_normal(mean, covar, size)
        # print(X)
        res = cls(pd.DataFrame(X, columns=features), features=features)
        return res

    @classmethod
    def from_normal_distribution_dependent_first(cls, features, mean, covar, size):
        data = pd.DataFrame()
        np.random.seed(seed=42)

        X = np.random.multivariate_normal(mean, covar, size)
        X = X[:,0]
        # print(X)
        res = cls(pd.DataFrame(X, columns=features), features=features)
        return res
