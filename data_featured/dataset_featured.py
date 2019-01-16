import pandas as pd
import numpy as np


class DatasetFeatures:

    def __init__(self, data:pd.DataFrame, features):
        self.data = data
        self.features = features

        mmin = data.min()
        mmax = data.max()
        feature_size = mmax - mmin

        # margin = 0.4
        margin = 0.0
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

