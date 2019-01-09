import pandas as pd
import numpy as np


class DatasetFeatures:

    def __init__(self, data, features):
        self.data = data
        self.features = features

    @classmethod
    def from_file(cls, file_name, force=False):
        pass

    @classmethod
    def from_normal_distribution_independent(cls, features=['F1', 'F2'], mean=[20, 10], var=[12, 4], size=1000, force=False):
        data = pd.DataFrame()

        for l, m, v in zip(features, mean, var):
            data[l] = np.random.normal(m, v, size)

        res = cls(data, features=features)
        return res


# def testThisClass():
#     c = DatasetFeatures.from_normal_distribution_independent()
#     print(c.data)
