import pandas as pd
import numpy as np


class CumulativeProbCached:

    def __init__(self, data):
        self.data = data

    @classmethod
    def from_file(cls, file_name, force=False):
        pass

    @classmethod
    def from_normal_distribution_independent(cls, labels=['F1', 'F2'], mean=[20, 10], var=[12, 4], size=1000, force=False):
        F = len(mean)
        data = pd.DataFrame()

        for l, m, v in zip(labels, mean, var):
            data[l] = np.random.normal(m, v, size)

        res = cls(data)
        return res


def testThisClass():
    c = CumulativeProbCached.from_normal_distribution_independent()

    print(c.data)

