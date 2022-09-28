import numpy as np
import typing
from typing import List

class Dataset:
    def __init__(self, q2_zero, q2_plus, ff_zero, ff_plus, cov):
        self.q2_zero = q2_zero
        self.q2_plus = q2_plus
        self.ff_zero = ff_zero
        self.ff_plus = ff_plus
        self.cov = cov
        self.invcov = None
        #np.linalg.inv(ffdata.cov)
        self.n_zero = len(q2_zero)
        self.n_plus = len(q2_plus)
    
    def resample(self,n: int): 
        resampled = np.random.multivariate_normal(self.fflist(), self.cov, n)
        resampled_zero = resampled[:,:self.n_zero]
        resampled_plus = resampled[:,self.n_zero:]
        dataset_list = []
        for i in range(n):
            dataset_list.append(Dataset(self.q2_zero, self.q2_plus, resampled_zero[i], resampled_plus[i], self.cov))
        return dataset_list

    def fflist(self):
        return np.append(self.ff_zero, self.ff_plus)
    
    def q2list(self):
        return np.append(self.q2_zero, self.q2_plus)

    def get_inv_cov(self):
        if self.invcov is None: 
            self.invcov = np.linalg.inv(self.cov)
        return self.invcov