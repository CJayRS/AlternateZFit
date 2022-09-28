import numpy as np
import typing
from typing import list

class Dataset:
    def __init__(self,q2_zero,q2_plus,ff_zero,ff_plus,cov):
        self.q2_zero = q2_zero
        self.q2_plus = q2_plus
        self.ff_zero = ff_zero
        self.ff_plus = ff_plus
        self.cov = cov
        self.n_zero = len(q2_zero)
        self.n_plus = len(q2_plus)
    
    def resample(self,n: int) -> list[]: 
        resampled = np.random.multivariate_normal(np.append(self.ff_zero,self.ff_plus),self.cov,n)
        resampled_zero = resampled[:self.n_zero,:]
        resampled_plus = resampled[self.n_zero:,:]
        dataset_list = []
        for i in len(n):
            dataset_list.append(Dataset(self.q2_zero,self.q2_plus,resampled_zero[i],resampled_plus[i],self.cov))
        return dataset_list

