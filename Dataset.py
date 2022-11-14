import numpy as np
import typing
from typing import List
from Constants import *

class Dataset:
    def __init__(self, q2_zero, q2_plus, ff_zero, ff_plus, cov):
        self.q2_zero = np.array(q2_zero)
        self.q2_plus = np.array(q2_plus)
        self.ff_zero = np.array(ff_zero)
        self.ff_plus = np.array(ff_plus)
        self.cov = cov
        self.invcov = None
        #np.linalg.inv(ffdata.cov)
        self.n_zero = len(q2_zero)
        self.n_plus = len(q2_plus)
    
    def resample(self,n: int,ranseed = None):
        # if seed != None:
        #     np.random.seed(ranseed) 
        resampled = np.random.multivariate_normal(self.fflist(), self.cov, n)
        resampled_zero = resampled[:,:self.n_zero]
        resampled_plus = resampled[:,self.n_zero:]
        dataset_list = []
        for i in range(n):
            #ds_i = Dataset(self.q2_zero, self.q2_plus, resampled_zero[i], resampled_plus[i], self.cov)
            #ds_i.set_dbinputs((np.array(self.q2_zero), np.array(resampled_zero[i]), chi0plusBsK,zpluspole),(np.array(self.q2_plus), np.array(resampled_plus[i]), chi1minusBsK,zpluspole))
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
    
    def set_dbinputs(self,fzinputs,fpinputs):
        self.fpinputs = fpinputs
        self.fzinputs = fzinputs
        print(fzinputs)
        print((np.array(self.q2_zero), np.array(self.ff_zero), chi0plusBsK,zpluspole))
        print(fpinputs)
        print((np.array(self.q2_plus), np.array(self.ff_plus), chi1minusBsK,zpluspole))