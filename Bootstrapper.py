from Dataset import *
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import freeze_support
from tqdm import tqdm

class Bootstrapper:
    @staticmethod
    def bootstrap(ds: Dataset, model_class, nboot):
        
        model = model_class(ds.n_zero, ds.n_plus)
        initial_guess = model.fit(ds, np.linspace(0.1, 0.1, ds.n_zero + ds.n_plus))
        resampled = ds.resample(nboot)
        def _worker_job(ds: Dataset):
            return model.fit(ds, initial_guess)
        with Pool(processes=None) as pool:
            outputcoeffs = []
            for coeff in tqdm(pool.imap(_worker_job, resampled), total = nboot):#map
                outputcoeffs.append(coeff)
            return outputcoeffs
        
        