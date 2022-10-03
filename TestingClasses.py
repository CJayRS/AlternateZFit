from ast import main
from copyreg import pickle
import numpy as np
from Constants import *
from Dataset import Dataset
from Models import *
from DatasetFactory import DatasetFactory
from Bootstrapper import *
import warnings
import pickle
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

if __name__ == '__main__':
    num_insamples = 20
    nboot = 500
    inputq2_list = np.random.uniform(17.5, 23.5, (num_insamples, 5))
    inputq2zero = np.sort(inputq2_list[:,:3])
    inputq2plus = np.sort(inputq2_list[:,3:])
    
    output_dict_z = {}
    output_dict_alt = {}
    for i in range(num_insamples):
        q2in = (inputq2zero[i].tolist(), inputq2plus[i].tolist())
        data = DatasetFactory.generate(*q2in)
        outval = Bootstrapper.bootstrap(data,ZFitModel,nboot)
        outvalalt = Bootstrapper.bootstrap(data,AltModel,nboot)
        listindex = (tuple(q2in[0]),tuple(q2in[1]))
        output_dict_z[listindex] = outval
        output_dict_alt[listindex] = outvalalt
    

    f = open("z_outdata_p9.pkl", "wb")
    pickle.dump(output_dict_z,f)
    f.close()
    f = open("alt_outdata_p9.pkl", "wb")
    pickle.dump(output_dict_alt,f)
    f.close()