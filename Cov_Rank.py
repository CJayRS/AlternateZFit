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
import matplotlib.pyplot as plt

xspace = np.linspace(17.5,23.5,601)
detspace = []
cond = []
for x in xspace:
    data = DatasetFactory.generate((17.5,x,23.5),(17.5,23.5))
    detspace.append(np.log(np.linalg.cond(data.cov)))
    #Eigs = np.linalg.eigvals(data.cov)
    #cond.append(np.log(np.max(Eigs)/np.min(Eigs)))

plt.scatter(xspace,detspace)
plt.xlabel("middle input val GeV^2")
plt.ylabel("natural log of condition number")
plt.show()

#print(np.linalg.eigvals(DatasetFactory.generate((17.5,17.6,23.5),(17.5,23.5)).get_inv_cov()))