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

for x in xspace:
    data = DatasetFactory.generate((17.5,23.5),(17.5,x,23.5))
    detspace.append(np.log(np.linalg.cond(data.cov)))

plt.scatter(xspace,detspace)
plt.show()