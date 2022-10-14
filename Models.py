import numpy as np
from Constants import *
from Dataset import Dataset
import scipy

class Model:
    def predict(self, q2_zero: list, q2_plus: list, coefficients: list):
        raise NotImplementedError()
    
    def constraints(self):
        raise NotImplementedError()
    
    def loss(self, predictedffs: list, true_data: Dataset):
        trueffs = true_data.fflist()
        invcov = true_data.get_inv_cov()
        ssum = 0
        for i in range(len(predictedffs)):
            for j in range(len(predictedffs)):
               ssum += (predictedffs[i]-trueffs[i])*invcov[i, j]*(predictedffs[j]-trueffs[j])
        return ssum

    def fit(self, ds: Dataset, init_params: list, tolerance = 1e-04):
        obj_func = lambda coefficients: self.loss(self.predict(ds.q2_zero, ds.q2_plus, coefficients), ds)
        return scipy.optimize.minimize(obj_func, init_params, method='trust-constr', constraints=self.constraints(), tol = tolerance).x
    
    def __init__(self, n_zero, n_plus):
        self.n_zero = n_zero
        self.n_plus = n_plus
    
    def ffat0(self, coeffs: list):
        return self.predict([],[0],coeffs)[0]

    
        
        
class ZFitModel(Model):
    def predict(self, q2_zero: list, q2_plus: list, coefficients: list):
        returnlist = []
        if self.n_zero is None:
            self.n_zero = 3
        for t in q2_zero:
            tempsum = 0
            for n, an in enumerate(coefficients[:self.n_zero]):
                tempsum += 1/phizero(t)*an*z(t)**n #(1-t/(mBstar**2))*an*z(t)**n
            returnlist.append(tempsum)
        for t in q2_plus:
            tempsum = 0
            for n, an in enumerate(coefficients[self.n_zero:]):
                tempsum += 1/phiplus(t)*an*z(t)**n #(1-t/(mBstar**2))*an*z(t)**n
            returnlist.append(tempsum)
        return returnlist
    def constraints(self):
        return [{'type': 'eq', 'fun': lambda inlist: self.predict([0], [], inlist)[0]-self.predict([], [0], inlist)[0]}, {'type': 'ineq', 'fun': lambda inlist: 1- sum([inlist[i]**2 for i in range(len(inlist))])}]
    
class AltModel(Model):
    def predict(self, q2_zero: list, q2_plus: list, coefficients: list):
        returnlist = []
        for t in q2_zero:
            tempsum = 0
            for n, an in enumerate(coefficients[:self.n_zero]):
                tempsum += an*alt_polynomial(t, n) #(1-t/(mBstar**2))*an*z(t)**n
            returnlist.append(tempsum/phizero(t))
        for t in q2_plus:
            tempsum = 0
            for n, an in enumerate(coefficients[self.n_zero:]):
                tempsum += an*alt_polynomial(t, n) #(1-t/(mBstar**2))*an*z(t)**n
            returnlist.append(tempsum/phiplus(t))
        return returnlist
    def constraints(self):
        return [{'type': 'eq', 'fun': lambda inlist: self.predict([0], [], inlist)[0]-self.predict([], [0], inlist)[0]}, {'type': 'ineq', 'fun': lambda inlist: 1- sum([inlist[i]**2 for i in range(len(inlist))])}]
    

    class DispBound:
        def __init__(self,dataset: Dataset):
            self.ds = dataset
        def predict_bounds(self, q2out: list, nboot = 100, intervaltype = "simple"):
            zero_bounds = []
            plus_bounds = []
            for i,q2 in enumerate(q2out):
                zero_bounds.append((i,i))
                plus_bounds.append((i,i))
            return zero_bounds,plus_bounds
        
#db.predict_bounds([list of q2 i care about finding the bounds for],nboot) -> return 2 lists of tuples: (f_0 lower val at q0, f_0 upper val at q0)...