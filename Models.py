import numpy as np
from Constants import *
from Dataset import Dataset
import scipy
from BGL import *
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

def compute_bounds(tl,fzinputs,fpinputs):
    """Compute dispersive bounds at a list of t = qsq values given input
        data on f+ and f0
        tl : one or a list of t values
        fpinputs = (tinl,fl,chi)
            tinl : t-values where form factor f+ is known
            fl : corresponind f+ form factor values
            chi : chi for f+
            zpole : pole location for f+
        fzinputs = (zl,fl,chi)
            zl : z-values where form factor f0 is known
            fl : corresponind f0 form factor values
            chi : chi for f0
            zpole : pole location for f0

        returns 1D array fp,dfp,fz,dfz if tl is a single value
        returns array of shape (nt,4) if tl has nt values
        returns np.array([0,0,0,0]) if unitarity check on inputs fails
    """
    tpin,fpin,chip,zppole=fpinputs
    tzin,fzin,chiz,zzpole=fzinputs
    zpin=zed(tpin,tcut,t0)
    zzin=zed(tzin,tcut,t0)

    #phifp=phi(tin,3,2,tcut,t0,tm,(eta,48.0*pi,1.0))*blaschke(tin,tcut,fpluspoles)*fpin
    #phifz=phi(tin,1,1,tcut,t0,tm,(eta,16.0*pi/(tp*tm),1.0))*blaschke(tin,tcut,fzeropoles)*fzin
    # if only one pole for each form factor, calculate Blaschke factors more directly
    Fp=phi(tpin,3,2,tcut,t0,tm,(etaBsK,48.0*pi,1.0))*((zpin-zppole)/(1.0-zpin*zppole))*fpin
    #Fz=phi(tzin,1,1,tcut,t0,tm,(eta,16.0*pi/(tp*tm),1.0))*((zzin-zzpole)/(1.0-zzin*zzpole))*fzin
    Fz=phi(tzin,1,1,tcut,t0,tm,(etaBsK,16.0*pi/(tp*tm),1.0))*fzin

    dil=[np.prod((1.0-z*np.delete(zpin,i))/(z-np.delete(zpin,i))) for i,z in enumerate(zpin)]
    Fpdl=Fp*dil*(1.0-zpin**2)
    Gp=Gmatrix(zpin)
    chimchibarp=chip-np.dot(Fpdl,np.dot(Gp,Fpdl)) # should be positive
    if chimchibarp<0.0:
        print('unitarity failed for f+ inputs: ',chimchibarp)
        return np.array([0,0,0,0])

    dil=[np.prod((1.0-z*np.delete(zzin,i))/(z-np.delete(zzin,i))) for i,z in enumerate(zzin)]
    Fzdl=Fz*dil*(1.0-zzin**2)
    Gz=Gmatrix(zzin)
    chimchibarz=chiz-np.dot(Fzdl,np.dot(Gz,Fzdl)) # should be positive
    if chimchibarz<0.0:
        print('unitarity failed for f0 inputs: ',chimchibarz)
        return np.array([0,0,0,0])

    # start t- and z0-dependent stuff
    if type(tl)==float:
        tl=[tl]
    boundsl=np.zeros((len(tl),4))
    for i,t in enumerate(tl):
        z0=zed(t,tcut,t0)

        jot=1.0e-6
        dtl=np.abs(t-tpin)
        if np.min(dtl)<jot: # numerically avoid problems it t is one of the input t-values
            fp,dfp=fpin[np.argmin(dtl)],0.0
        else:
            d0=np.prod((1.0-z0*zpin)/(z0-zpin))
            phipt=phi(t,3,2,tcut,t0,tm,(etaBsK,48.0*pi,1.0))*((z0-zppole)/(1.0-z0*zppole))
            fp,dfp=(-np.dot(Fpdl,1.0/(zpin-z0))/d0,
                    sqrt(chimchibarp/(1-z0*z0))/abs(d0))/phipt

        dtl=np.abs(t-tzin)
        if np.min(dtl)<jot:
            fz,dfz=fzin[np.argmin(dtl)],0.0
        else:
            d0=np.prod((1.0-z0*zzin)/(z0-zzin))
            #phizt=phi(t,1,1,tcut,t0,tm,(eta,16.0*pi/(tp*tm),1.0))*((z0-zzpole)/(1.0-z0*zzpole))
            phizt=phi(t,1,1,tcut,t0,tm,(etaBsK,16.0*pi/(tp*tm),1.0))
            fz,dfz=(-np.dot(Fzdl,1.0/(zzin-z0))/d0,
                    sqrt(chimchibarz/(1-z0*z0))/abs(d0))/phizt

        boundsl[i]=fz,dfz,fp,dfp

    if boundsl.shape==(1,4):
        return boundsl[0]
    else:
        return boundsl
class DispBound:
    def predict_zero_bounds(self, ds: Dataset):
        
        fz,dfz,fp,dfp = compute_bounds(0.0,(np.array(ds.q2_zero), np.array(ds.ff_zero), chi0plusBsK,zpluspole),(np.array(ds.q2_plus), np.array(ds.ff_plus), chi1minusBsK,zpluspole))
        return (fz-dfz,fz+dfz), (fp-dfp, fp+dfp)
        
#db.predict_bounds([list of q2 i care about finding the bounds for],nboot) -> return 2 lists of tuples: (f_0 lower val at q0, f_0 upper val at q0)...


