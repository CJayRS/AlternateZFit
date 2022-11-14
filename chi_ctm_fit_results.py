#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 16:25:03 2022

@author: jflynn

Compute formfactors and covariance matrix for any choice of reference
kinematics based on HMChPT fit results.

@author: Andreas Juettner Andreas.Juttner@cern.ch, juettner@soton.ac.uk
         07.2022
@author: Jonathan Flynn j.m.flynn@soton.ac.uk
         29 July 2022 reworked into a class
"""

import numpy as np
import h5py
import masses

'''
Dictionaries for systematic variations of fits and mapping for indices of
coefficients relevant in the chiral-continuum limit.

pole+/-100MeV applies only to f0
pole+/-30MeV  applies only to f+

The arguments give the indices of the fit coefficients needed for f+ and f0
respectively in the chi-ctm limit.
'''
fitvariations={}
fitvariations['BstoK'] = {'chosen':            [[0,2],[0,2,3]],
                          'pole-100MeV':       [[0,2],[0,2,3]],
                          'pole+100MeV':       [[0,2],[0,2,3]],
                          'pole-30MeV':        [[0,2],[0,2,3]],
                          'pole+30MeV':        [[0,2],[0,2,3]],
                          'pmin1':             [[0,2],[0,2,3]],
                          'pmax3':             [[0,2],[0,2,3]],
                          'no FV':             [[0,2],[0,2,3]],
                          'nchi0':             [[0,1],[0,1,2]],
                          'nCL0nchi0':         [[0,1],[0,1,2]],
                          'nCL0analytic':      [[0,2],[0,2,3]],
                          'nCL0':              [[0,2],[0,2,3]],
                          'nE2':               [[0,2,3],[0,2,3]],
                          'nla':               [[0,2],[0,2,3]],
                          'include amqsq':     [[0,2],[0,2,3]],
                          'include amqalpha':  [[0,2],[0,2,3]],
                          'fPitofK':           [[0,2],[0,2,3]],
                          'fPitof0':           [[0,2],[0,2,3]],
                          'analytic':          [[0,2],[0,2,3]],
                          'amqsq and amqalpha':[[0,2],[0,2,3]]}

'''
Dictionaries for flat systematic errors from HMChPT fits.

These errors are applied to computed values of form factors and are
taken to be completely correlated. For ffvalues fi and fj, their 
covariance is sigma_i sigma_j fi fj, where sigma is read from the
dictionary below.
'''
flatsystematic={}
flatsystematic['BstoK']={'fsp':0.019,
                         'fs0':0.021}

def E_qsq(qsq,Mi,Mo):
   """
   final-state-energy E for given qsq
   """
   return (Mi**2+Mo**2-qsq)/(2.0*Mi)

def ff_qsq(qsq,Mi,Mo,pole,coeff):
  '''
  heavy-meson ChPT expression for ff in chi-ctm limit
  '''
  Evec=E_qsq(qsq,Mi,Mo)
  return [1./(E+pole)*np.sum([E**i*coeff[i] for i in range(len(coeff))]) for E in Evec]

def G_qsq(qsq,Mi,Mo,pole,coeff):
  '''
  list of terms in HMChPT expression in chi-ctm limit
  '''
  Evec=E_qsq(qsq,Mi,Mo)
  return np.array([1./(Evec+pole)*Evec**i for i in range(len(coeff))])

def Cffn_HMChPT(qsqp,qsq0,covp0,cp_BstoK,c0_BstoK,Deltap,Delta0,Min,Mout):
  '''
  covariance matrix for combined f+ and f0 at reference q^2 values
  '''
  Nqp=len(qsqp)
  Nq0=len(qsq0)
  Np=len(cp_BstoK)
  #N0=len(c0_BstoK) # not used
  Cf=np.zeros((Nqp+Nq0,Nqp+Nq0))
  for i in range(Nqp):
    Gvec0 = G_qsq(qsqp[i],Min,Mout,Deltap,cp_BstoK)
    for j in range(Nqp):
      Gvec1 = G_qsq(qsqp[j],Min,Mout,Deltap,cp_BstoK)
      Cf[i,j] = np.dot(Gvec0,np.dot(covp0[:Np,:Np],Gvec1)) 
    for j in range(Nq0):
      Gvec1 = G_qsq(qsq0[j],Min,Mout,Delta0,c0_BstoK)
      dum   = np.dot(Gvec0,np.dot(covp0[:Np,Np:],Gvec1))
      Cf[i,Nqp+j] = dum
      Cf[Nqp+j,i] = dum
  for i in range(Nq0):
    for j in range(Nq0):
      Gvec0 = G_qsq(qsq0[i],Min,Mout,Delta0,c0_BstoK)
      Gvec1 = G_qsq(qsq0[j],Min,Mout,Delta0,c0_BstoK)
      Cf[Nqp+i,Nqp+j] = np.dot(Gvec0,np.dot(covp0[Np:,Np:],Gvec1))
  return Cf 

"""
The hdf5 input file has for each fit a group 'result' or 'varresult' which
contains datasets
   'central'
   'error'
   'covariance'
   'Bootstraps'
-- The 'central' result comes from a fit to the 3pt ratios computed using the
   original ensemble.
-- The covariance has been computed using
     diff = boots - central
     cov = np.dot(diff.T,diff)/len(diff)
   and the errors from sqrt(diag(cov)).

In the code below, the bootstrap samples are read in and those for f+ and f0
combined. If mean_from_boots is True for synthetic_points(), then the central
values are
    boots.mean(axis=0)
and the covariance is found from
    diff = boots - boots.mean(axis=0)
    cov = np.dot(diff.T,diff)/len(diff)
This means that the central values, covariance matrix (and hence the errors)
differ from the inputs for 'central', 'covariance' and 'error'.

If mean_from_boots is False, the central values and covariances are 
computed as in the input hdf5 file:
     diff = boots - central
     cov = np.dot(diff.T,diff)/len(diff)

Combining the boot samples for f+ and f0 also means that the covariance 
between values for f+ and f0 can be computed, in addition to f+ with f+ and
f0 with f0.
"""

class HMChiPTfit():
    """
    Class for results of a chiral-continuum fit and variations of the fit
    for form factors f+ and f0 for semileptonic decay of incoming hadron
    Hin (a string) to outgoing hadron Hout (a string). For example:
        
        BstoKresults=chi_ctm_fit_results('Bs','K',qsqmin=17.5)
    
    qsqmin = suggested minimum q-squared/GeV^2 at which to compute
             form factors
    
    synthetic_points() computes synthetic data points for f+ and f0
    respectively using a list of q^2 values for f+ and a list for f0.
    """
    def __init__(self,Hin,Hout,qsqmin=0):
        self.decay=Hin+'to'+Hout
        self.f=h5py.File(self.decay+'.h5','r')
        self.Min=self.f[self.decay+'/kinematics/MParent'][()]
        self.Mout=self.f[self.decay+'/kinematics/MChild'][()]
        self.qsqmax=self.f[self.decay+'/kinematics/qsqmax'][()]
        assert self.Min==eval('masses.m'+Hin)
        assert self.Mout==eval('masses.m'+Hout)
        assert self.qsqmax==(self.Min-self.Mout)**2
        self.qsqmin=qsqmin

        self.variations=fitvariations[self.decay]
        self.plusvars=self.f[self.decay+'/plus/Variations']
        self.zerovars=self.f[self.decay+'/zero/Variations']
        self.pluschosen=self.f[self.decay+'/plus/Chosen']
        self.zerochosen=self.f[self.decay+'/zero/Chosen']
        
    def lsR(self): # cf ls -R
        """
        recursively visit all names in hdf5 input file and return a list
        """
        namelist=[]
        self.f.visit(namelist.append)
        return namelist

    def synthetic_points(self,qsqp,qsq0,fitvariation='chosen',
                         mean_from_boots=True,report=False):
        """
        Create synthetic data points at q-squared values qsqp and qsq0 for
        f+ and f0 respectively
       
        qsqp : list of q-squareds at which to compute f+
        qsql : list of q-squareds at which to compute f0
        fitvariation : string to choose chosen fit or a variation,
                       default is 'chosen'
        mean_from_boots: if True compute central values and covariance
                         of the input fit coeffs entirely from the boot
                         samples (as discussed above); if False use the
                         input central values themselves and cov matrix
                         computed using them in place of boot means.
        report : if True returns dictionary of info on the inputs
                 used in generating synthetic data points

        Returns concatenated list of f+(q^2) and f0(q^2) values and their
        covariance matrix, plus report (if requested) of choices used in
        chi-ctm fit for f+ and f0.
        """
        v=fitvariation
        indp=self.variations[v][0]
        ind0=self.variations[v][1]
        # 'chosen' isn't a key for the 'Variations' group, so asking for it
        # will trigger KeyError and give you the 'chosen' fit.
        try: # if have key for this variation, use it
            Deltap=self.plusvars[v+'/varchoices/pole'][()]
            cp=self.plusvars[v+'/varresult/central'][indp] # fp coeffs
            BScp=self.plusvars[v+'/varresult/Bootstraps'][:,indp]
            grpp=self.plusvars[v+'/varchoices']
        except KeyError: # otherwise use 'chosen' fit
            Deltap=self.pluschosen['choices/pole'][()]
            cp=self.pluschosen['result/central'][indp] # fp coeffs
            BScp=self.pluschosen['result/Bootstraps'][:,indp]
            grpp=self.pluschosen['choices']
        try: # if have key for this variation, use it
            Delta0=self.zerovars[v+'/varchoices/pole'][()]
            c0=self.zerovars[v+'/varresult/central'][ind0] # f0 coeffs
            BSc0=self.zerovars[v+'/varresult/Bootstraps'][:,ind0]
            grp0=self.zerovars[v+'/varchoices']
        except KeyError: # otherwise use 'chosen' fit
            Delta0=self.zerochosen['choices/pole'][()]
            c0=self.zerochosen['result/central'][ind0] # f0 coeffs
            BSc0=self.zerochosen['result/Bootstraps'][:,ind0]
            grp0=self.zerochosen['choices']

        # choicesp['nE'] = max power of E in HMChPT f+, so expect
        # Npl=1+choicesp['nE']. Likewise expect N0l=1+choices0['nE'].
        Npl=len(cp) # number of parameters f+
        N0l=len(c0) # number of parameters f0

        # combine samples into single array
        BSall=np.r_['1',BScp,BSc0]  
        if mean_from_boots:
            bsmean=BSall.mean(axis=0)
            bscov=np.cov(BSall,rowvar=False,ddof=0)
        else: # use central values
            bsmean=np.concatenate((cp,c0))
            diff=BSall-bsmean
            bscov=np.dot(diff.T,diff)/len(diff)
            
    
        # construct central ff values and cov matrix for reference qsq values
        ffp = ff_qsq(qsqp,self.Min,self.Mout,Deltap,bsmean[:Npl])
        ff0 = ff_qsq(qsq0,self.Min,self.Mout,Delta0,bsmean[Npl:])
        ffcov = Cffn_HMChPT(qsqp,qsq0,bscov,bsmean[:Npl],bsmean[Npl:],
                          Deltap,Delta0,self.Min,self.Mout)

        if report:
           rpt={}
           rpt['variation']=v
           rpt['f+']={}
           rpt['f+']['choices']=dict([(k,grpp[k][()]) for k in grpp.keys()])
           rpt['f+']['num coeffs']=Npl
           rpt['f+']['coeffs']=cp
           rpt['f+']['num qsq']=len(qsqp)
           rpt['f+']['qsq']=qsqp
           if len(qsqp)>Npl:
               rpt['f+']['warn']='more qsq points than input coeffs'
           rpt['f0']={}
           rpt['f0']['choices']=dict([(k,grp0[k][()]) for k in grp0.keys()])
           rpt['f0']['num coeffs']=N0l
           rpt['f0']['coeffs']=c0
           rpt['f0']['num qsq']=len(qsq0)
           rpt['f0']['qsq']=qsq0
           if len(qsq0)>N0l:
               rpt['f0']['warn']='more qsq points than input coeffs'
           rpt['input cov']=bscov
 
           return np.concatenate((ffp,ff0)),ffcov,rpt
        else:
           return np.concatenate((ffp,ff0)),ffcov

    def flat_covariance(self,ffp,ff0):
       """Return a covariance matrix for flat systematic errors given
          input f+ and f0 form factor values ffp and ff0 respectively.
       """
       sp=flatsystematic[self.decay]['fsp']
       s0=flatsystematic[self.decay]['fs0']
       fl=np.concatenate((sp*ffp,s0*ff0))
       return np.outer(fl,fl)
    
   
if __name__=="__main__":
    """
    usage example
    """
    from disperr import disperr
    
    variations=fitvariations['BstoK']
    bstok=HMChiPTfit('Bs','K',qsqmin=17.5)
    qsqmin,qsqmax=bstok.qsqmin,bstok.qsqmax
    
    for mfb in (True,False):
        if mfb:
            print('Use boostrap means for central values and in computing covmat')
        else:
            print('\nUse central fit values in place of bootstrap means')
        for iv,v in enumerate(variations.keys()): # loop over systematic variations       
            Npl=len(variations[v][0])
            N0l=len(variations[v][1])
            qsqp=np.linspace(qsqmin,qsqmax,Npl) # Npl equispaced ref qsq values
            qsq0=np.linspace(qsqmin,qsqmax,N0l) # N0l equispaced ref qsq values 
            ffl,ffcov,rpt=bstok.synthetic_points(qsqp,qsq0,v,report=True,
                                                 mean_from_boots=mfb)
            dff=np.sqrt(np.diag(ffcov))
            ffp=ffl[:Npl]
            ff0=ffl[Npl:]
            Deltap=rpt['f+']['choices']['pole']
            Delta0=rpt['f0']['choices']['pole']
            if iv == 0:
                print(19*' ','qsq values                  :',
                      ' '.join(['%.2f'%qsq for qsq in qsqp]),'/',
                      ' '.join(['%.2f'%qsq for qsq in qsq0]))
            print('%-19s pole+=%+.3f / pole-=%+.3f :'%(v,Deltap,Delta0),
                  ' '.join(disperr(ffp,dff[:Npl])),'/',
                  ' '.join(disperr(ff0,dff[Npl:])))
