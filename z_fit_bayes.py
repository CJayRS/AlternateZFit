# -*- coding: utf-8 -*-
"""Created Oct 2022

z-fit by Bayesian inference using synthetic input f(q^2) values and
their covariance matrix

@author: jflynn

"""
from math import sqrt
import numpy as np
import scipy.special as spsp
from kinematic_functions import zed#,qsq,k,wl 

def pvalue(Ndof,chisq):
  """Goodness-of-fit p-value

  Returns probability that a chi-squared value > chisq
  would occur in a chi-squared distribution with Ndof
  degrees of freedom. Result is given by the complemented
  incomplete gamma function gammaincc(Ndof/2,chisq/2),
  which is 1 - CDF(chisqdistrib(Ndof,chisq)) where CDF
  means cumulative distribution function. Numerical recipes
  uses Q for gammaincc.
  """
  return spsp.gammaincc(Ndof/2.,chisq/2.)

def Yzc(z,zmax,p0lbl,nbplus,nbzero,fpseq,fpden,f0seq,f0den):
    """Calculate a *column* of the Yz matrix with a^0_0, (zeroth
       coefficient of z-expansion for f0(), expressed in terms
       of a^+_0, ... , a^+_{nbplus-1) and a^0_1, ... ,
       a^0_{nbzero-1)
    
       Yz(z,p0lbl,nbplus,nbzero)=[Y_0(z),Y_1(z),...,Y_N(z)]
       
       where N=nbplus+nbzero-2
    """
    if p0lbl=='p': # if lbl is p(lus), upper part is Y_K^+(z)
      seqp=np.array(fpseq(z,nbplus))/fpden(z)
      seq0=np.zeros(nbzero-1)
    else: # lbl is 0(zero), lower part is Y_k^0(z)
      seqp=np.array(fpseq(zmax,nbplus))*f0den(zmax)/(fpden(zmax)*f0den(z))
      seq0=(np.array(f0seq(z,nbzero))-np.array(f0seq(zmax,nbzero)))[1:]/f0den(z)
    return np.concatenate((seqp,seq0))

def fpmzmax(zmax,nbplus,nbzero,fpseq,fpden,f0seq,f0den):
    """Calculate the z-dept coefficients such that
    
       np.dot(fpmzmax,bparams) = fplus(zmax,bplus)-fzero(zmax,bzero)
       
       where bparams is the concatenation of the parameters
       bplus and bzero
       
       lambda*(fpmzmax.bparams) is added to the chi-squared, to
       implement the kinematic constraint, where lambda is a Lagrange
       multiplier
    """
    seqp=np.array(fpseq(zmax,nbplus))/fpden(zmax)
    seq0=np.array(f0seq(zmax,nbzero))/f0den(zmax)
    return np.concatenate((seqp,-seq0))


class z_fit():
    """
    Perform a z-fit to a set of input form-factor pts, given by q-squared
    locations and form factor values at those locations.
    
    arguments
        fittype: a class with functions fplus, fplusseq, fplusden and
                 fzero, fzeroseq, fzeroden; eg BCL, BGL
        qsqp: qsq values for f+ points
        qsq0: qsq values for f0 points
        ff: list of f+(qsqp_i) and f0(qsqp0_i) values
        cov: covariance matrix of the f+(qsq) and f0(qsq) values
        nbplus,nbzero: number of terms in z series for f+ and f0
                       respectively; by default same as number of
                       input f+ and f0 points respectively
    """

    def __init__(self,fittype,qsqp,qsq0,ff,cov,
                 nbplus=0,nbzero=0):
        #npts=len(bskpts)
        self.qsqp=qsqp
        self.qsq0=qsq0
        self.ff=ff
        self.cov=cov
        self.nplus=len(qsqp)
        self.nzero=len(qsq0)
        self.ptlbls=[char for char in self.nplus*'p'+self.nzero*'0']
        self.nbplus=nbplus # number of terms in z series for f+
        if nbplus==0:# default is same as number of f+ input pts
            self.nbplus=len(self.qsqp)
        self.nbzero=nbzero # number of terms in z series for f0
        if nbzero==0: # default is same as number of f0 input pts
            self.nbzero=len(self.qsq0) 
        self.npars=self.nbplus+self.nbzero
        self.fittypelbl=fittype.fittypelbl
        self.fitlbl=fittype.fitlbl
        self.info=fittype.info()
        self.tcut=fittype.tcut
        self.t0=fittype.t0
        self.fplus=fittype.fplus
        self.fplus_seq,self.fplus_den=fittype.fplus_seq,fittype.fplus_den
        self.fzero=fittype.fzero
        self.fzero_seq,self.fzero_den=fittype.fzero_seq,fittype.fzero_den

    def fit(self):

        zinputl=zed(np.concatenate((self.qsqp,self.qsq0)),self.tcut,self.t0)
        cinv=np.linalg.inv(self.cov)
        
        fplus=self.fplus
        fplusseq,fplusden=self.fplus_seq,self.fplus_den
        fzero=self.fzero
        fzeroseq,fzeroden=self.fzero_seq,self.fzero_den

        Yzmat=np.array([Yz(z,lbl,self.nbplus,self.nbzero,
                           fplusseq,fplusden,fzeroseq,fzeroden)
                       for z,lbl in zip(zinputl,self.ptlbls)]).transpose()
        zmax=zed(0,self.tcut,self.t0)
        fpm=fpmzmax(zmax,self.nbplus,self.nbzero,
                    fplusseq,fplusden,fzeroseq,fzeroden)
    
        Ycinv=np.dot(Yzmat,cinv)
        B=np.concatenate((2.0*np.dot(Ycinv,self.ff),((0.0,))))
        #M=2.0*np.dot(Ycinv,Yzmat.transpose())
        A=np.zeros((self.npars+1,self.npars+1))
        A[:self.npars,:self.npars]=2.0*np.dot(Ycinv,Yzmat.transpose())
        A[-1,:self.npars]=fpm
        A[:self.npars,-1]=fpm
        A=0.5*(A+A.transpose())
    
        # do the fit: inverting A is susceptible to roundoff, could be
        # better to use SVD
        Ainv=np.linalg.inv(A)
        fitpars=np.dot(Ainv,B)
        bplus=fitpars[:self.nbplus]
        bzero=fitpars[self.nbplus:-1]
        # Cov matrix of fitted params is 2*np.linalg.inv(A), which 
        # includes Lagrange multiplier for kinematic constraint.
        # Cov matrix of bplus and bzero is *np.linalg.inv(A)[:-1,:-1].
        bcov=2.0*Ainv[:-1,:-1]

        diffs=self.ff-np.dot(fitpars[:-1],Yzmat)
        chisq=np.dot(diffs,np.dot(cinv,diffs))
        npts=len(self.ff)
        ndof=npts-(self.npars-1) # -1 for f+(0)=f0(0) constraint
        pval=pvalue(ndof,chisq)
          
        # check constraint f+(0)=f0(0)
        fp,f0=fplus(zmax,bplus),fzero(zmax,bzero)
        constraintcheck=abs(2.0*(fp-f0)/(fp+f0))
        seq=np.array(fplusseq(zmax,self.nbplus))/fplusden(zmax)
        dfp=sqrt(np.dot(seq,np.dot(bcov[:self.nbplus,:self.nbplus],seq)))
        seq=np.array(fzeroseq(zmax,self.nbzero))/fzeroden(zmax)
        df0=sqrt(np.dot(seq,np.dot(bcov[self.nbplus:,self.nbplus:],seq)))
        
        results={}
        results['results']={}
        res=results['results']
        res['bplus']=bplus
        res['bzero']=bzero
        res['bcov']=bcov
        res['chisq']=chisq
        res['ndof']=ndof
        res['pval']=pval
        res['f+(0)']=fp
        res['df+(0)']=dfp
        res['f0(0)']=f0
        res['df0(0)']=df0
        res['constraintcheck']=constraintcheck
        results['info']=self.info
        results['Amat']=A
        results['fpm']=fpm
        
        return results
