#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGL functions for z-fits

Created on Mon Aug 16 19:26:27 2021

@author: jflynn
"""
from math import sqrt, pi
import numpy as np
from kinematic_functions import qsq,zed

def phi(qsq,a,b,tcut,t0,tm,pars):
    """Outer function
    
      (a,b)=(3,2) for f+, (1,1) for f0
      K=48*pi for f+, K=16*pi/(tp*tm) for f0
      eta=1 for Bs to K, Bs to Ds; eta=3/2 for B to pi; eta=2 for B to D
    """
    eta,K,chi=pars
    rq=np.sqrt(tcut-qsq)
    rm=sqrt(tcut-tm)
    r0=sqrt(tcut-t0)
    fac=sqrt(eta/(K*chi))
    return fac*rq**((a+1)/2)*(rq+r0)*(rq+sqrt(tcut))**(-b-3)*(rq+rm)**(a/2)/sqrt(r0)

def blaschke(qsq,tcut,ml):
    """Blaschke factor for np array of pole masses ml

       ml has to be a numpy array because we check its size.
       
       If qsq is an array of length nq and ml has length nm>0, then
       this calculates an array of shape nm by nq, and then takes
       products over the nm values at each q-squared, leaving an
       array of length nq.
    """
    if ml.size==0:
      return 1.0
    else:
      return np.array([zed(qsq,tcut,m**2) for m in ml]).prod(axis=0)

class BGL:
  """BGL parametrisation for z-fits

  """
  def __init__(self,fpluspoles=[2.0],fzeropoles=[2.0],
                    chiplus=1.0,chizero=1.0,
                    tcut=1.0,t0=0.5,
                    tp=1.0,tm=0.75,
                    eta=1.0):
    self.chiplus=chiplus
    self.chizero=chizero
    self.fpluspoles=fpluspoles
    self.fzeropoles=fzeropoles
    self.tcut=tcut
    self.t0=t0
    self.tp=tp
    self.tm=tm
    self.eta=eta
    self.Bphiplus0=phi(0.0,3,2,self.tcut,self.t0,self.tm,(self.eta,48.0*pi,self.chiplus))
    self.Bphiplus0*=blaschke(0.0,self.tcut,self.fpluspoles)
    self.Bphizero0=phi(0.0,1,1,self.tcut,self.t0,self.tm,(self.eta,16.0*pi/(self.tp*self.tm),self.chizero))
    self.Bphizero0*=blaschke(0.0,self.tcut,self.fzeropoles)
  def __str__(self):
    return 'BGL fit functions'
  def fplus(self,z,bplus):
    t=qsq(z,self.tcut,self.t0)
    seq=[b*z**k for k,b in enumerate(bplus)]
    den=phi(t,3,2,self.tcut,self.t0,self.tm,(self.eta,48.0*pi,self.chiplus))
    den*=blaschke(t,self.tcut,self.fpluspoles)
    return sum(seq)*self.Bphiplus0/den
  def fplus_seq(self,z,nbplus):
    return [z**k for k in range(nbplus)]
  def fplus_den(self,z):
    t=qsq(z,self.tcut,self.t0)
    den=phi(t,3,2,self.tcut,self.t0,self.tm,(self.eta,48.0*pi,self.chiplus))
    den*=blaschke(t,self.tcut,self.fpluspoles)
    return den/self.Bphiplus0
  def fzero(self,z,bzero):
    t=qsq(z,self.tcut,self.t0)
    seq=[b*z**k for k,b in enumerate(bzero)]
    den=phi(t,1,1,self.tcut,self.t0,self.tm,(self.eta,16.0*pi/(self.tp*self.tm),self.chizero))
    den*=blaschke(t,self.tcut,self.fzeropoles)
    return sum(seq)*self.Bphizero0/den
  def fzero_seq(self,z,nbzero):
    return [z**k for k in range(nbzero)]
  def fzero_den(self,z):
    t=qsq(z,self.tcut,self.t0)
    den=phi(t,1,1,self.tcut,self.t0,self.tm,(self.eta,16.0*pi/(self.tp*self.tm),self.chizero))
    den*=blaschke(t,self.tcut,self.fzeropoles)
    return den/self.Bphizero0
