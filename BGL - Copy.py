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

# def phi(qsq,a,b,tcut,t0,tm,pars):
#     """Outer function
#
#        Form given in BGL97.
#    
#        (a,b)=(3,2) for f+, (1,1) for f0
#        K=48*pi for f+, K=16*pi/(tp*tm) for f0
#        eta=1 for Bs to K, Bs to Ds; eta=3/2 for B to pi; eta=2 for B to D
#     """
#     eta,K,chi=pars
#     rq=np.sqrt(tcut-qsq)
#     rm=sqrt(tcut-tm)
#     r0=sqrt(tcut-t0)
#     fac=sqrt(eta/(K*chi))
#     return fac*rq**((a+1)/2)*(rq+r0)*(rq+sqrt(tcut))**(-b-3)*(rq+rm)**(a/2)/sqrt(r0)

def phi(qsq,a,b,tl,pars):
    """Outer function

       Form given in BGL97.
    
       (a,b)=(3,2) for f+, (1,1) for f0
       K=48*pi for f+, K=16*pi/(tp*tm) for f0
       eta=1 for Bs to K, Bs to Ds; eta=3/2 for B to pi; eta=2 for B to D
    """
    tcut,t0,tp,tm=tl
    eta,K,chi=pars
    rq=np.sqrt(tcut-qsq)
    rm=sqrt(tcut-tm)
    r0=sqrt(tcut-t0)
    fac=sqrt(eta/(K*chi))
    return fac*rq**((a+1)/2)*(rq+r0)*(rq+sqrt(tcut))**(-b-3)*(rq+rm)**(a/2)/sqrt(r0)

# def phi(qsq,a,b,tl,pars):
#     """Outer function

#        This is like the form given in BGL97, but also allowing tcut and t+
#        to differ.
    
#        (a,b)=(3,2) for f+, (1,1) for f0
#        K=48*pi for f+, K=16*pi/(tp*tm) for f0
#        eta=1 for Bs to K, Bs to Ds; eta=3/2 for B to pi; eta=2 for B to D
#     """
#     tcut,t0,tp,tm=tl
#     eta,K,chi=pars
#     rq=np.sqrt(tcut-qsq)
#     rm=sqrt(tcut-tm)
#     r0=sqrt(tcut-t0)
#     rpq=np.sqrt(tp-qsq)
#     fac=sqrt(rq*eta/(K*chi))*(rpq*(rq+rm))**(a/2)
#     return fac*(rq+r0)*(rq+sqrt(tcut))**(-b-3)/sqrt(r0)

def phizero_bmrv(qsq,tl,pars):
    """Outer function

       Based on outer fns given in BMRV 2022.

       K=48*pi for f+, K=16*pi/(tp*tm) for f0
    """
    tcut,t0,tp,tm=tl
    eta,K,chi=pars
    rq=np.sqrt(tcut-qsq)
    #rm=sqrt(tcut-tm)
    r0=sqrt(tcut-t0)
    rpq=np.sqrt(tp-qsq)
    rmq=np.sqrt(tm-qsq)
    fac=sqrt(eta/(K*chi))
    fac*=np.sqrt(rpq*rmq*rq*np.abs(rq-r0)*(rq+r0)/r0)
    return fac/qsq**2

def phiplus_bmrv(qsq,tl,pars):
    """Outer function

       Based on outer fns given in BMRV 2022.

       K=48*pi for f+, K=16*pi/(tp*tm) for f0
    """
    tcut,t0,tp,tm=tl
    eta,K,chi=pars
    rq=np.sqrt(tcut-qsq)
    #rm=sqrt(tcut-tm)
    r0=sqrt(tcut-t0)
    rpq=np.sqrt(tp-qsq)
    rmq=np.sqrt(tm-qsq)
    fac=sqrt(eta/(K*chi))
    fac*=rpq*rmq*np.sqrt(rpq*rmq*rq*np.abs(rq-r0)*(rq+r0)/r0)
    return fac/qsq**(5/2)


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

     The form factors defined here are generically

     f(z) = \sum_n b_n z^n / (B(z)phi(z))

     which is the standard convention

  """
  def __init__(self,fpluspoles=[2.0],fzeropoles=[2.0],
                    chiplus=1.0,chizero=1.0,
                    tcut=1.0,t0=0.5,
                    tp=1.0,tm=0.75,
                    eta=1.0,
                    extralabel=''):
    self.fittypelbl='BGL'
    self.fitlbl='BGL'+extralabel
    self.chiplus=chiplus
    self.chizero=chizero
    self.fpluspoles=fpluspoles
    self.fzeropoles=fzeropoles
    self.tcut=tcut
    self.t0=t0
    self.tp=tp
    self.tm=tm
    self.eta=eta
    self.tl=tcut,t0,tp,tm
    self.Bphiplus0=phi(0.0,3,2,self.tl,(self.eta,48.0*pi,self.chiplus))
    self.Bphiplus0*=blaschke(0.0,self.tcut,self.fpluspoles)
    self.Bphizero0=phi(0.0,1,1,self.tl,(self.eta,16.0*pi/(self.tp*self.tm),self.chizero))
    self.Bphizero0*=blaschke(0.0,self.tcut,self.fzeropoles)
  def __str__(self):
    return 'BGL fit functions'
  def fplus(self,z,bplus):
    t=qsq(z,self.tcut,self.t0)
    seq=[b*z**k for k,b in enumerate(bplus)]
    den=phi(t,3,2,self.tl,(self.eta,48.0*pi,self.chiplus))
    den*=blaschke(t,self.tcut,self.fpluspoles)
    return sum(seq)/den
  def fplus_seq(self,z,nbplus):
    return [z**k for k in range(nbplus)]
  def fplus_den(self,z):
    t=qsq(z,self.tcut,self.t0)
    den=phi(t,3,2,self.tl,(self.eta,48.0*pi,self.chiplus))
    den*=blaschke(t,self.tcut,self.fpluspoles)
    return den
  def fzero(self,z,bzero):
    t=qsq(z,self.tcut,self.t0)
    seq=[b*z**k for k,b in enumerate(bzero)]
    den=phi(t,1,1,self.tl,(self.eta,16.0*pi/(self.tp*self.tm),self.chizero))
    den*=blaschke(t,self.tcut,self.fzeropoles)
    return sum(seq)/den
  def fzero_seq(self,z,nbzero):
    return [z**k for k in range(nbzero)]
  def fzero_den(self,z):
    t=qsq(z,self.tcut,self.t0)
    den=phi(t,1,1,self.tl,(self.eta,16.0*pi/(self.tp*self.tm),self.chizero))
    den*=blaschke(t,self.tcut,self.fzeropoles)
    return den

  def info(self):
    out={}
    out['fittype']=self.fittypelbl
    out['fitlbl']=self.fitlbl
    out['tcut']=self.tcut
    out['t0']=self.t0
    out['t+']=self.tp
    out['t-']=self.tm
    out['chiplus']=self.chiplus
    out['chizero']=self.chizero
    out['eta']=self.eta
    out['Bphiplus(0)']=self.Bphiplus0
    out['Bphizero(0)']=self.Bphizero0
    out['f+poles']=list(self.fpluspoles)
    out['f0poles']=list(self.fzeropoles)
    return out


# class BGL:
#   """BGL parametrisation for z-fits

#      The form factors defined here are generically

#      f(z) = \sum_n b_n z^n / (B(z)phi(z))

#      which is the standard convention

#   """
#   def __init__(self,fpluspoles=[2.0],fzeropoles=[2.0],
#                     chiplus=1.0,chizero=1.0,
#                     tcut=1.0,t0=0.5,
#                     tp=1.0,tm=0.75,
#                     eta=1.0,
#                     extralabel=''):
#     self.fittypelbl='BGL'
#     self.fitlbl='BGL'+extralabel
#     self.chiplus=chiplus
#     self.chizero=chizero
#     self.fpluspoles=fpluspoles
#     self.fzeropoles=fzeropoles
#     self.tcut=tcut
#     self.t0=t0
#     self.tp=tp
#     self.tm=tm
#     self.eta=eta
#     self.Bphiplus0=phi(0.0,3,2,self.tcut,self.t0,self.tm,(self.eta,48.0*pi,self.chiplus))
#     self.Bphiplus0*=blaschke(0.0,self.tcut,self.fpluspoles)
#     self.Bphizero0=phi(0.0,1,1,self.tcut,self.t0,self.tm,(self.eta,16.0*pi/(self.tp*self.tm),self.chizero))
#     self.Bphizero0*=blaschke(0.0,self.tcut,self.fzeropoles)
#   def __str__(self):
#     return 'BGL fit functions'
#   def fplus(self,z,bplus):
#     t=qsq(z,self.tcut,self.t0)
#     seq=[b*z**k for k,b in enumerate(bplus)]
#     den=phi(t,3,2,self.tcut,self.t0,self.tm,(self.eta,48.0*pi,self.chiplus))
#     den*=blaschke(t,self.tcut,self.fpluspoles)
#     return sum(seq)/den
#   def fplus_seq(self,z,nbplus):
#     return [z**k for k in range(nbplus)]
#   def fplus_den(self,z):
#     t=qsq(z,self.tcut,self.t0)
#     den=phi(t,3,2,self.tcut,self.t0,self.tm,(self.eta,48.0*pi,self.chiplus))
#     den*=blaschke(t,self.tcut,self.fpluspoles)
#     return den
#   def fzero(self,z,bzero):
#     t=qsq(z,self.tcut,self.t0)
#     seq=[b*z**k for k,b in enumerate(bzero)]
#     den=phi(t,1,1,self.tcut,self.t0,self.tm,
#             (self.eta,16.0*pi/(self.tp*self.tm),self.chizero))
#     den*=blaschke(t,self.tcut,self.fzeropoles)
#     return sum(seq)/den
#   def fzero_seq(self,z,nbzero):
#     return [z**k for k in range(nbzero)]
#   def fzero_den(self,z):
#     t=qsq(z,self.tcut,self.t0)
#     den=phi(t,1,1,self.tcut,self.t0,self.tm,
#             (self.eta,16.0*pi/(self.tp*self.tm),self.chizero))
#     den*=blaschke(t,self.tcut,self.fzeropoles)
#     return den

#   def info(self):
#     out={}
#     out['fittype']=self.fittypelbl
#     out['fitlbl']=self.fitlbl
#     out['tcut']=self.tcut
#     out['t0']=self.t0
#     out['t+']=self.tp
#     out['t-']=self.tm
#     out['chiplus']=self.chiplus
#     out['chizero']=self.chizero
#     out['eta']=self.eta
#     out['Bphiplus(0)']=self.Bphiplus0
#     out['Bphizero(0)']=self.Bphizero0
#     out['f+poles']=list(self.fpluspoles)
#     out['f0poles']=list(self.fzeropoles)
#     return out


class BGLdiffnorm:
  """BGL parametrisation for z-fits

     The form factors defined here are generically

     f(z) = B(zmax)phi(zmax) \sum_n b_n z^n / (B(z)phi(z))

     This means that f(zmax) = \sum_n b_n z^n. To get b_n's in the
     standard convention use
     
     b_n(std) = b_n B(zmax)phi(zmax)

  """
  def __init__(self,fpluspoles=[2.0],fzeropoles=[2.0],
                    chiplus=1.0,chizero=1.0,
                    tcut=1.0,t0=0.5,
                    tp=1.0,tm=0.75,
                    eta=1.0,
                    extralabel=''):
    self.fittypelbl='BGL'
    self.fitlbl='BGL'+extralabel
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

  def info(self):
    out={}
    out['fittype']=self.fittypelbl
    out['fitlbl']=self.fitlbl
    out['tcut']=self.tcut
    out['t0']=self.t0
    out['t+']=self.tp
    out['t-']=self.tm
    out['chiplus']=self.chiplus
    out['chizero']=self.chizero
    out['eta']=self.eta
    out['Bphiplus(0)']=self.Bphiplus0
    out['Bphizero(0)']=self.Bphizero0
    out['f+poles']=list(self.fpluspoles)
    out['f0poles']=list(self.fzeropoles)
    return out
