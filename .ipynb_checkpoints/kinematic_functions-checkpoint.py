#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qsq <-> z transformations

Created on Mon Aug 16 20:32:03 2021

@author: jflynn
"""
from math import sqrt
import numpy as np

def zed(t,tp,t0):
  rt=np.sqrt(tp-t)
  r0=np.sqrt(tp-t0)
  return (rt-r0)/(rt+r0)
  
def qsq(z,tp,t0):
  return tp-(tp-t0)*((1.0+z)/(1.0-z))**2

def k(qsq,tp,tm,M):
  """k, the three-momentum of the outgoing pseudoscalar
  """
  #k=sqrt(M**4+m**4+qsq**2 -
  #            2.0*((M**2+m**2)*qsq + (M*m)**2))/(2.0*M)
  return sqrt((tm-qsq)*(tp-qsq))/(2.0*M)

def wl(qsq,ml):
    mlq=ml*ml/qsq
    return (1-mlq)**2*(1+0.5*mlq)

