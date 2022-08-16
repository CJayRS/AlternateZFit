#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:24:59 2019
Updated 20191218

@author: jflynn

Masses labelled PDG2018 are from pages linked to

http://pdg.lbl.gov/2019/tables/contents_tables_mesons.html

with citation: M. Tanabashi et al. (Particle Data Group),
Phys. Rev. D 98, 030001 (2018) and 2019 update.

Masses labelled PDG2016 are from PDG 2016 with citation:
C. Patrignani and Particle Data Group 2016 Chinese Phys. C40 100001
"""
from math import sqrt

"""
light mesons
"""
mpi=(2*0.13957061+0.1349770)/3 # PDG2018
mK=(0.493677+0.497611)/2       # PDG2018

"""
D mesons
"""
mD=(1.86965+1.86483)/2         # PDG2018
mDstar=(2.00685+2.01026)/2     # PDG2018
mDs=1.96834                    # PDG2018

"""
Ds mesons
"""
mDs=1.96835                    # PDG2021 online
mDsstar=2.1122                 # PDG2021 online
mDsstar0plus=2.3178            # PDG2021 online

"""
B mesons
"""
mB=(5.27933+5.27964)/2         # PDG2018
mBstar=5.32470                 # PDG2018

# mBstar0plus is the scalar mBstar0+ mass for f0 pole in Bs to K.
# From Bardeen et al in 2003, hep-ph/0305049, this mass is
# 5.63 GeV (5627 in list below). Potentially update this from a
# compilation of results in Cheng and Yu arXiv:1704.01208. Results
# from a variety of models are in the range 5.526 to 5.756, or 
# 1.8% below to 2.2% above 5.63. We stick with 5.63.

# list of masses from Cheng and Yu, dropping old Godfrey et al rel
# quark model and Cheng et al result with delta term
masslist=[5706,5720,5730,5749,5749,5756, # relativistic quark models
          5683, # nonrel quark model
          5627,5708.2, # HMChPT
          5530, # unitarised ChPT
          5637, # chiral loops
          5720, # QCD sumrules
          5526,5592,5592,5615,5675,5678,5718] # others
mBstar0plus_CY=sum(masslist)/len(masslist)
# round this to 2 dp when expressed in GeV
mBstar0plus_CY=round(mBstar0plus_CY/10)/100
minCY,maxCY=min(masslist)/1000,max(masslist)/1000

mBstar0plus=5.63

"""
Bs mesons

Fixed by the values used in RHQ tuning
"""
mBs=5.36682                    # PDG2016
mBsstar=5.4154                 # PDG2016
mBsbar=5.40326 # (3*mBsstar+mBs)/4
DeltamBs=0.0486                # PDG2016


"""
Bc mesons

Masses for poles in Bs to Ds use Bc states from Eichten-Quigg
1902.09735
"""
centroid=6315.5 # mev; centroid of Bc and Bc* masses
dm11S0=-40.5 # Bc
# 1- for f+
dm13S1=13.5
dm23S1=582
dm33D1=691
dm33S1=964
# 0+ for f0
dm23P0=377
dm33P0=789
dm43P0=1121

#mBc=6.2749 # PDGLive Nov 2019
mBc=(centroid+dm11S0)/1000
mBcstar=(centroid+dm13S1)/1000
mBc0=(centroid+dm23P0)/1000

"""
Parameters for outer functions
"""
etaBpi='3/2'
etaBsK=1
etaBD=2
etaBsDs=1
chi1minusBsK=6.03e-4 # Gev^{-2}
chi0plusBsK=1.48e-2
chi1minusBsDs=4.59e-4 # GeV^{-2}; this is chi-tilde
chi0plusBsDs=6.20e-3

chi1minusBpi=chi1minusBsK
chi0plusBpi=chi0plusBsK
chi1minusBD=chi1minusBsDs
chi0plusBD=chi0plusBsDs



if __name__=='__main__':
  def z(t,tp,t0):
      rt=sqrt(tp-t)
      r0=sqrt(tp-t0)
      return (rt-r0)/(rt+r0)
  
  def qsq(z,tp,t0):
      return tp-(tp-t0)*((1+z)/(1-z))**2
  
  def printzt(M,m,Mth,mth):
      """
      M, m are initial and final state masses
      Mth, mth are masses determining the cut threshold
      """
      kl=('2ptcl threshold','sqrt(qsqmax)',
          't threshold','t+','t- = qsqmax','t0','zmax','zmin')
      
      tth=(Mth+mth)**2
      tp=(M+m)**2
      tm=(M-m)**2
      t0=tth-sqrt(tth*(tth-tm))
      zmax=z(0,tth,t0)
      zmin=z(tm,tth,t0)
      
      odict={'2ptcl threshold':Mth+mth,'sqrt(qsqmax)':M-m,
             't threshold':tth,
             't+':tp,'t- = qsqmax':tm,'t0':t0,'zmax':zmax,'zmin':zmin}
  
      for k in kl:
          v=odict[k]
          print('{:15s}  {:f}'.format(k,v))
    
  def polecheck(m,mth):
    if m<mth:
      print('subthreshold: use it!')
    else:
      print('above threshold: not needed')

        
  print('B to pi')
  printzt(mB,mpi,mB,mpi)
  
  print('')
  
  print('Bs to K')
  printzt(mBs,mK,mB,mpi)
  
  print('')
  
  print('B to D')
  printzt(mB,mD,mB,mD)

  print('')
  
  print('Bs to Ds')
  printzt(mBs,mDs,mB,mD)
  
  print('')
  
  print('Bs and Bs* masses: fixed from those used in RHQ tuning\n')
  print('mBs:      {:f}'.format(mBs))
  print('mBsstar:  {:f}'.format(mBsstar))
  print('mBsbar:   {:f}  (3*mBsstar+mBs)/4'.format(mBsbar))
  print('DeltamBs: {:f}'.format(DeltamBs))
  
  print('')
  print('')
 
  print('Poles for B pi, Bs K and Lambda_b p')
  print('')
  print('threshold mB+mpi: {:f}'.format(mB+mpi))
  print('')
  print('1- pole for f+')
  print('mB*:     {:f}  {:f}'.format(mBstar,mBstar**2))
  polecheck(mBstar,mB+mpi)
  print('')
  print('0+ pole for f0')
  print('mB*(0+): {:f}  {:f}'.format(mBstar0plus,mBstar0plus**2))
  polecheck(mBstar0plus,mB+mpi)
  
  print('\nNote: mB*(0+) masses from compilation in Cheng and Yu cover')
  print('range: {:.3f} to {:.3f}'.format(minCY,maxCY))
  up,dn=[100*(m/mBstar0plus-1.0) for m in (minCY,maxCY)]
  print('variation {:.1f}% to {:.1f}% from {:.3f}'.format(up,dn,mBstar0plus))
  
  print('')
  print('')
  
  print('Poles for B D, Bs Ds and Lambda_b Lambda_c')
  print('')
  print('threshold mB+mD : {:f}'.format(mB+mD))
  print('')
  print('1- poles for f+')
  for lbl,arg in (('mBc* 13S1',dm13S1),
                  ('23S1',dm23S1),
                  ('33D1',dm33D1),
                  ('33S1',dm33S1)):
      mass=(centroid+arg)/1000
      print('{:9s}  {:f}  {:f}'.format(lbl,mass,mass**2))
      polecheck(mass,mB+mD)
      
  print('')
  
  print('0+ poles for f0')
  for lbl,arg in (('23P0',dm23P0),
                  ('33P0',dm33P0),
                  ('43P0',dm43P0)):
      mass=(centroid+arg)/1000
      print('{:9s}  {:f}  {:f}'.format(lbl,mass,mass**2))
      polecheck(mass,mB+mD)
  
  print('')
  print('')
  
  print('Delta for BsK chi-cont extrapolation (Delta = Mpole-MBs)')
  print('Delta+ for f+: {:.4f}'.format(mBstar-mBs))
  print('Delta0 for f0: {:.4f}'.format(mBstar0plus-mBs))

  print('')
  print('')

  print('Parameters for outer functions')
  print('B to pi')
  print('{:15s}  {:s}'.format('eta',etaBpi))
  print('{:15s}  {:.2e}'.format('chi1-',chi1minusBpi))
  print('{:15s}  {:.2e}'.format('chi0+',chi0plusBpi))
  print('Bs to K')
  print('{:15s}  {:d}'.format('eta',etaBsK))
  print('{:15s}  {:.2e}'.format('chi1-',chi1minusBsK))
  print('{:15s}  {:.2e}'.format('chi0+',chi0plusBsK))
  print('B to D')
  print('{:15s}  {:d}'.format('eta',etaBD))
  print('{:15s}  {:.2e}'.format('chi1-tilde',chi1minusBD))
  print('{:15s}  {:.2e}'.format('chi0+',chi0plusBD))
  print('Bs to Ds')
  print('{:15s}  {:d}'.format('eta',etaBsDs))
  print('{:15s}  {:.2e}'.format('chi1-tilde',chi1minusBsDs))
  print('{:15s}  {:.2e}'.format('chi0+',chi0plusBsDs))
  
