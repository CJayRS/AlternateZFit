#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:00:44 2021
updated Dec 2021 for Bs to K

@author: jflynn

Dispersive bounds for Bs -> K l nu semileptonic decay

Using our RCB/UKQCD data
"""


import h5py as h5
from math import sqrt,pi
import numpy as np
import matplotlib.pyplot as plt
from plot_settings import plotparams
from kinematic_functions import zed#qsq,k,wl
from BGL import phi#,blaschke
#from scipy.optimize import minimize_scalar
import time

from numba import jit

plt.rcParams.update(plotparams)


@jit(nopython=True)
def ggzprod(zi,zj):
    """
    inner product <g_s|g_t>
    """
    return 1.0/(1.0-zi*zj)

@jit(nopython=True)
def Gmatrix(zl):
    """
    G matrix for values of z in list zl
    """
    npts=len(zl)
    G=np.zeros((npts,npts))
    for i,zi in enumerate(zl):
        G[i,i]=ggzprod(zi,zi)
        for j in range(i+1,npts):
            G[i,j]=ggzprod(zi,zl[j])
            G[j,i]=G[i,j]
    return G

# def phizero(z,tp,tm,eta): # from 2105.02497 for cf to my defns
#     rho0=sqrt(tp/(tp-tm))
#     fac=sqrt(eta*tp*tm/(2.0*pi))/(tp-tm)
#     return fac*(1.0+z)/((rho0+(1.0+z)/(1.0-z))**4*(1.0-z)**(2.5))

# def phiplus(z,tp,tm,eta): # from 2105.02497 for cf to my defns
#     rho0=sqrt(tp/(tp-tm))
#     fac=sqrt(2.0*eta/(3.0*pi*(tp-tm)))
#     return fac*(1.0+z)**2/((rho0+(1.0+z)/(1.0-z))**5*(1.0-z)**(4.5))


# light mesons
mpi=(2*0.13957061+0.1349770)/3 # PDG2018
mK=(0.493677+0.497611)/2       # PDG2018

# B mesons
mB=(5.27933+5.27964)/2
mBstar= 5.32470
mBs=5.36682
mBstar0plus=5.63

tp=(mBs+mK)**2
tm=(mBs-mK)**2 # = qsqmax
tcut=(mB+mpi)**2
t0=tcut-sqrt(tcut*(tcut-tm))# D mesons

etaBsK=1
chi1minusBsK=6.03e-4
chi0plusBsK=1.48e-2
fpluspoles=np.array([mBstar])
fzeropoles=np.array([])

zmax=zed(0,tcut,t0)

# z-values for pole positions
zpluspole=zed(mBstar**2,tcut,t0)
#zzeropole=zed(mBstar0plus**2,tcut,t0) # mBstar0plus is above threshold


nplus=2 # number of input values for f+
nzero=3 # number of input values for f0
qsqmin='17.50' # (17.50,18.,00,18.50,19.00)
path=''
"""
   In the h5 file, the inputs points are first for f+ then for f0
"""
with h5.File(path+'zfit_data_BstoK.h5', 'r') as f:
    gp=f.get('BstoK_refdata_qsqmin_{:s}_Np{:d}_Nz{:d}'.format(qsqmin,nplus,nzero))
    qsqinputl=np.array(gp['qsqref'])
    bskpts=np.array(gp['central'])
    bskcov=np.array(gp['tot_cov'])

print("qsqinputl = ",qsqinputl)

zinputl=zed(qsqinputl,tcut,t0)

npts=len(bskpts)
bskptslbls=[char for char in nplus*'+'+nzero*'0']

dbsk=np.sqrt(bskcov.diagonal())
bskcov=0.5*(bskcov + np.transpose(bskcov))

# input f+ and f0 values
tpin=qsqinputl[:nplus]
zpin=zed(tpin,tcut,t0)
fpin=bskpts[:nplus]
dfpin=dbsk[:nplus]
tzin=qsqinputl[nplus:]
zzin=zed(tzin,tcut,t0)
fzin=bskpts[nplus:]
dfzin=dbsk[nplus:]
nppts=len(zpin)
nzpts=len(zzin)

print('Inputs: first for f+ then f0')
with np.printoptions(precision=4,floatmode='fixed',
                      formatter={'float_kind':lambda x: '{:6.4f}'.format(x)}):
    print('t=q^2',tpin,tzin)
    print('z(t) ',zpin,zzin)
    print('f+/0 ',fpin,fzin)
    print('df+/0',dfpin,dfzin)
print('')

def compute_bounds(tl,fpinputs,fzinputs):
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
        #print('unitarity failed for f+ inputs: ',chimchibarp)
        return np.array([0,0,0,0])

    dil=[np.prod((1.0-z*np.delete(zzin,i))/(z-np.delete(zzin,i))) for i,z in enumerate(zzin)]
    Fzdl=Fz*dil*(1.0-zzin**2)
    Gz=Gmatrix(zzin)
    chimchibarz=chiz-np.dot(Fzdl,np.dot(Gz,Fzdl)) # should be positive
    if chimchibarz<0.0:
        #print('unitarity failed for f0 inputs: ',chimchibarz)
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

        boundsl[i]=fp,dfp,fz,dfz

    if boundsl.shape==(1,4):
        return boundsl[0]
    else:
        return boundsl


fpinputs=(tpin,fpin,chi1minusBsK,zpluspole)
fzinputs=(tzin,fzin,chi0plusBsK,zpluspole) # !!! zpluspole just a dummy here

fp,dfp,fz,dfz=compute_bounds(0.0,fpinputs,fzinputs)
print('bounds at t = qsq = 0')
print('fplus_high {:11.9f},  fplus_low {:11.9f}'.format(fp+dfp,fp-dfp))
print('fzero_high {:11.9f},  fzero_low {:11.9f}'.format(fz+dfz,fz-dfz))
print('')



# bootstrapping for f+(0) = f0(0)

start=time.time()

nboot=2000
ffsamp=np.random.multivariate_normal(bskpts,bskcov,size=nboot)
chi1m=chi1minusBsK
chi0p=chi0plusBsK


bounds0l=[] # to accumulate bounds for f+(0) = f0(0)
bounds0indices=[] # keep list of indices of samples which give a bound
nboottilde=0
nbootstar=0
for i,samp in enumerate(ffsamp):
    fpi=samp[:nplus]
    fzi=samp[nplus:]
    #chi1m=chi1minus
    #chi0p=chi0plus

    fpinputs=(tpin,fpi,chi1m,zpluspole)
    fzinputs=(tzin,fzi,chi0p,zpluspole)

    shark=compute_bounds(0.0,fpinputs,fzinputs)

    if shark.all()==False: # inputs don't satisfy unitarity
        pass
    else:
        nboottilde+=1
        fp,dfp,fz,dfz=shark
        if fp+dfp<=fz-dfz or fz+dfz<=fp-dfp:
            pass
        else: # if pass both checks record f(0) bounds and sample index
            nbootstar+=1
            bounds0l.append((fp,dfp,fz,dfz))
            bounds0indices.append(i)

print('Nboot      = {:d} samples to generate f+(0) = f0(0)'.format(nboot))
print('Nboottilde = {:d} ({:.1f} %) passed unitarity check'.format(nboottilde,
                                                                    100.0*nboottilde/nboot))
print('Nbootstar  = {:d} ({:.1f} %) passed kinematic constraint'.format(nbootstar,
                                                                        100.0*nbootstar/nboot))
print('')



# now do inner bootstrap on each of the nbootstar events

# values of t = qsq where we want to evaluate bounds
toutl=np.linspace(0.0,tm,10)
ntout=len(toutl)

# for inner bootstrap
n0=2
print('N0         = {:d} for inner bootstrap'.format(n0))
print('')

results=np.zeros((ntout,4,nbootstar))
for i,bounds0 in enumerate(bounds0l):

    fp,dfp,fz,dfz=bounds0
    fpup,fplo,fzup,fzlo=fp+dfp,fp-dfp,fz+dfz,fz-dfz
    fstarup=min(fp+dfp,fz+dfz)
    fstarlo=max(fp-dfp,fz-dfz)
    index=bounds0indices[i]
    fpi=ffsamp[index,:nplus]
    fzi=ffsamp[index,nplus:]
    tpin_inner=np.concatenate(([0.0],tpin))
    tzin_inner=np.concatenate(([0.0],tzin))
    # sample uniformly
    #fstarsamp=np.random.uniform(fstarlo,fstarup,size=n0)
    # or use n0 equally-spaced values (n0=2 gives just the endpoints)
    jot=1.0e-9
    fstarsamp=np.linspace(fstarlo+jot,fstarup-jot,n0)
    boundsinner=np.zeros((n0,ntout,4))
    for j,fstar in enumerate(fstarsamp):
        fpinner=np.concatenate(([fstar],fpi))
        fzinner=np.concatenate(([fstar],fzi))
        fpinputs_inner=(tpin_inner,fpinner,chi1m,zpluspole)
        fzinputs_inner=(tzin_inner,fzinner,chi0p,zpluspole)

        boundsinner[j]=compute_bounds(toutl,fpinputs_inner,fzinputs_inner)

    fpbarup=np.max(boundsinner[:,:,0]+boundsinner[:,:,1],axis=0)
    fpbarlo=np.min(boundsinner[:,:,0]-boundsinner[:,:,1],axis=0)
    fzbarup=np.max(boundsinner[:,:,2]+boundsinner[:,:,3],axis=0)
    fzbarlo=np.min(boundsinner[:,:,2]-boundsinner[:,:,3],axis=0)
    results[:,0,i]=fpbarup
    results[:,1,i]=fpbarlo
    results[:,2,i]=fzbarup
    results[:,3,i]=fzbarlo

stop=time.time()
dt=stop-start
print('Time taken: {:d}m{:d}s'.format(int(dt//60),round(dt-60*(dt//60))))
print('')

#print(toutl)
raven=results.mean(axis=2)
fplus=(raven[:,0]+raven[:,1])/2
fzero=(raven[:,2]+raven[:,3])/2
fplusup,fpluslo=raven[:,0],raven[:,1]
fzeroup,fzerolo=raven[:,2],raven[:,3]
covplus=np.array([np.cov(arg[:2],ddof=1) for arg in results])
covzero=np.array([np.cov(arg[2:],ddof=1) for arg in results])
sigplus=np.sqrt((fplusup-fpluslo)**2/12+(covplus[:,0,0]+covplus[:,1,1]+covplus[:,0,1])/3)
sigzero=np.sqrt((fzeroup-fzerolo)**2/12+(covzero[:,0,0]+covzero[:,1,1]+covzero[:,0,1])/3)
print('t        f+bds       f0bds')
for t,fp,fz,dfp,dfz in zip(toutl,(fplusup+fpluslo)/2.0,(fzeroup+fzerolo)/2.0,
                           np.round(10000*sigplus),np.round(10000*sigzero)):
    print('{:7.4f}  {:.4f}({:4.0f})  {:.4f}({:4.0f})'.format(t,fp,dfp,fz,dfz))
print('')

savestr='BSK_{:d}_{:d}_{:s}_disp_bds_naive_nbootstar={:d}'.format(nplus,nzero,qsqmin,nbootstar)
#savestr='BSK_{:d}_{:d}_{:s}_disp_bds_N0=200_nbootstar={:d}'.format(nplus,nzero,qsqmin,nbootstar)
# np.savetxt(savestr+'.dat',
#            np.array((toutl,(fplusup+fpluslo)/2.0,sigplus,(fzeroup+fzerolo)/2.0,sigzero)),
#            header=savestr)
