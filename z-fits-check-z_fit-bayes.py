#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 16:11:01 2022

Check running different kinds of z-fit (BCL, BGL).

@author: jflynn
"""

import time
#import h5py
import numpy as np
from math import sqrt
from kinematic_functions import zed,qsq #zed,qsq,k,wl
#from h5utils import h5writedict

import matplotlib.pyplot as plt
from plot_settings import plotparams
#plt.rcParams.update(plotparams)

#from BCL import BCL
from BGL import BGL
import masses

from chi_ctm_fit_results import HMChiPTfit
from z_fit_bayes import Yzc

plotdir='zbayesout'
savefigs = True

# add a flat systematic to the synthetic data points?
#add_flat_systematic=True

bstok=HMChiPTfit('Bs','K',qsqmin=17.5)
decaylbl=bstok.decay
qsqmin,qsqmax=bstok.qsqmin,bstok.qsqmax

# masses
mBs=bstok.Min
mK=bstok.Mout
Min,Mout=mBs,mK
mB=masses.mB
mpi=masses.mpi
mBstar=masses.mBstar
mBstar0plus=masses.mBstar0plus

tp=(mBs+mK)**2
tm=(mBs-mK)**2 # = qsqmax
tcut=(mB+mpi)**2
t0=tcut-sqrt(tcut*(tcut-tm))

zmax=zed(0,tcut,t0)

etaBsK=1
chi1minusBsK=6.03e-4
chi0plusBsK=1.48e-2
fpluspoles=np.array([mBstar])
fzeropoles=np.array([])

# bcl=BCL(fpluspoles=fpluspoles,fzeropoles=fzeropoles,
#         tcut=tcut,t0=t0)

# bclf0p=BCL(fpluspoles=fpluspoles,fzeropoles=np.array([mBstar0plus]),
#            tcut=tcut,t0=t0,extralabel='f0pole')

bgl=BGL(fpluspoles=fpluspoles,fzeropoles=fzeropoles,
        chiplus=chi1minusBsK,chizero=chi0plusBsK,
        tcut=tcut,t0=t0,tp=tp,tm=tm,eta=etaBsK)

#f=h5py.File(decaylbl+'_zfit_check_results.h5','w')

for variation in ('chosen',):#bstok.variations.keys():
    print('\nInput variation: '+variation)
    nplus,nzero=[len(bstok.variations[variation][arg]) for arg in (0,1)]
    ptlbls=[char for char in nplus*'p'+nzero*'0']
    
    qsqp=(17.5,23.7)#np.linspace(qsqmin,qsqmax,nplus)
    qsq0=(17.5,20.6,23.7)#np.linspace(qsqmin,qsqmax,nzero)
    qsqinputl=np.concatenate((qsqp,qsq0))
    zinputl=zed(qsqinputl,tcut,t0)
    
    inpts=np.array((0.993,3.156,0.490,0.647,0.874))
    incov=np.array([[0.00111562,0.00168573,0.00029496,0.00027699,0.00034324],
                    [0.00168573,0.00460525,0.00040696,0.00059994,0.00084741],
                    [0.00029496,0.00040696,0.00017755,0.0001194 ,0.00013803],
                    [0.00027699,0.00059994,0.0001194 ,0.00014451,0.00017785],
                    [0.00034324,0.00084741,0.00013803,0.00017785,0.00024815]])
    dinpts=np.sqrt(np.diag(incov))
    
    #inpts,incov=bstok.synthetic_points(qsqp,qsq0,variation)
    
    # if add_flat_systematic:
    #     #incov+=bstok.flat_covariance(inpts[:nplus],inpts[nplus:])
    #     incov+=bstok.flat_covariance(inpts[:nplus],inpts[nplus:])
    

    # print('Input form factor values and their covariance matrix')
    # print(inpts)
    # print(incov)

    priorp=(0.01,-0.02,-0.4,0.0,0.0,0.0,0.0,0.0) # ap0,ap1,ap2,...
    prior0=(-0.3,0.7,0.0,0.0,0.0,0.0,0.0,0.0)    # a01,a02,  ...
    priorvarp=np.array((0.02,0.04,0.8,1.0,1.0,1.0,1.0,1.0))**2
    priorvar0=np.array((0.6,1.4,1.0,1.0,1.0,1.0,1.0,1.0))**2
    # 'wide priors'
    #priorp=(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0) # ap0,ap1,ap2,...
    #prior0=(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)    # a01,a02,  ...
    #priorvarp=5.0*np.array((1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))**2
    #priorvar0=5.0*np.array((1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))**2

    fit,fittype,fitlbl=(bgl,'BGL','BGL')
    fpluszplotlbl='$[B(q^2)\phi(q^2)/B(0)\phi(0)]\, f^+$'
    fzerozplotlbl='$[B(q^2)\phi(q^2)/B(0)\phi(0)]\, f^0$'
    
    resdict={}

    #for fit,fittype in ((bgl,'BGL'),):#(bcl,bclf0p,bgl): 
    for nbplus,nbzero in ((2,3),(4,4)):#,(3,3),(3,4),(2,4),(4,5),(5,5),
                          #(5,6),(6,6),(6,7),(7,7)): 
        outlbl='_BsK_{:d}_{:d}_Kp={:d}_K0={:d}'.format(nplus,nzero,
                                                       nbplus,nbzero)
        outputname=fitlbl+outlbl
        
        resdict[(nbplus,nbzero)]={}
        
        fplus=fit.fplus
        fplusseq,fplusden=fit.fplus_seq,fit.fplus_den
        fzero=fit.fzero
        fzeroseq,fzeroden=fit.fzero_seq,fit.fzero_den

        a0=np.concatenate((priorp[:nbplus],prior0[:nbzero-1]))
        vara0=np.concatenate((priorvarp[:nbplus],priorvar0[:nbzero-1]))
        
        Zmat=np.array([Yzc(z,zmax,lbl,nbplus,nbzero,
                            fplusseq,fplusden,fzeroseq,fzeroden)
                       for z,lbl in zip(zinputl,ptlbls)])
        
        Cfinv=np.linalg.inv(incov)
        Ca0inv=np.diag(1.0/vara0)
        # no prior information
        #Ca0inv=np.zeros((nbplus+nbzero-1,nbplus+nbzero-1))

        Ctildeinv=np.dot(Zmat.T,np.dot(Cfinv,Zmat))+Ca0inv
        Ctilde=np.linalg.inv(Ctildeinv)
        atilde=np.dot(Zmat.T,np.dot(Cfinv,inpts))+np.dot(Ca0inv,a0)
        atilde=np.dot(Ctilde,atilde)
        
        nsamp=1000
        nchunk=1000
        asamp=np.zeros((nsamp,nbplus+nbzero))
        start=time.time()
        count=0
        ntries=0
        print("Generating nsamp...")
        while count<nsamp:
            atries=np.random.multivariate_normal(atilde,Ctilde,size=nchunk)
            for al in atries:
                ntries+=1
                apl=al[:nbplus]
                a00=np.dot(apl,fplusseq(zmax,nbplus))*fzeroden(zmax)/fplusden(zmax)
                a00-=np.dot(al[nbplus:],fzeroseq(zmax,nbzero)[1:])
                a0l=np.concatenate(([a00],al[nbplus:]))
                if np.dot(a0l,a0l)<=1 and np.dot(apl,apl)<=1.0:
                    asamp[count]=np.concatenate((apl,a0l))
                    #print(count)
                    count+=1
                    if count==nsamp:
                        break
        nsamp=len(asamp)
        print("Finished generating nsamp.")
        end=time.time()
        #s1='  '+fit.fitlbl+' z-bayes, '+'Kp={:d} K0={:d}'.format(nbplus,nbzero)
        print('  Took {:.2f}s'.format(end-start)) #s1+
        
        # clunky way to insert a00 values as a new column in asamp
        # so that each row is [a+0,...,a+(nbplus-1),a00,...,a0(nbzero-1)]
        #il=range(nbplus,nbplus+nbzero)
        #asamp=np.hstack((asamp,[[arg] for arg in a00]))
        #asamp[:,il]=asamp[:,np.roll(il,1)]
        
        amu=np.mean(asamp,axis=0)
        acov=np.cov(asamp,rowvar=False)
        
        print(amu[:nbplus])
        print(amu[nbplus:])
                
        plotpoints=30
        zl=np.linspace(-zmax,zmax,plotpoints)
        qsql=qsq(zl,tcut,t0)

        fpl=fplus(zl,amu[:nbplus])
        fpdenl=fplusden(zl)
        f0l=fzero(zl,amu[nbplus:])
        f0denl=fzeroden(zl)
          
        dfpl,df0l=np.zeros(len(zl)),np.zeros(len(zl))
        for i,z in enumerate(zl):
          seqp=np.array(fplusseq(z,nbplus))/fplusden(z)
          seq0=np.array(fzeroseq(z,nbzero))/fzeroden(z)
          dfpl[i]=sqrt(np.dot(seqp,np.dot(acov[:nbplus,:nbplus],seqp)))
          df0l[i]=sqrt(np.dot(seq0,np.dot(acov[nbplus:,nbplus:],seq0)))

        resdict[(nbplus,nbzero)]['f(0)']=fpl[-1]
        resdict[(nbplus,nbzero)]['ap']=amu[:nbplus]
        resdict[(nbplus,nbzero)]['dap']=np.sqrt(acov.diagonal()[:nbplus])
        resdict[(nbplus,nbzero)]['a0']=amu[nbplus:]
        resdict[(nbplus,nbzero)]['da0']=np.sqrt(acov.diagonal()[nbplus:])
        resdict[(nbplus,nbzero)]['f(0)']=fpl[-1]
        resdict[(nbplus,nbzero)]['df(0)']=dfpl[-1]
        resdict[(nbplus,nbzero)]['ntries']=ntries
        resdict[(nbplus,nbzero)]['nsamp']=len(asamp)
        

        fig=plt.figure() # plot as fn of q-squared
        #plt.subplot(122)
        plt.ylim([0,3.5])
        plt.tick_params(bottom=True,right=True,top=True,left=True,direction='in')
        line,=plt.plot(qsql,fpl,label='$f^+$')
        plt.fill_between(qsql,fpl+dfpl,fpl-dfpl,#color=line.get_color(),
                          alpha=0.3,edgecolor=None)
        plt.errorbar(qsqinputl[:nplus],inpts[:nplus],dinpts[:nplus],fmt='o',
                      markersize=3,
                      color=line.get_color())
        line,=plt.plot(qsql,f0l,label='$f^0$')
        plt.fill_between(qsql,f0l+df0l,f0l-df0l,#color=line.get_color(),
                          alpha=0.3,edgecolor=None)
        plt.errorbar(qsqinputl[nplus:],inpts[nplus:],dinpts[nplus:],fmt='o',
                      markersize=3,
                      color=line.get_color())
        plt.xlabel('$q^2$')
        plt.legend(loc='upper left')
        plt.axis(ymin=-0.5)
        plt.title(outputname,fontsize=12)
        plt.text(-0.215,2.0,'$f(0)={:.3f}\\pm{:.3f}$'.format(fpl[-1],dfpl[-1]),
                 ha='left')
        if savefigs:
            plt.savefig(plotdir+'/'+outputname+'_qsq.pdf',bbox_inches='tight')
        plt.savefig('fpf0_qsq.pdf',bbox_inches='tight')
        plt.show()
        
        plt.close() # so it's not retained in memory
        
        fig=plt.figure() # plot as fn of z
        fpinden=fplusden(zinputl[:nplus])
        f0inden=fzeroden(zinputl[nplus:])
        if fittype=='BGL': # normalisation for BGL f*Bphi(z)/Bphi(zmax) plot
            fpdenl/=fplusden(zmax)
            f0denl/=fzeroden(zmax)
            fpinden/=fplusden(zmax)
            f0inden/=fzeroden(zmax)
        #plt.subplot(121)
        plt.ylim([0,1.0])
        plt.tick_params(bottom=True,right=True,top=True,left=True,direction='in')
        line,=plt.plot(zl,fpdenl*fpl,
                       label=fpluszplotlbl)
        plt.fill_between(zl,fpdenl*(fpl+dfpl),
                          fpdenl*(fpl-dfpl),
                          #color=line.get_color(),
                          alpha=0.3,edgecolor=None)
        plt.errorbar(zinputl[:nplus],fpinden*inpts[:nplus],
                      fpinden*dinpts[:nplus],fmt='o',
                      markersize=3,
                      color=line.get_color())
        line,=plt.plot(zl,f0denl*f0l,
                       label=fzerozplotlbl)
        plt.fill_between(zl,f0denl*(f0l+df0l),
                          f0denl*(f0l-df0l),
                          #color=line.get_color(),
                          alpha=0.3,edgecolor=None)
        plt.errorbar(zinputl[nplus:],f0inden*inpts[nplus:],
                      f0inden*dinpts[nplus:],fmt='o',
                      markersize=3,
                      color=line.get_color())
        plt.xlabel('$z$')
        plt.legend(loc='upper right')
        plt.axis(ymin=-0.2)
        plt.text(-0.205,0.0,'$f(0)={:.3f}\\pm{:.3f}$'.format(fpl[-1],dfpl[-1]),
                 ha='left')
        plt.text(-0.205,-0.15,'Preliminary',fontsize=20,color='gray',alpha=0.2,
                  ha='left',fontfamily='sans-serif')
        plt.title(outputname,fontsize=12)
        if savefigs:
            plt.savefig(plotdir+'/'+outputname+'_z.pdf',bbox_inches='tight')
        plt.savefig('fpf0_z.pdf',bbox_inches='tight')
        plt.show()
        plt.close() # so it's not retained in memory

print('f(0) and acceptance rates for unitarity constraint')
for k in resdict.keys():
    s1='{:d} {:d}  '.format(k[0],k[1])
    f,df=resdict[k]['f(0)'],round(1000*resdict[k]['df(0)'])
    s1+='  f(0)={:6.3f}({:3d})  '.format(f,df)
    n,naccept=resdict[k]['ntries'],resdict[k]['nsamp']
    print(s1+'{:d}/{:d} = {:.2f}'.format(naccept,n,100*naccept/n))    
print('f+ coeffs')
for k in resdict.keys():
    s1='{:d} {:d}  '.format(k[0],k[1])
    al=resdict[k]['ap']
    dal=[round(arg) for arg in 1000*resdict[k]['dap']]
    s2l=['{:6.3f}({:3d})'.format(a,da) for a,da in zip(al,dal)]
    print(s1+' '.join(s2l))
print('f0 coeffs')
for k in resdict.keys():
    s1='{:d} {:d}  '.format(k[0],k[1])
    al=resdict[k]['a0']
    dal=[round(arg) for arg in 1000*resdict[k]['da0']]
    s2l=['{:6.3f}({:3d})'.format(a,da) for a,da in zip(al,dal)]
    print(s1+' '.join(s2l))

        #Yzc(z,zmax,p0lbl,nbplus,nbzero,fpseq,fpden,f0seq,f0den)
        # fitresults=zfit.fit()
        # fitresults['inputs']={}
        # for k,v in (('qsqp',qsqp),('qsq0',qsq0),('input points',inpts),
        #             ('input cov',incov)):
        #     fitresults['inputs'][k]=v
        # fitresults['info']['Min']=Min
        # fitresults['info']['Mout']=Mout
        # for k,v in (('t+',tp),('t-',tm)):
        #     if k in fitresults['info'].keys():
        #         pass
        #     else:
        #         fitresults['info'][k]=v
        #g=f.create_group(decaylbl+'/'+fit.fitlbl+'/'+variation)
        #h5writedict(g,fitresults)

#f.create_dataset('created',data=time.asctime())
#f.close()
