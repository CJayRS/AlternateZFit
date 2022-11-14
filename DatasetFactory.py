import numpy as np
from Dataset import Dataset
from Constants import *
import h5py as h5

class DatasetFactory:
    @staticmethod
    def generate(q2_zero, q2_plus):
        nplus = len(q2_plus)
        nzero = len(q2_zero)
        #
        qsq_refK_plus	= np.array(q2_plus)#np.array([23.7283556, 22.11456, 20.07895, 17.5000000]) # you can choose this freely np.array(np.linspace(23.5, 17.5, 7))
        qsq_refK_zero	= np.array(q2_zero)#np.array([23.7283556, 22.11456, 20.07895, 17.5000000]) # you can choose this freely np.array(np.linspace(23.5, 17.5, 7))

        #
        ksq_refK_plus 	= (mBsphys**4+(mKphys**2-qsq_refK_plus)**2-2*mBsphys**2*(mKphys**2+qsq_refK_plus))/(4*mBsphys**2)
        ksq_refK_zero 	= (mBsphys**4+(mKphys**2-qsq_refK_zero)**2-2*mBsphys**2*(mKphys**2+qsq_refK_zero))/(4*mBsphys**2)

        ErefK_plus 	 	= np.sqrt(mKphys**2+ksq_refK_plus)
        ErefK_zero 	 	= np.sqrt(mKphys**2+ksq_refK_zero)
        Deltapar	= + 0.263
        Deltaperp	= - 0.0416


        f=h5.File('BstoK_ref_ff_dat.hdf5', 'r')
        cp_BstoK=np.array(f.get('cp'))
        c0_BstoK=np.array(f.get('c0'))
        Cp0_BstoK=np.array(f.get('Cp0'))
        fp_BstoK 	= np.array(ff_E(ErefK_plus, Deltaperp, cp_BstoK))
        f0_BstoK 	= np.array(ff_E(ErefK_zero, Deltapar ,c0_BstoK))
        ff_ref		= np.r_[ fp_BstoK, f0_BstoK]
        Cp0_ref 	= cov_ff_p0(ErefK_plus, ErefK_zero, Cp0_BstoK, 2,3, Deltaperp, Deltapar)

        bskpts = np.r_[fp_BstoK, f0_BstoK]
        bskcov = np.array(Cp0_ref)

        qsqinputl = np.array(q2_plus + q2_zero)


        zinputl=zed(qsqinputl, tcut, t0)

        npts=len(bskpts)
        bskptslbls=[char for char in nplus*'+'+nzero*'0']

        dbsk=np.sqrt(bskcov.diagonal())
        bskcov=0.5*(bskcov + np.transpose(bskcov))

        # input f+ and f0 values
        tpin=qsqinputl[:nplus]
        zpin=zed(tpin, tcut, t0)
        fpin=bskpts[:nplus]
        dfpin=dbsk[:nplus]
        tzin=qsqinputl[nplus:]
        zzin=zed(tzin, tcut, t0)
        fzin=bskpts[nplus:]
        dfzin=dbsk[nplus:]
        nppts=len(zpin)
        nzpts=len(zzin)

        fpinputs=(tpin, fpin, chi1minusBsK, zpluspole)
        fzinputs=(tzin, fzin, chi0plusBsK, zpluspole) # !!! zpluspole just a dummy here
        zero_ffs = fzin#np.array((tzin, ))
        plus_ffs = fpin#np.array((tpin, ))
        fpinputs=(tpin,fpin,chi1minusBsK,zpluspole)
        fzinputs=(tzin,fzin,chi0plusBsK,zpluspole)
        ds = Dataset(q2_zero, q2_plus, zero_ffs, plus_ffs, bskcov)
        #ds.set_dbinputs(fzinputs, fpinputs)
        return ds
    
    