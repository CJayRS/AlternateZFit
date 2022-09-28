import numpy as np
import masses
from kinematic_functions import zed
from math import sqrt, pi

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

zmax=zed(0, tcut, t0)

# z-values for pole positions
zpluspole=zed(mBstar**2, tcut, t0)
#zzeropole=zed(mBstar0plus**2, tcut, t0) # mBstar0plus is above threshold
#nplus=2 # number of input values for f+
#nzero=3 # number of input values for f0
#qsqmin='17.50' # (17.50, 18., 00, 18.50, 19.00)
path=''



#@jit(nopython=True)
def z(t):
    return zed(t, tcut, t0)



polevalszero = []
polevalsplus = [5.324700**2]
chi0plus = 1.48e-02
chi1minus = 6.03e-04
chiplus = chi1minus
chizero = chi0plus
mKphys		= masses.mK
mBsphys		= masses.mBs
eta = 1
tstar = 29.349571
t0 = 16.505107
tplus = 34.368484
tminus = 23.728356



#(mKphys+mBsphys)**2



alpha = np.arctan((2*np.sqrt(tp-tcut)*np.sqrt(tcut-t0))/(2*tcut - tp - t0))
cos_alpha = np.cos(alpha)
sin_alpha = np.sin(alpha)
sin_2alpha = np.sin(2*alpha)

f0_norm = 1/np.sqrt(2*alpha)
f1_norm = (2*alpha - 2*sin_alpha**2 /alpha)**(-.5)
f2_b = (sin_alpha-sin_alpha**2 *cos_alpha/alpha)/(sin_alpha**2/alpha - alpha)
f2_c = -(sin_2alpha + 2*f2_b*sin_alpha)/(2*alpha)
f2_norm = (2*alpha*(1+f2_b**2+f2_c**2) + (1+f2_c)*f2_b*4*sin_alpha + f2_c*2*sin_2alpha)**(-.5)

def ff_E(Evec, pole, coeff):
  # construct ff from HMChPT in continuum limit
  return [1./(E+pole)*np.sum([E**i*coeff[i] for i in range(len(coeff))]) for E in Evec]

def cov_ff_p0(Evec_p, Evec_0, C,Np, N0, pole_p, pole_0):
  # construct covariance matrix for ff from HMChPT in continuum limit
  Y_E_p_vec   	= lambda E_p: np.r_[ np.array([1./(E_p+pole_p)*E_p**i for i in range(Np)])]
  Y_E_0_vec   	= lambda E_0: np.r_[ np.array([1./(E_0+pole_0)*E_0**i for i in range(N0)])]
  Cpp		= np.array([[np.dot(Y_E_p_vec(E1), np.dot(C[:Np, :Np], Y_E_p_vec(E2)))
					for E1 in Evec_p] for E2 in Evec_p])
  C00		= np.array([[np.dot(Y_E_0_vec(E1), np.dot(C[Np:, Np:], Y_E_0_vec(E2)))
					for E1 in Evec_0] for E2 in Evec_0])
  Cp0		= np.array([[np.dot(Y_E_p_vec(E1), np.dot(C[:Np:, Np:], Y_E_0_vec(E2)))
					for E1 in Evec_p] for E2 in Evec_0])
  M0		= np.r_['-1', Cpp  ,Cp0.T]
  M1		= np.r_['-1', Cp0  ,C00  ]
  M		= np.r_[M0, M1]
  return M

def phiplus(t, chi = chiplus):
    #chi = 1
    K = 48*np.pi
    a = 3
    b = 2
    rq = np.sqrt(tstar-t)
    rminus = np.sqrt(tstar-tminus)
    r0 = np.sqrt(tstar-t0)
    val = np.sqrt(eta/(K*chi))*(rq**((a+1)/2))*r0**(-1/2)*(rq+r0)*((rq+np.sqrt(tstar))**(-b-3))*(rq+rminus)**(a/2)
    for i in range(len(polevalsplus)):
        val *= (z(t)-z(polevalsplus[i]))/(1-np.conjugate(z(polevalsplus[i]))*z(t))
    return val



def phizero(t, chi = chizero):
    #chi = 1
    K = 16*np.pi/(tplus*tminus)
    a = 1
    b = 1
    rq = np.sqrt(tstar-t)
    rminus = np.sqrt(tstar-tminus)
    r0 = np.sqrt(tstar-t0)
    val = np.sqrt(eta/(K*chi))*(rq**((a+1)/2))*r0**(-1/2)*(rq+r0)*((rq+np.sqrt(tstar))**(-b-3))*(rq+rminus)**(a/2)
    for i in range(len(polevalszero)):
        val *= (z(t)-z(polevalszero[i]))/(1-np.conjugate(z(polevalszero[i]))*z(t))
    return val

def alt_polynomial(t, n):
    '''
    t: number - q^2 value to evaluate the functions at
    n: n^th polynomial to be evaluated
    returns: function evaluation (float)
    '''
    if n == 0:
        return f0_norm*1
    if n == 1:
        return f1_norm*(z(t)-(sin_alpha)/alpha)
    if n == 2:
        return f2_norm*(z(t)**2 + z(t)*f2_b + f2_c)