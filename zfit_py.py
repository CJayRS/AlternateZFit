from Constants import *
from pickle import MEMOIZE
from re import X
import h5py as h5
from math import sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
from plot_settings import plotparams
from kinematic_functions import zed#qsq, k,wl
from BGL import phi#, blaschke
from scipy.optimize import minimize_scalar
import scipy
import time
import masses
from tqdm import tqdm
from numba import jit
import cProfile
#from multiprocessing import Pool, TimeoutError, Process, Manager
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import freeze_support
import warnings
import snakeviz
import pstats

from Dataset import Dataset

warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")



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



def phi_ff(t, ff):
    if ff == 0:
        return phizero(t)
    elif ff == 1:
        return phiplus(t)


def zfunction(t, a, ff, functtype = "zfit"):
    #print(a)
    if functtype  == "zfit":
        if ff == 0:
            tempsum = 0
            for n, an in enumerate(a):
                tempsum += 1/phi_ff(t, ff)*an*z(t)**n #(1-t/(mBstar**2))*an*z(t)**n
            return tempsum
        elif ff == 1:
            tempsum = 0
            #K = len(a)
            #print(K)
            for n, an in enumerate(a):
                #print(n, an)
                tempsum += 1/phi_ff(t, ff)*an*z(t)**n #BCL: (1-t/(mBstar**2))*an*(z(t)**n - (-1)**(n-K)*n/K *z(t)**K)
            return tempsum
        else:
            print("invalid ff value")
    if functtype  == "altfit":
        
        if ff == 0 or 1:
            tempsum = 0
            for n, an in enumerate(a):
                tempsum += an*alt_polynomial(t, n) #(1-t/(mBstar**2))*an*z(t)**n
            return tempsum/phi_ff(t, ff)
        else:
            print("invalid ff value")







#ffs_from_input_q2([15, 20], [15, 17, 20])


def function_to_minimise(inputs, inputq2, nzero, nplus, inputffs, invcov, functtype = "zfit"):#, lam
    #make sure to define:inputq2, inputffs, inputcov, functtype = "zfit" before this
    coefficients = (inputs[0:nzero], inputs[nzero:nzero+nplus])
    ssum = 0
    fflist = [zfunction(q2, coefficients[0], 0,functtype) for q2 in inputq2[0]] + [zfunction(q2, coefficients[1], 1,functtype) for q2 in inputq2[1]]
    
    for i in range(len(fflist)):
        for j in range(len(fflist)):
            ssum += (fflist[i]-inputffs[i])*invcov[i, j]*(fflist[j]-inputffs[j])
    return ssum# + lam*(zfunction(0, coefficients[0], 0)-zfunction(0, coefficients[1], 1))**2


def worker_job(args):
    (resampledffdata_element, inputq2, invcov, functtype, constype, initial_guess, nzero, nplus)= args#, nprocess
    # if nprocess%10 == 0:
    #     print(nprocess)
    return scipy.optimize.minimize(function_to_minimise, initial_guess, args = (inputq2, nzero, nplus, resampledffdata_element, invcov, functtype), method='trust-constr', constraints=constype, tol = 1e-04).x

def zfitit(inputq2, nboot, functtype = "zfit"):
    ffdata = ffs_from_input_q2(inputq2[0], inputq2[1])
    resampledffdata = ffdata.resample(nboot)
    invcov = np.linalg.inv(ffdata.cov)
    initial_guess_forjack = scipy.optimize.minimize(function_to_minimise, np.linspace(0.1, 0.1, nzero+nplus), args = (inputq2, nzero, nplus, ffdata.fflist(), invcov), method='trust-constr', constraints=cons, tol = 1e-04)
    initial_guess = initial_guess_forjack.x
    if functtype == "zfit":
        constype = cons
    elif functtype == "altfit":
        constype = cons_alt
    inputs = [(ds.fflist(), inputq2, invcov, functtype, constype, initial_guess, nzero, nplus) for ds in resampledffdata]
    with Pool(processes=None) as pool:
        outputcoeffs = []
        for coeff in tqdm(pool.imap(worker_job, inputs), total = nboot):#map
            outputcoeffs.append(coeff)
    return outputcoeffs

if __name__ == '__main__':
    ninputsamples = 10
    inputq2sets = np.zeros((5, ninputsamples))
    for i in range(5):
        inputq2sets[i, :] = np.random.uniform(17.5, 23.78, ninputsamples)
    print(inputq2sets)
    inputq2 = np.array(([15, 17, 20], [15, 20]), dtype=object)
    nzero = 3 #len(inputq2[0])#+1
    nplus = 2 #len(inputq2[1])#+1
    genffdata = ffs_from_input_q2(inputq2[0], inputq2[1])
    inputffs = np.append(genffdata[0][1], genffdata[1][1]) #temporary (havent been resampled yet)
    print(inputffs)
    inputcov = genffdata[2]
    #lam = 10**3
    invcov = np.linalg.inv(inputcov)
    #cons = [{'type': 'ineq', 'fun': lambda inlist: zfunction(0, inlist[:nzero], 0)-zfunction(0, inlist[nzero:nzero+nplus], 1)},
    #        {'type': 'ineq', 'fun': lambda inlist: -zfunction(0, inlist[:nzero], 0)+zfunction(0, inlist[nzero:nzero+nplus], 1)}]
    cons = [{'type': 'eq', 'fun': lambda inlist: zfunction(0, inlist[:nzero], 0)-zfunction(0, inlist[nzero:nzero+nplus], 1)}, {'type': 'ineq', 'fun': lambda inlist: 1- sum([inlist[i]**2 for i in range(len(inlist))])}]
    #scipy.optimize.minimize(function_to_minimise, np.linspace(0.1, 0.1, nzero+nplus), args = (inputq2, inputffs, inputcov, "altfit"), method='trust-constr', constraints=cons).x #, bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)], ,functtype = "zfit"




    q2rangetoplot = np.linspace(0, 23, 24)
    coefficients = np.array([-0.16989614, -0.12108523,  0.04556504, -0.04126728, -0.05150325])
    ffstoplot = ffs_from_input_q2(inputq2[0], inputq2[1])
    inputffs = np.append(ffstoplot[0][1], ffstoplot[1][1])
    print(inputq2)
    print(ffstoplot)
    plt.scatter(ffstoplot[0][0], ffstoplot[0][1], color = "blue")
    plt.scatter(ffstoplot[1][0], ffstoplot[1][1], color = "orange")
    plt.plot(q2rangetoplot, [zfunction(q2, coefficients[:nzero], 0,"altfit") for q2 in q2rangetoplot], color = "blue")
    plt.plot(q2rangetoplot, [zfunction(q2, coefficients[nzero:nzero+nplus], 1,"altfit") for q2 in q2rangetoplot], color = "orange")
    print(zfunction(0, coefficients[nzero:nzero+nplus], 1,"altfit")-zfunction(0, coefficients[:nzero], 0,"altfit"))
    print(coefficients[:nzero], coefficients[nzero:nzero+nplus])




    inputq2 = np.array(([17, 18, 20], [17, 20]), dtype=object)


    nzero = 3 #len(inputq2[0])#+1
    nplus = 2 #len(inputq2[1])#+1
    genffdata = ffs_from_input_q2(inputq2[0], inputq2[1])
    inputffs = np.append(genffdata[0][1], genffdata[1][1]) #temporary (havent been resampled yet)
    #print(inputffs)
    inputcov = genffdata[2]
    #lam = 10**3
    invcov = np.linalg.inv(inputcov)
    #cons = [{'type': 'ineq', 'fun': lambda inlist: zfunction(0, inlist[:nzero], 0)-zfunction(0, inlist[nzero:nzero+nplus], 1)},
    #        {'type': 'ineq', 'fun': lambda inlist: -zfunction(0, inlist[:nzero], 0)+zfunction(0, inlist[nzero:nzero+nplus], 1)}]
    cons = [{'type': 'eq', 'fun': lambda inlist: zfunction(0, inlist[:nzero], 0)-zfunction(0, inlist[nzero:nzero+nplus], 1)}, {'type': 'ineq', 'fun': lambda inlist: 1- sum([inlist[i]**2 for i in range(len(inlist))])}]
    cons_alt = [{'type': 'eq', 'fun': lambda inlist: zfunction(0, inlist[:nzero], 0,"altfit")-zfunction(0, inlist[nzero:nzero+nplus], 1,"altfit")}, {'type': 'ineq', 'fun': lambda inlist: 1- sum([inlist[i]**2 for i in range(len(inlist))])}]


    ffdata = ffs_from_input_q2(inputq2[0], inputq2[1])
    invcov = np.linalg.inv(ffdata[2])

    
    #.x#, bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)], ,functtype = "zfit"


    q2rangetoplot = np.linspace(0, 23, 24)
    coefficients = np.array([ 0.0745317 , -0.3148924 ,  0.1163006 ,  0.01836438, -0.07009019])
    ffstoplot = ffs_from_input_q2(inputq2[0], inputq2[1])
    inputffs = np.append(ffstoplot[0][1], ffstoplot[1][1])
    #print(inputq2)
    #print(ffstoplot)
    # plt.scatter(ffstoplot[0][0], ffstoplot[0][1], color = "blue")
    # plt.scatter(ffstoplot[1][0], ffstoplot[1][1], color = "orange")
    # plt.plot(q2rangetoplot, [zfunction(q2, coefficients[:nzero], 0) for q2 in q2rangetoplot], color = "blue")
    # plt.plot(q2rangetoplot, [zfunction(q2, coefficients[nzero:nzero+nplus], 1) for q2 in q2rangetoplot], color = "orange")
    # print(zfunction(0, coefficients[nzero:nzero+nplus], 1)-zfunction(0, coefficients[:nzero], 0))
    # print(coefficients[:nzero], coefficients[nzero:nzero+nplus])




    #ffdata = ffs_from_input_q2(inputq2[0], inputq2[1])
    #a = np.random.multivariate_normal(np.append(ffdata[1][1], ffdata[0][1]), ffdata[2], size=10)#[:, 1]




    #inputq2 = np.array(([17.5, 19, 23], [17.5, 23]), dtype=object)
    nboot = 5000
#scipy.optimize.minimize(function_to_minimise, np.linspace(0.1, 0.1, nzero+nplus), args = (inputq2, nzero, nplus, np.append(ffdata[0][1], ffdata[1][1]), invcov), method='trust-constr', constraints=cons, tol = 1e-04)
if __name__ == '__main__':
    # Initialize profile class and call regression() function
    profiler = cProfile.Profile()
    profiler.enable()
    coarray = np.array(zfitit(inputq2, nboot)).T
    profiler.disable()
    profiler.dump_stats('dumpedstats.prof')
    #exit()


nzero = 3
nplus = 2

#inputq2 = np.array(([17.5, 19, 23], [17.5, 23]), dtype=object)
if __name__ == '__main__':
    coarray_alt = np.array(zfitit(inputq2, nboot, "altfit")).T



# plt.rcParams["figure.figsize"] = (15, 15)
# fig, axs = plt.subplots(3, 2)
# #fig.suptitle(r'Bounds at $q^2 = 0$ with randomly generated input form factor $q^2$ values')
# axs[0, 0].hist(coarray[0, :])
# axs[0, 0].set_title("c_0")
# axs[0, 1].hist(coarray[1, :])
# axs[0, 1].set_title("c_1")
# axs[1, 0].hist(coarray[2, :])
# axs[1, 0].set_title("c_2")
# axs[1, 1].hist(coarray[3, :])
# axs[1, 1].set_title("c_3")
# axs[2, 1].hist(coarray[4, :])
# axs[2, 1].set_title("c_4")
# plt.show()


if __name__ == '__main__':
    q2rangetoplot = np.linspace(0, 23, 70)
    ffarray = np.zeros((2, len(q2rangetoplot), np.shape(coarray)[1]))
if __name__ == '__main__':
    for i, q2 in enumerate(q2rangetoplot):
        for j in range(np.shape(coarray)[1]):
            ffarray[0, i,j] = zfunction(q2, coarray[:nzero, j], 0)
            ffarray[1, i,j] = zfunction(q2, coarray[nzero:nzero+nplus, j], 1)
    sigmabelow = round(np.shape(coarray)[1] * (1-0.6827)/2)
    sigmaabove = round(np.shape(coarray)[1] * (1+0.6827)/2)
    sorted_ffarray = np.sort(ffarray, 2)



if __name__ == '__main__':
    q2rangetoplot = np.linspace(0, 23, 70)
    ffstoplot = ffs_from_input_q2(inputq2[0], inputq2[1])
    inputffs = np.append(ffstoplot[0][1], ffstoplot[1][1])
    plt.scatter(ffstoplot[0][0], ffstoplot[0][1], color = "blue")
    plt.scatter(ffstoplot[1][0], ffstoplot[1][1], color = "orange")
    for i in range(np.shape(coarray)[1]):
        #plt.plot(q2rangetoplot, [zfunction(q2, coarray[:nzero, i], 0) for q2 in q2rangetoplot], color = "blue", alpha=0.005)
        #plt.plot(q2rangetoplot, [zfunction(q2, coarray[nzero:nzero+nplus, i], 1) for q2 in q2rangetoplot], color = "orange", alpha=0.005)
        plt.plot(q2rangetoplot, [sorted_ffarray[0, q2, sigmabelow] for q2 in range(len(q2rangetoplot))], color = "blue")
        plt.plot(q2rangetoplot, [sorted_ffarray[0, q2, sigmaabove] for q2 in range(len(q2rangetoplot))], color = "blue")
        plt.plot(q2rangetoplot, [sorted_ffarray[1, q2, sigmabelow] for q2 in range(len(q2rangetoplot))], color = "orange")
        plt.plot(q2rangetoplot, [sorted_ffarray[1, q2, sigmaabove] for q2 in range(len(q2rangetoplot))], color = "orange")
    plt.show()


if __name__ == '__main__':
    q2rangetoplot = np.linspace(0, 23, 70)
    ffarray_alt = np.zeros((2, len(q2rangetoplot), np.shape(coarray_alt)[1]))
    for i, q2 in enumerate(q2rangetoplot):
        for j in range(np.shape(coarray_alt)[1]):
            ffarray_alt[0, i,j] = zfunction(q2, coarray_alt[:nzero, j], 0,"altfit")
            ffarray_alt[1, i,j] = zfunction(q2, coarray_alt[nzero:nzero+nplus, j], 1,"altfit")
    sigmabelow_alt = round(np.shape(coarray_alt)[1] * (1-0.6827)/2)
    sigmaabove_alt = round(np.shape(coarray_alt)[1] * (1+0.6827)/2)
    sorted_ffarray_alt = np.sort(ffarray_alt, 2)



if __name__ == '__main__':
    coeff_squared = np.zeros((np.shape(coarray)[1]))
    coeff_alt_squared = np.zeros((np.shape(coarray_alt)[1]))

    for i in range(np.shape(coarray)[1]):
        coeff_squared[i] = np.sum([coarray[j, i]**2 for j in range(np.shape(coarray)[0])])
    for i in range(np.shape(coarray_alt)[1]):
        coeff_alt_squared[i] = np.sum([coarray_alt[j, i]**2 for j in range(np.shape(coarray_alt)[0])])
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.hist(coeff_squared, alpha=0.5, label = "traditional z-fit")
    plt.hist(coeff_alt_squared, alpha=0.5, label = "alternative z-fit")
    plt.xlabel("Sum of coefficients squared")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    avg_coeffs = np.mean(coarray, axis=1)
    avg_coeffs_alt = np.mean(coarray_alt, axis=1)
    print("coeff = ", avg_coeffs)

    print("coeff_alt = ", coeff_alt_squared)

    print("f0 z^2 term is ", f2_norm*avg_coeffs_alt[2])
    print("f0 z term is ", f2_norm*avg_coeffs_alt[2]*f2_b + f1_norm*avg_coeffs_alt[1])
    print("f0 constant term is ", f0_norm*avg_coeffs_alt[0] + f1_norm*avg_coeffs_alt[1]*(-(sin_alpha)/alpha) + f2_norm*avg_coeffs_alt[2]*f2_c)

    print("f+ z term is ", f1_norm*avg_coeffs_alt[4])
    print("f+ constant term is ", f0_norm*avg_coeffs_alt[3] + f1_norm*avg_coeffs_alt[4]*(-(sin_alpha)/alpha))



    plt.rcParams["figure.figsize"] = (20, 20)
    q2rangetoplot = np.linspace(0, 23, 70)
    ffstoplot = ffs_from_input_q2(inputq2[0], inputq2[1])
    inputffs = np.append(ffstoplot[0][1], ffstoplot[1][1])
    plt.scatter(ffstoplot[0][0], ffstoplot[0][1], color = "blue")
    plt.scatter(ffstoplot[1][0], ffstoplot[1][1], color = "orange")
    plt.xlabel("$q^2 (GeV^2)$")
    plt.ylabel("Form factor")
    for i in range(np.shape(coarray_alt)[1]):
        #plt.plot(q2rangetoplot, [zfunction(q2, coarray[:nzero, i], 0) for q2 in q2rangetoplot], color = "blue", alpha=0.005)
        #plt.plot(q2rangetoplot, [zfunction(q2, coarray[nzero:nzero+nplus, i], 1) for q2 in q2rangetoplot], color = "orange", alpha=0.005)
        plt.plot(q2rangetoplot, [sorted_ffarray_alt[0, q2, sigmabelow_alt] for q2 in range(len(q2rangetoplot))], color = "blue", label = "alt fit")
        plt.plot(q2rangetoplot, [sorted_ffarray_alt[0, q2, sigmaabove_alt] for q2 in range(len(q2rangetoplot))], color = "blue")
        plt.plot(q2rangetoplot, [sorted_ffarray_alt[1, q2, sigmabelow_alt] for q2 in range(len(q2rangetoplot))], color = "orange", label = "alt fit")
        plt.plot(q2rangetoplot, [sorted_ffarray_alt[1, q2, sigmaabove_alt] for q2 in range(len(q2rangetoplot))], color = "orange")
    for i in range(np.shape(coarray)[1]):
        #plt.plot(q2rangetoplot, [zfunction(q2, coarray[:nzero, i], 0) for q2 in q2rangetoplot], color = "blue", alpha=0.005)
        #plt.plot(q2rangetoplot, [zfunction(q2, coarray[nzero:nzero+nplus, i], 1) for q2 in q2rangetoplot], color = "orange", alpha=0.005)
        plt.plot(q2rangetoplot, [sorted_ffarray[0, q2, sigmabelow] for q2 in range(len(q2rangetoplot))], color = "green", label = "z fit")
        plt.plot(q2rangetoplot, [sorted_ffarray[0, q2, sigmaabove] for q2 in range(len(q2rangetoplot))], color = "green")
        plt.plot(q2rangetoplot, [sorted_ffarray[1, q2, sigmabelow] for q2 in range(len(q2rangetoplot))], color = "red", label = "z fit")
        plt.plot(q2rangetoplot, [sorted_ffarray[1, q2, sigmaabove] for q2 in range(len(q2rangetoplot))], color = "red")
    #plt.legend()
    plt.show()



    plt.rcParams["figure.figsize"] = (20, 10)
    q2rangetoplot = np.linspace(0, 23, 70)
    ffstoplot = ffs_from_input_q2(inputq2[0], inputq2[1])
    inputffs = np.append(ffstoplot[0][1], ffstoplot[1][1])
    plt.xlabel("$q^2 (GeV^2)$")
    #plt.ylabel("Form factor")
    for i in range(np.shape(coarray_alt)[1]):
        #plt.plot(q2rangetoplot, [zfunction(q2, coarray[:nzero, i], 0) for q2 in q2rangetoplot], color = "blue", alpha=0.005)
        #plt.plot(q2rangetoplot, [zfunction(q2, coarray[nzero:nzero+nplus, i], 1) for q2 in q2rangetoplot], color = "orange", alpha=0.005)
        plt.plot(q2rangetoplot, [abs((sorted_ffarray_alt[0, q2, sigmabelow_alt]-sorted_ffarray_alt[0, q2, sigmaabove_alt])/(sorted_ffarray[0, q2, sigmabelow]-sorted_ffarray[0, q2, sigmaabove])) for q2 in range(len(q2rangetoplot))], color = "blue", label = "alt fit")
        plt.plot(q2rangetoplot, [abs((sorted_ffarray_alt[1, q2, sigmabelow_alt]-sorted_ffarray_alt[1, q2, sigmaabove_alt])/(sorted_ffarray[1, q2, sigmabelow]-sorted_ffarray[1, q2, sigmaabove])) for q2 in range(len(q2rangetoplot))], color = "red", label = "alt fit")
    plt.show()
