#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### Sampling form factors from Milc paper results ~~~ https://journals.aps.org/prd/pdf/10.1103/PhysRevD.100.034501


# In[1]:


import numpy as np


# In[2]:


milc_tcut = 5.414**2
milc_t0 = 16.5
milc_K = 4
mBst1minus = 28.4
mBst0plus = 32.3


# In[3]:


bkplusmean = [0.3623,-0.9559,-0.8525,0.2785]
bkpluserr = [0.0178,0.1307,0.4783,0.6892]

bkzeromean = [0.1981,-0.1661,-0.6430,-0.3754]
bkzeroerr = [0.0101,0.1130,0.4385,0.4535]

bkmean = np.array(bkplusmean+bkzeromean)
bkerr = np.array(bkpluserr+bkzeroerr)

bkcorrmat = np.array([[1.0000, 0.6023, 0.0326, -0.1288, 0.7122, 0.6035, 0.5659, 0.5516],
[0.6023, 1.0000, 0.4735, 0.2677, 0.7518, 0.9086, 0.9009, 0.8903],
[0.0326, 0.4735, 1.0000, 0.9187, 0.5833, 0.7367, 0.7340, 0.7005],
[-0.1288, 0.2677, 0.9187, 1.0000, 0.4355, 0.5553, 0.5633, 0.5461],
[0.7122, 0.7518, 0.5833, 0.4355, 1.0000, 0.8667, 0.7742, 0.7337],
[0.6035, 0.9086, 0.7367, 0.5553, 0.8667, 1.0000, 0.9687, 0.9359],
[0.5659, 0.9009, 0.7340, 0.5633, 0.7742, 0.9687, 1.0000, 0.9899],
[0.5516, 0.8903, 0.7005, 0.5461, 0.7337, 0.9359, 0.9899, 1.0000]])

bkcovmat = np.zeros((np.shape(bkcorrmat)))
for i in range(np.shape(bkcorrmat)[0]):
    for j in range(np.shape(bkcorrmat)[1]):
        bkcovmat[i,j] = bkcorrmat[i,j]*bkerr[i]*bkerr[j]#**0.5**0.5

bkcovmat = np.array(bkcovmat)


# In[4]:


def z(q2):
    tcutq2 = np.sqrt(milc_tcut-q2)
    tcutt0 = np.sqrt(milc_tcut-milc_t0)
    return (tcutq2-tcutt0)/(tcutq2+tcutt0)


# In[5]:


def q2_from_z(z):
    return milc_tcut - (milc_tcut-milc_t0)*((1+z)/(1-z))**2 


# In[86]:


def fplus_from_q2(q2,bkplus=bkplusmean):
    prefactor = (1-q2/mBst1minus)**-1
    ssum = 0
    for k in range(milc_K):
        ssum += bkplus[k]*(z(q2)**k - (-1)**(k-milc_K) * (k/milc_K) * z(q2)**milc_K)
    return prefactor*ssum


# In[87]:


def fzero_from_q2(q2,bkzero=bkzeromean):
    prefactor = (1-q2/mBst0plus)**-1
    ssum = 0
    for k in range(milc_K):
        ssum += bkzero[k]*(z(q2)**k)
    return prefactor*ssum


# In[88]:


def resample_ff(list_of_q2,nboot):
    bknew = np.random.multivariate_normal(bkmean,bkcovmat,size = nboot)
    #print(np.shape(bknew))
    bkplus = bknew[:,:milc_K]
    bkzero = bknew[:,-milc_K:]
    #print(np.shape(bkzero))
    
    ffzero = np.zeros((len(list_of_q2),nboot))
    ffplus = np.zeros((len(list_of_q2),nboot))
    
    for index,q2 in enumerate(list_of_q2):
        for boot in range(nboot):
            #print(q2,bkplus[boot,:])
            ffplus[index,boot] = fplus_from_q2(q2,bkplus[boot,:])
            ffzero[index,boot] = fzero_from_q2(q2,bkzero[boot,:])
    
    return ffzero,ffplus


# In[ ]:




