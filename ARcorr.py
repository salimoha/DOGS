# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:06:20 2017

@author: KimuKook
"""
import numpy as np
from scipy import signal

def ARcorr(P,m,var_eps):
    m1 = np.array([1])
    m2 = np.concatenate((m1,P),axis=0)
    [r,p,k] = scipy.signal.residuez(m1,m2,tol=1e-3,rtype='avg')
    beta = np.zeros([m+1])
    for i in range(m+1):
        beta[i] = sum(np.multiply(r,p**i))
    Gama = np.correlate(beta,beta,"full")
    Gama = np.copy(Gama[m:2*m+1])
    Gama = np.dot(Gama,var_eps)
    Gama0 = np.copy(Gama[0])
    Gama = Gama[1:]/Gama0
    return Gama,Gama0