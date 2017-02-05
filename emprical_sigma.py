# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 14:41:25 2017

@author: KimuKook
"""

import numpy as np
# Can s and x be a matrix?
# Code are written as x and s are vectors.
# Example:
#    x = np.array([[1,2,3],[4,2,1],[4,5,6]])
#    s = np.array([2,2,2])

def emprical_sigma(x,s):
    N = x.shape[1]
    sigma = np.zeros(s.shape[1])
    for jj in range(s.shape[1]):
        mu = np.zeros([N//s[jj]])
        for i in range(N//s[jj]):
            inds = np.arange(i*s[jj],(i+1)*s[jj])
            mu[i] = (x[np.unravel_index(inds,x.shape,'F')]).mean(axis=0)
        sigma[jj] = (mu**2).mean(axis=0)
    return sigma