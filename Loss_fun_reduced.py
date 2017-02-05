# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:01:27 2017

@author: KimuKook
"""
import numpy as np
from scipy import optimize

def Loss_fun_reduced(tau,sigmac2):
    L = np.array([0])
    m = len(tau)
    H = np.zeros([m,m])
    Ls = np.zeros([m])
    DL = np.zeros([m])
    for ss in range(1,len(sigmac2)+1):
        for ii in range(m):
            as_ = np.arange(1,ss+1)
            Ls[ii] = 1 / ss * (1 + 2 * np.dot(1-np.divide(as_,ss),tau[ii]**as_.reshape(-1,1))) - sigmac2[ss-1]
            DL[ii] = 2*Ls[ii]*2/ss*(np.dot((as_-np.power(as_,2)/ss),np.power(tau[ii],(as_-1).reshape(-1,1))))
        H = H + np.multiply(Ls.reshape(-1,1),Ls)
    c = np.zeros([m])
    A_ineq = -np.identity(m)
    b_ineq = np.zeros([m])
    A_eq = np.ones([1,m])
    b_eq = np.array([1])
    x0 = np.ones([m,1])/m
    func = lambda x: 0.5*np.dot(x.T, np.dot(H, x)) + np.dot(c,x)
    jaco = lambda x: np.dot(x.T, H) + c
    cons = ({'type':'ineq',
             'fun':lambda x: b_ineq - np.dot(A_ineq,x),
             'jac':lambda x: -A_ineq},
            {'type':'eq',
             'fun':lambda x: b_eq - np.dot(A_eq,x),
             'jac':lambda x: -A_eq})
    opt = {'disp':False}
    A = optimize.minimize(func,x0,jac=jaco,constraints=cons,method='SLSQP',options=opt)
    L = 0.5 * np.dot(A.x.T, np.dot(H , A.x))
    DL = np.multiply(DL.T,A.x)
    return L,DL
#%%
tau = np.arange(1,10)
sigmac2 = np.array([0.1])
L,DL = Loss_fun_reduced(tau,sigmac2)
print(L)
print(DL)