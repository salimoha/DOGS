# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 21:12:05 2017

@author: KimuKook
"""

import numpy as np    
import scipy
from scipy import optimize

def optimum_A(tau,sigmac2):
    # tau need to be a vector
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
    A_ineq = np.identity(m)
#    b_ineq = np.zeros([m])
    A_eq = np.ones([1,m])
    b_eq = np.array([1])
    x0 = np.ones([m,1])/m
    func = lambda x: 0.5*np.dot(x.T, np.dot(H, x)) + np.dot(c,x)
    jaco = lambda x: np.dot(x.T, H) + c
    cons = ({'type':'ineq',
             'fun':lambda x: x,
             'jac':lambda x: A_ineq},
            {'type':'eq',
             'fun':lambda x: b_eq - np.dot(A_eq,x),
             'jac':lambda x: -A_eq})
    opt = {'disp':False}
    res_cons = optimize.minimize(func,x0,jac=jaco,constraints=cons,method='SLSQP',tol=1e-10,options=opt)
    return res_cons.x