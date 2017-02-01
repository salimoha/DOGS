# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:17:04 2017

@author: KimuKook
"""

import numpy as np
import math
from scipy import optimize

def emprical_sigma(x,s):
    N = len(x)
    sigma = np.zeros(len(s))
    for jj in range(len(s)):
        mu = np.zeros([N//s[jj]])
        for i in range(N//s[jj]):
            inds = np.arange(i*s[jj],(i+1)*s[jj])
            mu[i] = (x[np.unravel_index(inds,x.shape,'F')]).mean(axis=0)
        sigma[jj] = (mu**2).mean(axis=0)
    return sigma
    
    
def stationary_statistical_learning_reduced(x,m):
    N = len(x)
    M = math.floor(math.sqrt(N))
    s = np.arange(1,2*M+1)
    variance = x.var(axis=0)*len(x)/(len(x)-1)
    x = np.divide(x-x.mean(axis=0),((x.var(axis = 0))*len(x)/(len(x)-1))**(1/2))
    
    sigmac2 = emprical_sigma(x,s)
    tau = np.arange(1,m+1)/(m+1)
    fun = lambda tau:Loss_fun_reduced(tau,sigmac2)
    jac = 
    theta = np.zeros([2*m])
    bnds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
    res_con = optimize.minimize(fun,tau,jac=jac,method='SLSQP',bounds=bnds,options=opt)
    theta[m+1:2*m+1] = np.copy(res_con.x)
    theta = optimum_A(theta[m+1:2*m+1],sigmac2)
    moment2_model,corr_model = Thoe_moment2(np.concatenate((np.array([1]),np.array([0]),theta),axis=0),N)
    sigma2_N = moment2_model[:,-1] * variance
    return sigma2_N,theta,moment2_model,corr_model,sigmac2

#%%
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
    return A.x
    #%%
def Thoe_moment2(theta,N):
    # should be okay
    # muhan 1/28
    m = int(len(theta)/2-1)
    corr_model = exponential_correlation(theta[2:m+2],theta[m+2:],N)
    s = np.arange(1,N+1).reshape(-1,1)
    moment2 = theta[1] + theoritical_sigma(corr_model.reshape(-1,1),s,theta[0])
    return moment2,corr_model
    #%%
def exponential_correlation(A,tau,N):
    # should be okay
    # muhan 1/28
    corr = np.zeros([1,N])
    for ii in range(1,len(tau)+1):
        corr = corr + A[np.unravel_index(ii-1,A.shape,'F')]*np.power(tau[ii-1],np.arange(1,N+1))
    return corr    
#%%

def theoritical_sigma(corr,s,sigma02):
    # should be okay
    # muhan 1/28
    sigma = np.zeros([len(s)])
    for ii in range(len(s)):
        sigma[ii] = 1
        for jj in range(1,s[ii]):
            sigma[ii] = sigma[ii] + 2*(1-jj/s[ii])*corr[jj-1]
        sigma[ii] = sigma02 * sigma[ii] / s[ii]
    return sigma