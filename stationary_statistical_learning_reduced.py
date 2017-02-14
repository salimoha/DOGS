# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:17:04 2017

@author: KimuKook
"""

import numpy as np
import pandas as pd
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
    print("--------------")
    sigmac2 = emprical_sigma(x,s)
    tau = np.ones((1,m))/(m+1)
    fun = lambda tau:Loss_fun_reduced(tau,sigmac2)
    jac = lambda tau:jac_Loss_fun_reduced(tau,sigmac2)
    theta = np.zeros([2*m])
#    bnds = ((0,1),(0,1),(0,1),(0,1))
#    bnds = (np.zeros((1,m)), np.ones((1,m)))
    bnds = tuple([ (0,1) for i in range(int(m))])
    opt = {'disp':False}
    print("--------------")
    res_con = optimize.minimize(fun,tau,jac=jac,method='L-BFGS-B',bounds=bnds,options=opt)
#    theta[m+1:2*m+1] = np.copy(res_con.x)
    theta_tau = np.copy(res_con.x)
#    theta = optimum_A(theta[m+1:2*m+1],sigmac2)
    theta_A = optimum_A(theta_tau,sigmac2)
    theta = np.concatenate((theta_A,theta_tau), axis=0)
    print("theta = ", theta)
    moment2_model,corr_model = Thoe_moment2(np.concatenate((np.array([1]),np.array([0]),theta),axis=0),N)
    print("moment2 = ", moment2_model.shape)
    sigma2_N = moment2_model[-1] * variance
    print("---------============--------------")
    print("moment2 = ", moment2_model[-1])
    print("var = ", variance)
    print("---------============--------------")
    print("sigma2N = ", sigma2_N)
#    return theta
#    return theta,moment2_model,corr_model,sigmac2, sigma2_N
    return sigma2_N

#%%
def Loss_fun_reduced(tau,sigmac2):
#   This function minimizes the linear part of the loss function that is a least square fit for the varaince of time averaging errror
#   This is done using alternative manimization as fixing the tau value fixed.
#   Autocorrelation function is rho = A_1 tau_1^k + ... +A_m tau_m^k
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
        
    H1 = np.concatenate((H, -1*np.ones([m,1])),axis=1)
    H2 = np.concatenate((np.ones([1,m]),np.ones([1,1])*0),axis=1)
    Ah = np.vstack((H1,H2))
    b = np.vstack(( 0*np.ones([m,1]),1))
    
    A_lambda = np.dot(np.linalg.pinv(Ah),b)
    A = np.copy(A_lambda[:m])
    L = 0.5 * np.dot(A.T, np.dot(H , A))
    return L

def Loss_fun_reduced2(tau,sigmac2):
#   This function minimizes the linear part of the loss function that is a least square fit for the varaince of time averaging errror
#   This is done using alternative manimization as fixing the tau value fixed.
#   Autocorrelation function is rho = A_1 tau_1^k + ... +A_m tau_m^k
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
    A_ineq = np.identity(m)
    b_ineq = np.zeros([m])
    A_eq = np.ones([1,m])
    b_eq = np.array([1])
    x0 = np.ones([m,1])/m
    func = lambda x: 0.5*np.dot(x.T, np.dot(H, x)) + np.dot(c,x)
    jaco = lambda x: np.dot(x.T, H) + c
    cons = ({'type':'ineq',
             'fun':lambda x: np.dot(A_ineq,x) -  b_ineq,
             'jac':lambda x: A_ineq},
            {'type':'eq',
             'fun':lambda x: b_eq - np.dot(A_eq,x),
             'jac':lambda x: -A_eq})
    opt = {'disp':False}
    
    A = optimize.minimize(func,x0,jac=jaco,constraints=cons,method='SLSQP',options=opt)
    L = 0.5 * np.dot(A.x.T, np.dot(H , A.x))
    return L

    
    
def jac_Loss_fun_reduced(tau,sigmac2):
#   This function minimizes the linear part of the loss function that is a least square fit for the varaince of time averaging errror
#   This is done using alternative manimization as fixing the tau value fixed.
#   Autocorrelation function is rho = A_1 tau_1^k + ... +A_m tau_m^k
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
    A_ineq = np.identity(m)
    b_ineq = np.zeros([m])
    A_eq = np.ones([1,m])
    b_eq = np.array([1])
    x0 = np.ones([m,1])/m
    func = lambda x: 0.5*np.dot(x.T, np.dot(H, x)) + np.dot(c,x)
    jaco = lambda x: np.dot(x.T, H) + c
    cons = ({'type':'ineq',
             'fun':lambda x: np.dot(A_ineq,x) -  b_ineq,
             'jac':lambda x: A_ineq},
            {'type':'eq',
             'fun':lambda x: b_eq - np.dot(A_eq,x),
             'jac':lambda x: -A_eq})
    opt = {'disp':False}
    
    A = optimize.minimize(func,x0,jac=jaco,constraints=cons,method='SLSQP',options=opt)
#    L = 0.5 * np.dot(A.x.T, np.dot(H , A.x))
    DL = np.multiply(DL.T,A.x)

#scipy.optimize.leastsq

    return DL
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
# CAN WE write this function more efficient
def theoritical_sigma(corr,s,sigma02):
    # should be okay
    # muhan 1/28
    sigma = np.zeros([len(s)*1.0])
    for ii in range(int(len(s))):
        sigma[ii] = 1.0
        for jj in range(1,int(s[ii])):
            sigma[ii] = sigma[ii] + 2*(1-jj/s[ii])*corr[jj-1]
        sigma[ii] = sigma02 * sigma[ii] / s[ii]
    return sigma
    

#from __future__ import division
import csv
import numpy as np
import matplotlib.pyplot as plt

#     This file is part of DELTADOGS package.
#     DELTADOGS is free software for global optimization of computationally expensive function evaluaitons
#     you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#     DELTADOGS is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with DELTADOGS.  If not, see <http://www.gnu.org/licenses/>.

# Author: Shahrouz Alimohammadi
# Modified: Dec. 2016


# KSE simulation using IMEXRKi4CBA(3s)
data1FilePath = "UQ/data1.txt"

def transient_detector(x=[]):
# transient_time_detector(x) is an automatic procedure to determine the nonstationary part a signal from the stationary part.
#  It finds the transient time of the simulation using the minimum variance intreval.
#  INPUT:
#  x: is the signal which after some transient part the signal becomes stationary
#  OUTPUT:
#  ind: is the index of signal that after that the signal could be considered as a stationry signal.

# If you use this code please cite:
# Beyhaghi, P., Alimohammadi, S., and Bewley, T., A multiscale, asymptotically unbiased approach to uncertainty quantification 
# in the numerical approximation of infinite time-averaged statistics. Submitted to Journal of Uncertainity Quantification. 

    N = len(x)
    k = np.int_([N/2])
    y = np.zeros((k, 1))
    for kk in np.arange(k):
        y[kk] = np.var(x[kk+1:])*1.0/(N-kk-1.0)
    y = np.array(-y)
    ind = y.argmax(0)
    print('index of transient point in the signal:')
    print(ind)
    return ind




def readInputFile(filePath):
#     reads a time series data from a file

#    retVal = []
#    with open(filePath, 'rb') as csvfile:
#        filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#        for row in filereader:
#            retVal.append([int(row[0]), int(row[1]), int(row[2])])
    retVal=[]
    with open(filePath) as file:
         line=file.readline()
         arr=[float(a) for a in line.split(',')]
 #        retVal.append(file.readline())
         retVal.append(arr)
    return retVal[0]

# TEST
x = readInputFile(data1FilePath)
x = x[:10000]
index = transient_detector(x)



## sampled time intervals 1
t = np.arange(0., len(x))
# red dashes transient detector, green curve simulation results of KSE
plt.plot(t, x, '-g')
plt.plot([index,index], [np.min(x)/2.0, np.max(x)], '--r')
plt.show()



    
m=12.0    
#sigmac2 = np.random.rand(100)
#tau=np.ones((m,1))/m
#tau = pd.DataFrame(np.ones((m,1))/m).values
#x = np.random.rand(100)
x = pd.DataFrame(x).values

sigma2_N = stationary_statistical_learning_reduced(x,m)

print("sigma2_N == ", sigma2_N)




#%%

import scipy.io as sio
fm = sio.loadmat('/Users/shahrouz/Desktop/IC8.mat')

d = fm['Drag']
x = d[0][1]
m=4
sigma2_N = stationary_statistical_learning_reduced(x,m)



