# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:10:13 2017

@author: KimuKook
"""

import numpy as np
import math
from scipy.spatial import Delaunay

def tringulation_search_bound_constantK(inter_par,xi,K,ind_min):
    global n
    tri = Delaunay(xi.T)
    Sc = np.zeros([np.shape(tri)[0]])
    Scl = np.zeros([np.zeros(tri)[0]])
    for ii in range(np.shape(tri)[0]):
        xc,R2 = circhyp(xi[:,tri[ii,:]],n)
        x = xi[:,tri[ii,:]] @ np.ones([n+1,1])/(n+1)
        Sc[ii] = interpolate_val(x,inter_par) - K * (R2 - np.linalg.norm(x-xc)**2)
        if np.sum(ind_min == tri[ii,:]):
            Scl[ii] = Sc[ii]
        else:
            Scl[ii] = math.inf
    # Global one    
    t = np.min(Sc)
    ind = np.argmin(Sc)
    R2,xc = circhyp(xi[:,tri[ind,:]],n)
    x = xi[:,tri[ind,:]] @ np.ones([n+1,1])/(n+1)
    xm,ym = Constant_K_Search(x,inter_par,xc,R2,K)
    # Local one
    t = np.min(Sc)
    ind = np.argmin(Sc)
    R2,xc = circhyp(xi[:,tri[ind,:]],n)
    # Notice!! ind_min may have a problen as an index
    x = np.copy(xi[:,ind_min-1])
    xml,yml = Constant_K_Search(x,inter_par,xc,R2,K)
    if yml < ym:
        xm = yml
        ym = yml
    return xm,ym
        
def Constant_K_Search(x0,inter_par,xc,R2,K):
#    This funciron minimizes the search funciton in the specified simplex with xc as circumcenter of that simplex and R2 as the circumradius of that simplex
#   the search funciton is: s(x) = p(x) - K e(x)
#   where the p(x) is the surrogate model: usually polyharmonic spline (RBF) phi = r^3
#   the artificali uncertatintiy fucniton is：e(x) = R2-norm(x-xc)
#   K: is a constant paramtere that specifies a tradeoff bwtween gloabl exploration (e - K large) and local refinemnet (p - K small) 
#   K is dependant on the mesh size. Its changes is proporstional  to the inverse of the rate as mesh size. 
#    Initially the algorithm tends to explore globally. and as the algorithm procceeds it becomes dense at the position of a global minimizer.
    global lb,ub
    costfun,costjac = lambda x:Contious_search_cost(x,inter_par,xc,R2,K)
    opt = {'disp':True}
    res = optimize.minimize(costfun,x0,jac=costjac,method='SLSQP',bounds,options=opt)
    x = res.x
    y = res.fun
    return x,y
    
def Contious_search_cost(x,inter_par,xc,R2,K):
    M = interpolate_val(x,inter_par) - K*(R2 - np.linalg.norm(x-xc)**2)
    num_arguments = expecting()
    if num_arguments > 1:
        DM = interpolate_grad(x,inter_par) + 2*K*(x-xc)
        return M,DM
    return M
#%%

t2 = np.arange(1,4)
val = np.amin(t2)
val2 = np.min(t2)
pos = np.argmin(t2)
print(val2,pos)
#%%
def circhyp(x, N):
    # circhyp     Circumhypersphere of simplex
    #   [xC, R2] = circhyp(x, N) calculates the coordinates of the circumcenter
    #   and the square of the radius of the N-dimensional hypersphere
    #   encircling the simplex defined by its N+1 vertices.
    #   Author: Shahoruz Alimohammadi
    #   Modified: Jan., 2017
    #   DELTADOGS package

    test = np.sum(np.transpose(x) ** 2, axis=1)
    test = test[:, np.newaxis]
    m1 = np.concatenate((np.matrix((x.T ** 2).sum(axis=1)), x))
    M = np.concatenate((np.transpose(m1), np.matrix(np.ones((N + 1, 1)))), axis=1)
    a = np.linalg.det(M[:, 1:N + 2])
    c = (-1.0) ** (N + 1) * np.linalg.det(M[:, 0:N + 1])
    D = np.zeros((N, 1))
    for ii in range(N):
        M_tmp = np.copy(M)
        M_tmp = np.delete(M_tmp, ii + 1, 1)
        D[ii] = ((-1.0) ** (ii + 1)) * np.linalg.det(M_tmp)
        # print(np.linalg.det(M_tmp))
    # print(D)
    xC = -D / (2.0 * a)
    #	print(xC)
    R2 = (np.sum(D ** 2, axis=0) - 4 * a * c) / (4.0 * a ** 2)
    #	print(R2)
    return R2, xC
#%%
def interpolate_val(x, inter_par):
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi

        S = xi - x
        #             print np.dot(v.T,np.concatenate([np.ones((1,1)),x],axis=0)) + np.dot(w.T,np.sqrt(np.diag(np.dot(S.T,S))))**3
        return np.dot(v.T, np.concatenate([np.ones((1, 1)), x], axis=0)) + np.dot(w.T, (np.sqrt(np.diag(np.dot(S.T, S))) ** 3))
#%%
def interpolate_grad(x, inter_par):
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        n = x.shape[0]
        N = xi.shape[1]
        g = np.zeros((n))
        for ii in range(N):
            X = x[:, 0] - xi[:, ii]
            g = g + 3 * w[ii] * X.T * np.linalg.norm(X)
        # print("--------------")
        #                 print v[ii]
        #             print(g)
        #             print("--------------")
        #             print(v[1:])
        g = g + v[1:, 0]

        return g.T
#%%
import inspect,dis

def expecting():
    """Return how many values the caller is expecting"""
    f = inspect.currentframe()
    f = f.f_back.f_back
    c = f.f_code
    i = f.f_lasti
    bytecode = c.co_code
    instruction = bytecode[i+3]
    if instruction == dis.opmap['UNPACK_SEQUENCE']:
        howmany = bytecode[i+4]
        return howmany
    elif instruction == dis.opmap['POP_TOP']:
        return 0
    return 1
#%%
inter_par = Inter_par("NPS")
xi = np.array([[0.5,0.8,0.2,0.6]])
fun = lambda x:  np.multiply(x-0.45,x-0.45)
yi = fun(xi)
print(yi)
K = 3
inter_par = interpolateparameterization(xi,yi,inter_par)
#ymin = np.min(yi)
#ind_min = np.argmin(yi)
#xm,ym = tringulation_search_bound_constantK(inter_par,xi,K,ind_min)

#%%
import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt
class Inter_par():
    def __init__(self,method="NPS", w=0, v=0, xi=0,a=0):
        self.method = "NPS"
        self.w=[]
        self.v=[]
        self.xi=[]
        self.a=[]
        
#def interpolateparameterization(xi, yi, inter_par):
#    n = xi.shape[0]
#    m = xi.shape[1]
#    if inter_par.method =='NPS':
#        A= np.zeros(shape=(m,m))
#        for ii in range(0,m,1): # for ii =0 to m-1 with step 1; range(1,N,1)
#            for jj in range(0,m,1):
#                A[ii,jj] = (np.dot(xi[:,ii] - xi[:,jj],xi[:,ii] - xi[:,jj]))**(3.0/2.0)
#
#        V = np.concatenate((np.ones((1,m)), xi), axis=0)
#        A1 = np.concatenate((A, np.transpose(V)),axis=1)
#        A2 = np.concatenate((V, np.zeros(shape=(n+1,n+1) )), axis=1)
#        yi = yi[np.newaxis,:]
#        # print(yi.shape)
#        b = np.concatenate([np.transpose(yi), np.zeros(shape=(n+1,1))])
#        #      b = np.concatenate((np.transpose(yi), np.zeros(shape=(n+1,1) )), axis=0)
#        A = np.concatenate((A1,A2), axis=0)
#        wv = np.linalg.solve(A,b)
#        inter_par.w = wv[:m]
#        inter_par.v = wv[m:]
#        inter_par.xi = xi
#        return inter_par
#        #      print(V)

def fun(x,  alpha=0.01):
    y = np.array((x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0)
    return y.T
#    return (x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0
