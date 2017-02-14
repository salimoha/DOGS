# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:10:13 2017

@author: KimuKook
"""
import pandas as pd
import numpy as np
import math
from scipy.spatial import Delaunay


# %%
def circhyp(x, N):
    # circhyp     Circumhypersphere of simplex
    #   [xC, R2] = circhyp(x, N) calculates the coordinates of the circumcenter
    #   and the square of the radius of the N-dimensional hypersphere
    #   encircling the simplex defined by its N+1 vertices.
    #   Modified: Jan., 2017


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


def tringulation_search_bound_constantK(inter_par, xi, K, ind_min):
    n = xi.shape[0]
    # [R2, xC] = utl.circhyp(xi[:, tri.simplices[ind, :]], n)
    tri = Delaunay(xi.T)  # fix for 1D
    Sc = np.zeros([np.shape(tri.simplices)[0]])
    Scl = np.zeros([np.shape(tri.simplices)[0]])
    for ii in range(np.shape(tri.simplices)[0]):
        R2, xc = circhyp(xi[:, tri.simplices[ii, :]], n)
        x = np.dot(xi[:, tri.simplices[ii, :]], np.ones([n + 1, 1]) / (n + 1))
        Sc[ii] = interpolate_val(x, inter_par) - K * (R2 - np.linalg.norm(x - xc) ** 2)
        if np.sum(ind_min == tri.simplices[ii, :]):
            Scl[ii] = Sc[ii]
        else:
            Scl[ii] = np.inf
    # Global one    
    t = np.min(Sc)
    ind = np.argmin(Sc)
    R2, xc = circhyp(xi[:, tri.simplices[ind, :]], n)
    x = np.dot(xi[:, tri.simplices[ind, :]], np.ones([n + 1, 1]) / (n + 1))
    xm, ym = Constant_K_Search(x, inter_par, xc, R2, K)
    # Local one
    t = np.min(Sc)
    ind = np.argmin(Sc)
    R2, xc = circhyp(xi[:, tri.simplices[ind, :]], n)
    # Notice!! ind_min may have a problen as an index
    x = np.copy(xi[:, ind_min - 1])
    xml, yml = Constant_K_Search(x, inter_par, xc, R2, K)
    if yml < ym:
        xm = np.copy(xml)
        ym = np.copy(yml)
    return xm.reshape(-1, 1), ym


def Constant_K_Search(x0, inter_par, xc, R2, K, lb=[], ub=[]):
    #    This funciron minimizes the search funciton in the specified simplex with xc as circumcenter of that simplex and R2 as the circumradius of that simplex
    #   the search funciton is: s(x) = p(x) - K e(x)
    #   where the p(x) is the surrogate model: usually polyharmonic spline (RBF) phi = r^3
    #   the artificali uncertatintiy fucniton is：e(x) = R2-norm(x-xc)
    #   K: is a constant paramtere that specifies a tradeoff bwtween gloabl exploration (e - K large) and local refinemnet (p - K small)
    #   K is dependant on the mesh size. Its changes is proporstional  to the inverse of the rate as mesh size.
    #    Initially the algorithm tends to explore globally. and as the algorithm procceeds it becomes dense at the position of a global minimizer.
    #     global lb,ub
    #     costfun,costjac = lambda x:Contious_search_cost(x,inter_par,xc,R2,K)
    n = x0.shape[0]
    costfun = lambda x: Contious_search_cost(x, inter_par, xc, R2, K)
    costjac = lambda x: Contious_search_cost_grad(x, inter_par, xc, R2, K)
    opt = {'disp': True}
    # TODO: boundas 0 to 1 all dimetnsions.. fix with lb and ub
    bnds = tuple([(0, 1) for i in range(int(n))])
    x00 = x0
    x0 = pd.DataFrame(x00).values
    # TODO: the output of minimize fucntion is np array (n,). For interpolte_val the input is (n,1)
    # TODO:  S=xi-x has problem in side this function
    # TODO: fix the input information for jacobi!!!!!!!!!!
    res = optimize.minimize(costfun, x0, method='L-BFGS-B', bounds=bnds, options=opt)
    x = res.x
    y = res.fun
    return x, y


# gradient of soncstant K search
def Contious_search_cost_grad(x, inter_par, xc, R2, K):
    DM = interpolate_grad(x, inter_par).reshape(-1, 1) + 2 * K * (x - xc)
    dm = pd.DataFrame(DM)
    return DM.T
    # return dm.values


# value of consatn K search
def Contious_search_cost(x, inter_par, xc, R2, K):
    M = interpolate_val(x, inter_par) - K * (R2 - np.linalg.norm(x - xc) ** 2)
    return M


# Muhan-->implementaiton #TODO
# def Contious_search_cost(x,inter_par,xc,R2,K):
#     M = interpolate_val(x,inter_par) - K*(R2 - np.linalg.norm(x-xc)**2)
#     num_arguments = expecting()
#     if num_arguments > 1:
#         DM = interpolate_grad(x,inter_par) + 2*K*(x-xc)
#         return M,DM
#     return M
# #%%

# %%
def interpolate_val(x, inter_par):
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        try:
            S = xi - x
            return np.dot(v.T, np.concatenate([np.ones((1, 1)), x], axis=0)) + np.dot(w.T, (
            np.sqrt(np.diag(np.dot(S.T, S))) ** 3))
        except:
            S = xi - np.tile(x.reshape(-1, 1), xi.shape[1])
            return np.dot(v.T, np.concatenate([np.ones(1), x], axis=0).reshape(-1, 1)) + np.dot(w.T, (
            np.sqrt(np.diag(np.dot(S.T, S))) ** 3))


# %%
def interpolate_grad(x, inter_par):
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        n = x.shape[0]
        N = xi.shape[1]
        g = np.zeros((n))
        x1 = np.copy(x)
        x = pd.DataFrame(x1).values
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


# %%
import inspect, dis


def expecting():
    """Return how many values the caller is expecting"""
    f = inspect.currentframe()
    f = f.f_back.f_back
    c = f.f_code
    i = f.f_lasti
    bytecode = c.co_code
    instruction = bytecode[i + 3]
    if instruction == dis.opmap['UNPACK_SEQUENCE']:
        howmany = bytecode[i + 4]
        return howmany
    elif instruction == dis.opmap['POP_TOP']:
        return 0
    return 1


# %%





inter_par = Inter_par("NPS")
xi = np.array([[0.5, 0.8, 0.2, 0.6]])
fun = lambda x: np.multiply(x - 0.45, x - 0.45)
yi = fun(xi)
print(yi)
K = 3
inter_par = interpolateparameterization(xi, yi, inter_par)
ymin = np.min(yi)
ind_min = np.argmin(yi)
xm, ym = tringulation_search_bound_constantK(inter_par, xi, K, ind_min)


# %%



def fun(x, alpha=0.1):
    y = np.array((x[0, :] - 0.45) ** 2.0 + alpha * (x[1, :] - 0.45) ** 2.0)
    return y.T


#    return (x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0


inter_par = Inter_par("NPS")
xi = np.array([[0.5000, 0.8000, 0.5000, 0.2000, 0.5000], [0.5000, 0.5000, 0.8000, 0.5000, 0.2000]])
# xi=np.random.rand(2,3)
# x=np.array([[0.5],[0.5]])
# yi=np.random.rand(1,3)
yi = fun(xi)
print (yi)
# yi = np.array(yi)
# print(yi.shape)
# print(xi.shape)
#
inter_par = interpolateparameterization(xi, yi, inter_par)

ymin = np.min(yi)
ind_min = np.argmin(yi)
xm, ym = tringulation_search_bound_constantK(inter_par, xi, K, ind_min)


# %%
def test(x, y):
    n_a = expecting()
    if n_a > 1:
        return x + y, x - y
    return x * y


c1, s1 = test(11, 2)
print('c1', c1, 's1', s1)
# ft = lambda x:test(x,2)
# c,s = ft(11)
# print(c)