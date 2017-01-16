import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt


import numpy           as np
import scipy.sparse    as sp
from   optimize.snopt7 import SNOPT_solver
from scipy.optimize import minimize, rosen, rosen_der


class Inter_par():
    def __init__(self, method="NPS", w=0, v=0, xi=0, a=0):
        self.method = "NPS"
        self.w = []
        self.v = []
        self.xi = []
        self.a = []


def interpolateparameterization(xi, yi, inter_par):
    xi = xi.values
    yi = yi.values
    n = xi.shape[0]
    m = xi.shape[1]
    if inter_par.method == 'NPS':
        A = np.zeros(shape=(m, m))
        for ii in range(0, m, 1):  # for ii =0 to m-1 with step 1; range(1,N,1)
            for jj in range(0, m, 1):
                A[ii, jj] = (np.dot(xi[:, ii] - xi[:, jj], xi[:, ii] - xi[:, jj])) ** (3.0 / 2.0)

        V = np.concatenate((np.ones((1, m)), xi), axis=0)
        A1 = np.concatenate((A, np.transpose(V)), axis=1)
        A2 = np.concatenate((V, np.zeros(shape=(n + 1, n + 1))), axis=1)
        # yi = yi[np.newaxis, :]
        # print(yi.shape)
        # b = np.concatenate([np.transpose(yi), np.zeros(shape=(n + 1, 1))])
        # b = np.concatenate([yi, np.zeros(shape=(n + 1, 1))])
        b = np.concatenate((np.transpose(yi), np.zeros(shape=(n+1,1) )), axis=0)
        A = np.concatenate((A1, A2), axis=0)
        wv = np.linalg.solve(A, b)
        inter_par.w = wv[:m]
        inter_par.v = wv[m:]
        inter_par.xi = pd.DataFrame(xi)
        # inter_par.xi = xi
    return inter_par


#      print(V)

def interpolate_val(x, inter_par):
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi.values
        x = pd.DataFrame(x)
        # xi = inter_par.xi
        # print("-------------")
        # print(x)
        # x = np.expand_dims(x, axis=1)
        S = xi - x.values
        #             print np.dot(v.T,np.concatenate([np.ones((1,1)),x],axis=0)) + np.dot(w.T,np.sqrt(np.diag(np.dot(S.T,S))))**3
        return np.dot(v.T, np.concatenate([np.ones((1, 1)), x], axis=0)) + np.dot(w.T, (
        np.sqrt(np.diag(np.dot(S.T, S))) ** 3))


def interpolate_grad(x, inter_par):
    # calculating the gradinet of the interpolation
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi.values
        x=pd.DataFrame(x)
        x = x.values
        n = x.shape[0]
        N = xi.shape[1]
        g = np.zeros((n))
        for ii in range(N):
            X = x[:, 0] - xi[:, ii]
            g = g + 3 * w[ii] * X.T * np.linalg.norm(X)

        g = g + v[1:, 0]
        G = pd.DataFrame(g)
        return G.values
    else:
        Warning("Interpolation method has not been implimented yet....!!!!")


def interpolate_hessian(x, inter_par):
    if inter_par.method == "NPS" or self.method == 1:
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi.values
        n = x.shape[0]
        N = xi.shape[1]
        g = np.zeros((n))
        n = x.shape[0]

        H = np.zeros((n, n))
        for ii in range(N):
            X = x[:, 0] - xi[:, ii]
            if np.linalg.norm(X) > 1e-5:
                H = H + 3 * w[ii] * ((X * X.T) / np.linalg.norm(X) + np.linalg.norm(X) * np.identity(n))
        return H


def fun(x, alpha=0.01):
    y = np.array((x[0, :] - 0.45) ** 2.0 + alpha * (x[1, :] - 0.45) ** 2.0)
    Y=pd.DataFrame(yi)
    return Y.values

def inter_min(x,inter_par, Ain=[], bin=[]):
    # %find the minimizer of the interpolating function starting with x
    rho = 0.9 # backtracking paramtere
    n = x.shape[0]
#     start the serafh method
    iter = 0
    # x = pd.DataFrame((x))
    x0 = np.zeros((n, 1))
    # while iter < 10:
    H = np.zeros((n, n))
    g = np.zeros((n, 1))
    y = interpolate_val(x, inter_par)
    g = interpolate_grad(x, inter_par)
    #H = interpolate_hessian(x, inter_par)
        # Perform the Hessian modification
    #H = modichol(H, 0.01, 20);
    #H = (H + H.T)/2.0
#         optimizaiton for finding hte right direction
    objfun3 = lambda x: (interpolate_val(x, inter_par))
    grad_objfun3 = lambda x:  interpolate_grad(x, inter_par)
    res = minimize(objfun3, x0, method='L-BFGS-B', jac=grad_objfun3, options={'gtol': 1e-6, 'disp': True})
    return res.x, res.fun

#    return (x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0

def sntoya_objF(status, x):
    F = np.array([x[1],  # objective row
                  x[0] ** 2 + 4.0 * x[1] ** 2,
                  (x[0] - 2.0) ** 2 + x[1] ** 2])
    # objfun = lambda x: costSearch(x, inter_par, xc, R2, y0)

    return status, F


def sntoya_objFG(status, x):
    F = np.array([x[1],  # objective row
                  x[0] ** 2 + 4.0 * x[1] ** 2,
                  (x[0] - 2.0) ** 2 + x[1] ** 2])

    G = np.array([2 * x[0], 8 * x[1], 2 * (x[0] - 2), 2 * x[1]])
    return status, F, G


#
# function [x y CS]=Constant_K_Search_snopt(x,inter_par_, inter_Par_,xc_,R2_,Search)
# % This funciton finds the minimizer of the search function i a simplex
# %              minimize s^k_i(x) = p^k(x) - K e^k_i(x)
# %             subject to CS^k_(l,i) = g^k_l(x) - K e^k_i(x) <= 0
# %                        A x <= b
# % input:
# % x: interpolating initial point for search function
# % inter_par: interpolation parameters
# % xc: circumcenter for the simplex that x is located at
# % R2: square of circumradiaus for the simplex that x is located at
# % output:
# % x: the minimizer of search function
# % y: minimum of search fucntion at x
# % cse:
# % created by: Shahrouz Alimo & Pooriya Beyhaghi
# % last modification: Oct/7/2015
# %

# function [x,y] = Adoptive_K_Search_new(x,inter_par,xc,R2,y0)
# % Find the minimizer of the search function in a simplex
#
# fun=@(x) cost(x,inter_par,xc,R2,y0);
# options = optimoptions(@fmincon,'Algorithm','sqp','GradObj','On','DerivativeCheck','Off');
# [x,y] = fmincon(fun,x,[],[],[],[],x*0, x*0+1,[], options);
# end
#

# x = pd.DataFrame(x)

# Search function for adaptive K
def costSearch(x,inter_par,xc,R2,y0):
    x = pd.DataFrame(x)
    p = interpolate_val(x,inter_par)
    x = x.values
    e = R2 - np.dot((x-xc).T, (x-xc))
    Mm = -e*1.0/(p-y0)
    M = pd.DataFrame(Mm).values
    if p < y0:
        M=float("inf")
    return M

def kgradSearch(x,inter_par,xc,R2,y0):
    # gets x as a np.array element
    x = pd.DataFrame(x)
    p = interpolate_val(x,inter_par)
    gp = interpolate_grad(x, inter_par)
    # x = x.values
    e = R2 - np.dot((x-xc).T, (x-xc))
    ge = -2 * (x - xc)
    ge=ge.values
    # gp = np.expand_dims(gp, axis=1)
    DMm = -ge / (p - y0) + e * gp / (p - y0)**2.0
    DM = pd.DataFrame(DMm)
    if p < y0:
        DM = gp * 0.0
    return DM.values.T


def Adoptive_K_Search(x, inter_par, xc, R2, y0):
    objfun = lambda x: costSearch(x,inter_par,xc,R2,y0)
    grad_objfun  = lambda x: kgradSearch(x,inter_par,xc,R2,y0)
    y1 = objfun(x)
    res = minimize(objfun, x0, method='L-BFGS-B', jac=grad_objfun, options={'gtol': 1e-6, 'disp': True})

    return res.x, res.fun


xiT = hstack((xU,xE))
def  tringulation_search_bound(inter_par,xiT,ind_min,y0):
    xiT  = pd.DataFrame(xiT)
    xp = pd.DataFrame(xiT[ind_min])
    xp = xp.values
    [xm,ym] = inter_min(xp, inter_par)


    return xm, ym, cse


# function[xm
# ym
# cse] = tringulation_search_bound(inter_par, xiT, ind_min)
#
# global n
# y0
# % keyboard
# [xm, ym] = inter_min(xiT(:, ind_min), inter_par);
# if (ym > y0)
#     ym = inf;
#     cse = 2;
# tri = delaunayn(xiT.
# ');
# for ii=1:size(tri, 1)
# [xc, R2] = circhyp2(xiT(:, tri(ii,:)), n);
# if R2~=inf
# % initialization with body center of each simplex
# x=xiT(:,
#     tri(ii,:))*ones(n + 1, 1) / (n + 1);
# % [x, ym] = Adoptive_K_Search(x, inter_par, xc, R2);
# Sc(ii) = (interpolate_val(x, inter_par) - y0) / (R2 - norm(x - xc) ^ 2);
# Scl(ii) = Sc(ii);
# if ismember(ind_min, tri(ii,:))~ = 1
# Scl(ii) = inf;
# end
# else
# Sc(ii) = inf;
# Scl(ii) = inf;
# end \
# % [x, ym] = Adoptive_K_Search(x, inter_par, xc, R2);
# end
# [t, ind] = min(Sc);
# [xc, R2] = circhyp(xiT(:, tri(ind,:)), n);
# x = xiT(:, tri(ind,:))*ones(n + 1, 1) / (n + 1);
# % keyboard
# [xm, ymg] = Adoptive_K_Search_new(x, inter_par, xc, R2, y0);
# [t, ind] = min(Scl);
# [xc, R2] = circhyp(xiT(:, tri(ind,:)), n);
# x = xiT(:, tri(ind,:))*ones(n + 1, 1) / (n + 1);
# [xml, yml] = Adoptive_K_Search_new(x, inter_par, xc, R2, y0);
# if yml < 2 * ymg
# xm = xml;
# end
# end
#
# end
#
#




