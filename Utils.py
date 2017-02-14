import numpy as np
from scipy import optimize
np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt





def modichol(A, alpha, beta):
    n = A.shape[1]  # size of A
    L = np.identity(n)
    ####################
    D = np.zeros((n, 1))
    c = np.zeros((n, n))
    ######################
    D[0] = np.max(np.abs(A[0, 0]), alpha)
    c[:, 0] = A[:, 0]
    L[1:n, 0] = c[1:n, 0] / D[0]

    for j in range(1, n - 1):
        c[j, j] = A[j, j] - (np.dot((L[j, 0:j] ** 2).reshape(1, j), D[0:j]))[0, 0]
        for i in range(j + 1, n):
            c[i, j] = A[i, j] - (np.dot((L[i, 0:j] * L[j, 0:j]).reshape(1, j), D[0:j]))[0, 0]
        theta = np.max(c[j + 1:n, j])
        D[j] = np.array([(theta / beta) ** 2, np.abs(c[j, j]), alpha]).max()
        L[j + 1:n, j] = c[j + 1:n, j] / D[j]
    j = n - 1;
    c[j, j] = A[j, j] - (np.dot((L[j, 0:j] ** 2).reshape(1, j), D[0:j]))[0, 0]
    D[j] = np.max(np.abs(c[j, j]), alpha)
    return np.dot(np.dot(L, (np.diag(np.transpose(D)[0]))), L.T)







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


class Inter_par():
    def __init__(self, method="NPS", w=0, v=0, xi=0, a=0):
        self.method = "NPS"
        self.w = []
        self.v = []
        self.xi = []
        self.a = []


def interpolateparameterization(xi, yi, inter_par):
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
        yi = yi[np.newaxis, :]
        # print(yi.shape)
        b = np.concatenate([np.transpose(yi), np.zeros(shape=(n + 1, 1))])
        #      b = np.concatenate((np.transpose(yi), np.zeros(shape=(n+1,1) )), axis=0)
        A = np.concatenate((A1, A2), axis=0)
        wv = np.linalg.solve(A, b)
        inter_par.w = wv[:m]
        inter_par.v = wv[m:]
        inter_par.xi = xi
        return inter_par
        #      print(V)

def regressionparametarization(xi,yi, sigma, inter_par):
    # Notice xi, yi and sigma must be a two dimension matrix, even if you want it to be a vector.
    # or there will be error
    n = xi.shape[0]
    N = xi.shape[1]
    if inter_par.method == 'NPS':
        A = np.zeros(shape=(N, N))
        for ii in range(0, N, 1):  # for ii =0 to m-1 with step 1; range(1,N,1)
            for jj in range(0, N, 1):
                A[ii, jj] = (np.dot(xi[:, ii] - xi[:, jj], xi[:, ii] - xi[:, jj])) ** (3.0 / 2.0)
        V = np.concatenate((np.ones((1, N)), xi), axis=0)
        w1 = np.linalg.lstsq((np.dot(np.diag(np.divide(1, sigma[0])), V.T)), np.divide(yi, sigma).T)
        w1 = np.copy(w1[0])
        b = np.mean(np.divide(np.dot(V.T,w1)-yi.reshape(-1,1),sigma)**2)
        wv = np.zeros([N+n+1])
        if b < 1:
            wv[N:] = np.copy(w1.T)
            rho = 1000
            wv = np.copy(wv.reshape(-1,1))
        else:
            rho = 1.1
            fun = lambda rho:smoothing_polyharmonic(rho,A,V,sigma,yi,n,N,1)
            sol = optimize.fsolve(fun,rho)
            b,db,wv = smoothing_polyharmonic(sol,A,V,sigma,yi,n,N,3)
        inter_par.w = wv[:N]
        inter_par.v = wv[N:]
        inter_par.xi = xi
        yp = np.zeros([N])
        while(1):
            for ii in range(N):
                yp[ii] = interpolate_val(xi[:,ii],inter_par)
            residual = np.max(np.divide(np.abs(yp-yi),sigma[0]))
            if residual < 2:
                break
            rho *= 0.9
            b,db,wv = smoothing_polyharmonic(rho,A,V,sigma,yi,n,N)
            inter_par.w = wv[:N]
            inter_par.v = wv[N:]
    return inter_par, yp
    

def smoothing_polyharmonic(rho, A, V, sigma, yi, n, N,num_arg):
    # Notice: num_arg = 1 will return b
    #         num_arg = else will return b,db,wv
    A01 = np.concatenate((A + rho * np.diag(sigma ** 2), np.transpose(V)), axis=1)
    A02 = np.concatenate((V, np.zeros(shape=(n + 1, n + 1))), axis=1)
    A1 = np.concatenate((A01, A02), axis=0)
    b1 = np.concatenate([yi.reshape(-1,1), np.zeros(shape=(n + 1, 1))])
    wv = np.linalg.solve(A1, b1)
    b = np.mean(np.multiply(wv[:N],sigma)**2*rho**2) - 1
    bdwv = np.concatenate([np.multiply(wv[:N],sigma.reshape(-1,1)**2), np.zeros((n + 1, 1))])
    Dwv = np.linalg.solve(-A1, bdwv)
    db = 2 * np.mean(np.multiply(wv[:N]**2*rho + rho**2*np.multiply(wv[:N],Dwv[:N]),sigma**2))
    if num_arg == 1:
        return b
    else:
        return b,db,wv
        
        
def interpolate_hessian(x, inter_par):
    if inter_par.method == "NPS" or self.method == 1:
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
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
    return y.T

#    return (x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0


def bounds(bnd1, bnd2, n):
    #   find vertex of domain for a box domain.
    #   INPUT: n: dimension, bnd1: lower bound, bnd2: upper bound.
    #   OUTPUT: vertex of domain. 2^n number vector of n-D.
    #   Example:
    #           n = 3
    #           bnd1 = np.zeros((n, 1))
    #           bnd2 = np.ones((n, 1))
    #           bnds = bounds(bnd1,bnd2,n)
    #   Author: Shahoruz Alimohammadi
    #   Modified: Dec., 2016
    #   DELTADOGS package
    bnds = np.kron(np.ones((1, 2 ** n)), bnd2)
    for ii in range(n):
        tt = np.mod(np.arange(2 ** n) + 1, 2 ** (n - ii)) <= 2 ** (n - ii - 1) - 1
        bnds[ii, tt] = bnd1[ii];
    return bnds


def mindis(x, xi):
    # function [y,x1,index] = mindistance(x,xi)
    # % calculates the minimum distance from all the existing points
    # % xi all the previous points
    # % x the new point
    y = float('inf')
    N = xi.shape[1]
    for i in range(N):
        y1 = np.linalg.norm(x[:, 0] - xi[:, i])
        if y1 < y:
            y = np.copy(y1)
            x1 = np.copy(xi[:, i])
            index = np.copy(i)
    return y, index, x1


def vertex_find(A, b, lb, ub):
    if len(lb) != 0:
        Vertex = np.matrix([[], []])
        m = A.shape[0]
        n = A.shape[1]
        if m == 0:
            Vertex = bounds(lb, ub, len(lb))
        else:
            for r in range(0, min(n, m) + 1):
                from itertools import combinations
                C = [c for c in combinations(range(1, m + 1), r)]
                C = [list(c) for c in C]
                D = [d for d in combinations(range(1, n + 1), n - r)]
                D = [list(d) for d in D]
                if r == 0:
                    F = np.array(bounds(lb, ub, n))
                    for kk in range(F.shape[1]):
                        x = np.copy(F[:, kk]).reshape(-1, 1)
                        if (np.dot(A, x) - b).min() < 1e-6:
                            Vertex = np.column_stack((Vertex, x))
                else:
                    for ii in range(len(C)):
                        index_A = np.copy(list(C[ii]))
                        v1 = [i for i in range(1, m + 1)]
                        index_A_C = np.setdiff1d(v1, index_A)
                        A1 = np.copy(A[index_A - 1, :])
                        b1 = np.copy(b[index_A - 1])
                        for jj in range(len(D)):
                            index_B = np.copy(list(D[jj]))
                            v2 = [i for i in range(1, n + 1)]
                            index_B_C = np.setdiff1d(v2, index_B)
                            F = bounds(lb[index_B - 1], ub[index_B - 1], n - r)
                            A11 = np.copy(A1[:, index_B - 1])
                            A12 = np.copy(A1[:, index_B_C - 1])
                            for kk in range(F.shape[1]):
                                A11 = np.copy(A1[:, index_B - 1])
                                A12 = np.copy(A1[:, index_B_C - 1])
                                xd = np.linalg.inv(A12) * (b1 - A11 * F[:, kk])
                                x = np.zeros((2)).reshape(-1, 1)
                                x[index_B - 1] = F[:, kk]
                                x[index_B_C - 1] = xd
                                if r == m or (np.dot(A[index_A_C - 1, :], x) - b[index_A_C - 1]).min() < 0:
                                    [y, x1, index] = mindis(x, Vertex)

                                    if (x - ub).max() < 1e-6 and (x - lb).min() > -1e-6 and y > 1e-6:
                                        Vertex = np.column_stack((Vertex, x))
                                        # Vertex = np.concatenate((Vertex, x),axis = 1)
                                        # print('------------!!!!-----------')
                                        # print(y)
                                        # print(x1)
                                        # print(index)
                                        print('--------------!!!!---------')
                                        # quit()
    else:
        m = A.shape[0]
        n = A.shape[1]
        from itertools import combinations
        C = [c for c in combinations(range(1, m), n)]
        C = [list(c) for c in C]
        Vertex = list()
        for ii in range(len(C)):
            index_A = np.copy(list(C[ii]))
            v1 = [i for i in range(1, m + 1)]
            index_A_C = np.setdiff1d(v1, index_A)
            A1 = np.copy(A[index_A - 1, :])
            b1 = np.copy(b[index_A - 1])
            A2 = np.copy(A[index_A_C - 1])
            b2 = np.copy(b[index_A_C - 1])
            x = np.rray(np.linalg.inv(A1) * b1)
            if (A2 * x - b2).max() < 1e-6:
                Vertex = np.column_stack((Vertex, x))
    return Vertex


from scipy.spatial import Delaunay



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

    def inter_min(x, inter_par, Ain=[], bin=[]):
        # %find the minimizer of the interpolating function starting with x
        rho = 0.9  # backtracking paramtere
        n = x.shape[0]
        #     start the serafh method
        iter = 0
        x0 = np.zeros((n, 1))
        # while iter < 10:
        H = np.zeros((n, n))
        g = np.zeros((n, 1))
        y = interpolate_val(x, inter_par)
        g = interpolate_grad(x, inter_par)
        # H = interpolate_hessian(x, inter_par)
        # Perform the Hessian modification
        # H = modichol(H, 0.01, 20);
        # H = (H + H.T)/2.0
        #         optimizaiton for finding hte right direction
        objfun3 = lambda x: (interpolate_val(x, inter_par))
        grad_objfun3 = lambda x: interpolate_grad(x, inter_par)
        res = minimize(objfun3, x0, method='L-BFGS-B', jac=grad_objfun3, options={'gtol': 1e-6, 'disp': True})
        return res.x, res.fun



def ismember(A,B):
    return [np.sum(a == B) for a in A]

def points_neighbers_find(x,xE,xU,Bin,Ain):
    [delta_general, index,x1] = mindis(x, np.concatenate((xE,xU ), axis=1) )
    
    active_cons = []
    b = Bin - np.dot(Ain,x)
    for i in range(len(b)):
        if b[i][0] < 1e-3:
            active_cons.append(i+1)
    active_cons = np.array(active_cons)
    
    active_cons1 = []
    b = Bin - np.dot(Ain,x1)
    for i in range(len(b)):
        if b[i][0] < 1e-3:
            active_cons1.append(i+1)
    active_cons1 = np.array(active_cons1)
    
    if len(active_cons) == 0 or min(ismember(active_cons,active_cons1)) == 1:
        newadd = 1
        success = 1
        if mindis(x,xU) == 0:
            newadd = 0
    else:
        success = 0
        newadd = 0
        xU = np.concatenate((xU,x),axis=0)
    return x, xE, xU, newadd, success	