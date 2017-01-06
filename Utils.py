import numpy as np

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


def bounds(bnd1,bnd2,n):
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
	bnds = np.matlib.repmat(bnd2,  1, 2**n)
	for ii in range(n):
	    tt = np.mod(np.arange(2**n)+1, 2**(n-ii)) <= 2**(n-ii-1)-1
	    bnds[ii, tt] = bnd1[ii];
	return bnds



def mindis(x,xi):
# function [y,x1,index] = mindistance(x,xi)
# % calculates the minimum distance from all the existing points
# % xi all the previous points
# % x the new point
    y=float('inf')
    N=xi.shape[1]
    for i in range(N):
        y1 = np.linalg.norm(x[:,0]-xi[:,i])
        if y1<y:
            y=np.copy(y1)
            x1 = np.copy(xi[:,i])
            index=np.copy(i)
    return y,index,x1


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
