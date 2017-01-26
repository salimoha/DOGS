# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 17:40:41 2017

@author: KimuKook
"""
import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt


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
                        print  index_A
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
                                        quit()
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


# %%
A = np.array([[1, 2]])
b = np.array([[1]])
lb = np.array([[0], [0]])
ub = np.array([[1], [1]])
# %%
V = vertex_find(A, b, lb, ub)
print(V)
# %%
V = np.matrix([[], []])
# %%
xi = np.matrix([[1, 0, 1, 0], [1, 1, 0, 0]])
# %%
V = np.matrix([[], []])
# %%
F = bounds(lb, ub, 2)
x1 = F[:, 1]
t2 = (A * x1) - b
t1 = np.dot(A, x1)
print(t1, t2)
# %%
V = np.matrix([[1, 0], [0, 0]])
x = np.matrix([[1], [0]])
[y, x1, index] = mindis(x, V)
print(y, y > 1e-6)

