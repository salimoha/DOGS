import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;
pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False

import math
from scipy import optimize


def emprical_sigma(x, s):
    N = len(x)
    sigma = np.zeros(len(s))
    for jj in range(len(s)):
        mu = np.zeros([N // s[jj]])
        for i in range(N // s[jj]):
            inds = np.arange(i * s[jj], (i + 1) * s[jj])
            mu[i] = (x[np.unravel_index(inds, x.shape, 'F')]).mean(axis=0)
        sigma[jj] = (mu ** 2).mean(axis=0)
    return sigma


def stationary_statistical_learning_reduced(x, m):
    N = len(x)
    M = math.floor(math.sqrt(N))
    s = np.arange(1, 2 * M + 1)
    variance = x.var(axis=0) * len(x) / (len(x) - 1)
    x = np.divide(x - x.mean(axis=0), ((x.var(axis=0)) * len(x) / (len(x) - 1)) ** (1 / 2))

    sigmac2 = emprical_sigma(x, s)
    tau = np.arange(1, m + 1) / (m + 1)

    fun = lambda tau: Loss_fun_reduced(tau, sigmac2, 1)
    jac = lambda tau: Loss_fun_reduced(tau, sigmac2, 2)
    theta = np.zeros([2 * m])
    bnds = tuple([(0, 1) for i in range(int(m))])
    opt = {'disp': False}
    res_con = optimize.minimize(fun, tau, jac=jac, method='L-BFGS-B', bounds=bnds, options=opt)
    theta_tau = np.copy(res_con.x)

    theta_A = optimum_A(theta_tau, sigmac2)
    theta = np.concatenate((theta_A, theta_tau), axis=0)
    moment2_model, corr_model = Thoe_moment2(np.concatenate((np.array([1]), np.array([0]), theta), axis=0), N)

    sigma2_N = moment2_model[-1] * variance
    print("---------============--------------")
    print("moment2 = ", moment2_model[-1])
    print("var = ", variance)
    print("theta =", theta)
    print("sigma2N = ", sigma2_N)
    print("---------============--------------")
    return sigma2_N, theta, moment2_model


def Loss_fun_reduced(tau, sigmac2, num_arg):
    #   This function minimizes the linear part of the loss function that is a least square fit for the varaince of time averaging errror
    #   This is done using alternative manimization as fixing the tau value fixed.
    #   Autocorrelation function is rho = A_1 tau_1^k + ... +A_m tau_m^k
    #   If num_arg == 1, return L
    #   If num_arg == 2, return DL
    #   If num_arg == 3, return L,DL
    L = np.array([0])
    m = len(tau)
    H = np.zeros([m, m])
    Ls = np.zeros([m])
    DL = np.zeros([m])
    for ss in range(1, len(sigmac2) + 1):
        for ii in range(m):
            as_ = np.arange(1, ss + 1)
            Ls[ii] = 1 / ss * (1 + 2 * np.dot(1 - np.divide(as_, ss), tau[ii] ** as_.reshape(-1, 1))) - sigmac2[ss - 1]
            DL[ii] = 2 * Ls[ii] * 2 / ss * (
            np.dot((as_ - np.power(as_, 2) / ss), np.power(tau[ii], (as_ - 1).reshape(-1, 1))))
        H = H + np.multiply(Ls.reshape(-1, 1), Ls)

    # H1 = np.concatenate((H, -1*np.ones([m,1])),axis=1)
    #    H2 = np.concatenate((np.ones([1,m]),np.ones([1,1])*0),axis=1)
    #    Ah = np.vstack((H1,H2))
    #    if np.cond(Ah) > 1000:
    #    b = np.vstack(( 0*np.ones([m,1]),1))
    #
    #    A_lambda = np.dot(np.linalg.pinv(Ah),b)
    #    A = np.copy(A_lambda[:m])
    #    L = 0.5 * np.dot(A.T, np.dot(H , A))
    #    else:
    fun = lambda x: 0.5 * np.dot(x.T, np.dot(H, x))
    jac = lambda x: np.dot(x.T, H)
    Aineq = np.identity(m)
    bineq = np.zeros([m])
    Aeq = np.ones([1, m])
    beq = np.ones([1])
    cons = ({'type': 'ineq', 'fun': lambda x: bineq + np.dot(Aineq, x), 'jac': lambda x: Aineq},
            {'type': 'eq', 'fun': lambda x: beq - np.dot(Aeq, x), 'jac': lambda x: -Aeq})
    x0 = np.ones([m, 1]) / m
    res = optimize.minimize(fun, x0, jac=jac, constraints=cons, method='SLSQP')
    L = res.fun
    DL = np.multiply(DL, res.x)
    if num_arg == 1:
        return L
    elif num_arg == 2:
        return DL
    else:
        return L, DL


def optimum_A(tau, sigmac2):
    # tau need to be a vector
    m = len(tau)
    H = np.zeros([m, m])
    Ls = np.zeros([m])
    DL = np.zeros([m])
    for ss in range(1, len(sigmac2) + 1):
        for ii in range(m):
            as_ = np.arange(1, ss + 1)
            Ls[ii] = 1 / ss * (1 + 2 * np.dot(1 - np.divide(as_, ss), tau[ii] ** as_.reshape(-1, 1))) - sigmac2[ss - 1]
            DL[ii] = 2 * Ls[ii] * 2 / ss * (
            np.dot((as_ - np.power(as_, 2) / ss), np.power(tau[ii], (as_ - 1).reshape(-1, 1))))
        H = H + np.multiply(Ls.reshape(-1, 1), Ls)
    c = np.zeros([m])
    A_ineq = np.identity(m)
    b_ineq = np.zeros([m])
    A_eq = np.ones([1, m])
    b_eq = np.array([1])
    x0 = np.ones([m, 1]) / m
    func = lambda x: 0.5 * np.dot(x.T, np.dot(H, x)) + np.dot(c, x)
    jaco = lambda x: np.dot(x.T, H) + c
    cons = ({'type': 'ineq',
             'fun': lambda x: b_ineq + np.dot(A_ineq, x),
             'jac': lambda x: A_ineq},
            {'type': 'eq',
             'fun': lambda x: b_eq - np.dot(A_eq, x),
             'jac': lambda x: A_eq})
    opt = {'disp': False}
    A = optimize.minimize(func, x0, jac=jaco, constraints=cons, method='SLSQP', options=opt)
    return A.x


def Thoe_moment2(theta, N):
    # Equaiton (?) in the paper
    m = int(len(theta) / 2 - 1)
    corr_model = exponential_correlation(theta[2:m + 2], theta[m + 2:], N)
    s = np.arange(1, N + 1).reshape(-1, 1)
    moment2 = theta[1] + theoritical_sigma(corr_model.reshape(-1, 1), s, theta[0])
    return moment2, corr_model


def exponential_correlation(A, tau, N):
    # exponential correlation model
    # sum A_i tau^k
    corr = np.zeros([1, N])
    for ii in range(1, len(tau) + 1):
        corr = corr + A[np.unravel_index(ii - 1, A.shape, 'F')] * np.power(tau[ii - 1], np.arange(1, N + 1))
    return corr


def theoritical_sigma(corr,s,sigma02):
    sigma = np.zeros([len(s)*1.0])
    for ii in range(int(len(s))):
        sigma[ii] = 1.0
        for jj in range(1,int(s[ii])):
            sigma[ii] = sigma[ii] + 2*(1-jj/s[ii])*corr[jj-1]
        sigma[ii] = sigma02 * sigma[ii] / s[ii]
    return sigma