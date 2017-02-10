import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt

from scipy import optimize

def regressionparametarization(xi,yi, sigma, inter_method):
    n = xi.shape[0]
    N = xi.shape[1]
    #  Import the class!!!!
    if inter_method == 'NPS':
        A = np.zeros(shape=(N, N))
        for ii in range(0, N, 1):  # for ii =0 to m-1 with step 1; range(1,N,1)
            for jj in range(0, N, 1):
                A[ii, jj] = (np.dot(xi[:, ii] - xi[:, jj], xi[:, ii] - xi[:, jj])) ** (3.0 / 2.0)

        V = np.concatenate((np.ones((1, N)), xi), axis=0)
#        A1 = np.concatenate((A, np.transpose(V)), axis=1)
#        A2 = np.concatenate((V, np.zeros(shape=(n + 1, n + 1))), axis=1)
#        yi = yi[np.newaxis, :]
#        # print(yi.shape)
#        b = np.concatenate([np.transpose(yi), np.zeros(shape=(n + 1, 1))])
#        #      b = np.concatenate((np.transpose(yi), np.zeros(shape=(n+1,1) )), axis=0)
#        A = np.concatenate((A1, A2), axis=0)
#        wv = np.linalg.solve(A, b)
#        inter_par.w = wv[:m]
#        inter_par.v = wv[m:]
#        inter_par.xi = xi
#        return inter_par

        # Muhan Modified
        w1 = np.linalg.solve((np.dot(np.diag(np.divide(1,sigma)),V.T)),np.divide(yi,sigma).reshape(-1,1))
        b = np.mean(np.divide(np.dot(V.T,w1)-yi.reshape(-1,1),sigma)**2)
        wv = np.zeros([N+n+1])
#        if b < 1:
#            wv[:N] = 0
#            wv[N+1:] = w1
#            rho = 1000
#            wv = wv.reshape(-1,1)
#        else:
        rho = 1.1
        print(rho)
        # TODO add another label to identify how many outputs you want.
        fun = lambda rho:smoothing_polyharmonic_fs(rho,A,V,sigma,yi,n,N)
        sol = optimize.fsolve(fun,rho)
        b,db,wv = smoothing_polyharmonic(sol,A,V,sigma,yi,n,N)
        print(b)
        print(wv)
#    return inter_par, yp
#%%
xi = np.array([[1,2,0,1],[2,1,4,2],[6,3,1,2]])
yi = np.array([2,4,6,8])
sigma = np.array([2,7,4,2])
inter_method = 'NPS'
regressionparametarization(xi,yi, sigma, inter_method)
#%%
def smoothing_polyharmonic(rho, A, V, sigma, yi, n, N):
    A01 = np.concatenate((A + rho * np.diag(sigma ** 2), np.transpose(V)), axis=1)
    A02 = np.concatenate((V, np.zeros(shape=(n + 1, n + 1))), axis=1)
    A1 = np.concatenate((A01, A02), axis=0)
    b1 = np.concatenate([yi.reshape(-1,1), np.zeros(shape=(n + 1, 1))])
    wv = np.linalg.solve(A1, b1)
    b = np.mean(np.multiply(wv[:N],sigma)**2*rho**2) - 1
    bdwv = np.concatenate([np.multiply(wv[:N],sigma.reshape(-1,1)**2), np.zeros((n + 1, 1))])
    Dwv = np.linalg.solve(-A1, bdwv)
    db = 2 * np.mean(np.multiply(wv[:N]**2*rho + rho**2*np.multiply(wv[:N],Dwv[:N]),sigma**2))
    return b, db, wv
def smoothing_polyharmonic_fs(rho, A, V, sigma, yi, n, N):
    A01 = np.concatenate((A + rho * np.diag(sigma ** 2), np.transpose(V)), axis=1)
    A02 = np.concatenate((V, np.zeros(shape=(n + 1, n + 1))), axis=1)
    A1 = np.concatenate((A01, A02), axis=0)
    b1 = np.concatenate([yi.reshape(-1,1), np.zeros(shape=(n + 1, 1))])
    wv = np.linalg.solve(A1, b1)
    b = np.mean(np.multiply(wv[:N],sigma)**2*rho**2) - 1
    bdwv = np.concatenate([np.multiply(wv[:N],sigma.reshape(-1,1)**2), np.zeros((n + 1, 1))])
    Dwv = np.linalg.solve(-A1, bdwv)
    db = 2 * np.mean(np.multiply(wv[:N]**2*rho + rho**2*np.multiply(wv[:N],Dwv[:N]),sigma**2))
    return b
    
    #%%
#
# function[inter_par, yp] = regressionparametarization(xi, yi,
# sigma, inter_method)
# n = size(xi, 1);
# N = size(xi, 2);
#
# % while 1
#     if inter_method == 1
#         % keyboard
#         A = zeros(N, N);
# % calculate
# regular
# A
# matrix
# for polyharmonic spline
#     for ii = 1: 1: N
# for jj = 1: 1: N
# A(ii, jj) = ((xi(:, ii) - xi(:, jj))' * (xi(:,ii) - xi(:,jj))) ^ (3 / 2);
# end
# end
#
# V = [ones(1, N);
# xi];
# w1 = diag(1. / sigma) * V
# '\(yi./sigma).';
# b = mean(((V
# '*w1-yi.')./ sigma
# ').^2);
#
# % keyboard
# if b < 1
#        % keyboard
# wv(1:N)=0;
# wv(N + 1:N + n + 1)=w1;
# rho = 1000;
# wv = wv.
# ';
# else
# rho = 1.1;
# options = optimset('Jacobian', 'on', 'display', 'none');
# fun =
#
#
# @(rho)
#
#
# smoothing_polyharmonic(rho, A, V, sigma, yi, n, N);
# rho = fsolve(fun, rho, options);
# [b, db, wv] = fun(rho);
#
# end
#
# inter_par
# {1} = 1;
# inter_par
# {2} = wv(1:N); inter_par
# {3} = wv(N + 1:N + n + 1);
# inter_par
# {4} = xi;
#
# while 1
#     for ii=1:N
#     yp(ii) = interpolate_val(xi(:, ii), inter_par);
#     end
#     if max(abs(yp - yi). / sigma) < 2
#         break
#     end
#     rho = rho * 0.9;
#     [b, db, wv] = smoothing_polyharmonic(rho, A, V, sigma, yi, n, N);
#     inter_par
#     {2} = wv(1:N); inter_par
#     {3} = wv(N + 1:N + n + 1);
#     end
#     end
#     % end
#     if inter_method == 2
#         inter_par = [];
#         yp = [];
#     disp(sprintf('Wrong Interpolation method'));
#     end
#     end
#
# clear
# all
# close
# all
# clc
#
# % check
# the
# regression
# n = 1;
# N = 6;
# fun =
#
#
# @(x)
#
#
# x. ^ 2;
# xi = [0 0.1 0.2 0.3 0.5 1];
# for ii=1:N
# yi_real(ii) = fun(xi(:, ii));
# end
# sigma = [0.01 0.2 0.01 0.2 0.01 0.2];
# V = randn(1, N). * sigma;
# yi = yi_real + V;
#
# rho = 0:0.001:0.5;
#
# [inter_par, yp] = regressionparametarization(xi, yi, sigma, 1);
#
# % figure(1)
# % plot(rho, weight)
#
# % [t, ind] = min(abs(weight - 1));
#
# xx = 0:0.1:1;
# for ii=1:length(xx)
# yp1(ii) = interpolate_val(xx(ii), inter_par);
# % yp2(ii) = interpolate_val(xx(ii), inter_par2);
# % yp3(ii) = interpolate_val(xx(ii), inter_par3);
# end
#
# figure(2)
# plot(xx, yp1, 'b--', xx, xx. ^ 2, 'k-')
# hold
# on
# plot(xx, xx. ^ 2, 'k-')
# hold
# on
# errorbar(xi, yi, sigma, '.')
# % xlim([0 1])
# #
# #     function[b, db, wv] = smoothing_polyharmonic(rho, A, V, sigma, yi, n, N)
# #     A1 = [A + rho * diag(sigma. ^ 2) V'; V zeros(n+1,n+1)];
# #           wv = A1 \ [yi.'; zeros(n+1,1)];
# # b=mean((wv(1:N).*sigma').^ 2 * rho ^ 2)-1;
# #     Dwv = -A1 \ [wv(1:N).*(sigma. ^ 2)
# #     ' ;zeros(n+1,1)];
# # db=2*mean((wv(1:N).^2*rho+rho^2*wv(1:N).*Dwv(1:N)).*(sigma.^2).');
# #     end
