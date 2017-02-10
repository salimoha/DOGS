import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

import Utils as utl
"""
Modified Feb 10 2017

@author: Shahrouzalimo & KimuKook
"""

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt


y0 = 0
n = 2

lob = np.zeros((n,1))
upb = np.ones((n,1))
bnd1 = np.copy(lob)
bnd2 = np.copy(upb)

m = 2*n
K=10
iter_max = 10
Ain = np.vstack((np.eye((n)),-np.eye((n))))
bin = np.vstack((bnd2,-bnd1))

xU = utl.bounds(bnd1,bnd2,n)
# inter_method = 1;

inter_par = Inter_par("NPS")

Initial_point = np.hstack((0.5*np.ones((n, 1)),  0.3*np.ones((n, 1))))
xE = Initial_point[:,0]
xe=pd.DataFrame((xE))
xE = xe.values
delta0 = 0.2
Ae = np.kron(np.ones((1,n)), xE) + delta0*np.eye((n))
xE = np.hstack((xE , Ae))
# for ii in range(1,xE.shape[1],1):
#     yE[0,ii] = fun(xE[:,ii])
yE = fun(xE)

for kk in range(iter_max):

    inter_par = interpolateparameterization(xE,yE,inter_par)
    ymin = np.min(yi)
    ind_min = np.argmin(yi)
    xm,ym = tringulation_search_bound_constantK(inter_par,xi,K,ind_min)
#TODO: fix the mindis output...
    if utl.mindis(xm,xE)[0] < 1e-5:
        break

    xE = np.hstack((xE,xm))
    yE = np.hstack((yE,fun(xm)))

#%%


#
# % Calculate
# the
# Initial
# trinagulation
# points
#
# % Calculate
# initial
# delta
# % Initial_point4 = [0.45 * ones(n, 1) 0.1 * ones(n, 1)];
# Initial_point = [0.5 * ones(n, 1) 0.3 * ones(n, 1)];
# % Initial_point = 0.2 + 0.6 * rand(n, 4);
# % Initial_point = round(Initial_point * 8) / 8;
# if n == 4
#     xE = Initial_point(:, initial_index);
#     else
#     xE = Initial_point(:, initial_index);
#     end
#
#     % Initial
#     points: the
#     midpoint
#     point and its
#     neighber
#     ell = 3;
#     % delta0 = 10 / (2 ^ ell);
#     delta0 = 0.2;
#     xE = [xE repmat(xE, 1, n) + delta0 * eye(n)];
#
#     % %
#     % Disceret
#     metric
#     at
#     the
#     initial
#     support
#     points
#     DeltaU = min(pdist2(xU
#     ', xE')');
#
#            % Calculate
#     the
#     function
#     at
#     initial
#     points
#     for ii=1:size(xE, 2)
#     yE(ii) = fun(xE(:, ii));
#     end
#     pE = yE;
#     interpolate_index = ones(1, length(yE));
#     Nm = 20;
#     % Nm = 2 ^ ell;
#     rho = 1;
#
#     for kk=1:3
#
#     for k=1:iter_max
#             % keyboard
#     stop = 0;
#     inter_par = interpolateparametarization(xE, yE, inter_method, interpolate_index);
#     % keyboard \
#       % y0 = min(yE) - range(yE) / Nm;
#     yup = [];
#     yrp = [];
#     % Calculate
#     discerete
#     search
#     function
#     at
#     the
#     support
#     points.
#     DeltaU = min(pdist2(xU
#     ', xE')');
#     yr = DeltaU * 0;
#     for ii=1:size(xU, 2)
#     yr(ii) = interpolate_val(xU(:, ii), inter_par);
#     end
#     yu = yr. / DeltaU;
#     % check
#     the
#     minimum
#     value
#     of
#     xU
#     if size(xU, 2) > 0
#     [tup, ind_up] = min(yu);
#     else
#     tup = inf;
#     end
#
#     if tup < 0
#     [tr, ind_r] = min(yr);
#     [tt, ind] = min(yE);
#     x0 = xE(:, ind);
#     if alg_case == 1
#             % keyboard
#     x=min_decrease(x0, xU(:,
#         ind_r), inter_par);
#     else
#     % keyboard
#     x = xU(:, ind_r);
#     end
#     x = round(x * Nm) / Nm;
#     if mindis(x, xU) < 1e-4
#     x=xU(:,
#         ind_r); xU(:, ind_r)=[];
#     end
#     else
#     while 1
#         [y, ind_min] = min(yE);
#         x = tringulation_search_bound(inter_par, [xE xU], ind_min);
#         if interpolate_val(x, inter_par) < y0
#             [tt, ind_r] = min(yE);
#             x0 = xE(:, ind_r);
#             x = min_decrease(x0, x, inter_par);
#             break
#         end
#         Activation_label = check_activated(x, xE, xU);
#         if Activation_label == 1
#             break
#         else
# x = round(x*Nm)/Nm;
#             xU = [xU x];
#         end
#     end
#     end
#     x = round(x * Nm) / Nm;
#     ss = (interpolate_val(x, inter_par) - y0) / mindis(x, xE);
#     if mindis(x, xE) < 5e-16
#         keyboard
#         break
# end
# if (5 * tup < ss & tup > 0)
#     % keyboard
#     x = xU(:, ind_up);
#     xU(:, ind_up)=[];
# end
# ym = fun(x);
#
# % true = check_add_point(x, ym, [xE xU], inter_par, newadd, size(xE, 2));
# xE = [xE x];
# yE = [yE ym];
# figure(1)
# subplot(2, 1, 1)
# plot(yE)
# subplot(2, 1, 2)
# plot(xE
# ')
# drawnow
#
# end
#
# Nm = 2 * Nm;
# end
# if n == 2
# figure(2)
# my_2dcontourf(fun, -5, 5)
# plot(-5 + 10 * xE(1,:), -5 + 10 * xE(2,:), 'ks', 'markerFacecolor', 'w', 'markersize', 10)
# hold
# on
# plot(-5 + 10 * xU(1,:), -5 + 10 * xU(2,:), 'k*', 'markersize', 10)
# set(gca, 'fontsize', 15)
# % saveas(gcf, ['DeltaZ\n=' num2str(n) '_alg' num2str(alg_case) 'initial_index' num2str(initial_index) '.eps']);
# print(['DeltaZ\n=' num2str(n) '_alg' num2str(alg_case) 'initial_index' num2str(initial_index)], '-depsc')
# else
# figure(2)
# plot(1:length(yE), yE(1:end), '-', 'linewidth', 2)
# % axis([0 12 0 150])
# grid
# on
# set(gca, 'fontsize', 30)
# axes('Position', [0.7, 0.3, 0.28, 0.28])
# ind = find(yE < 10);
# plot(ind, yE(ind), '-', 'linewidth', 1.5)
# set(gca, 'fontsize', 15)
# grid
# on
# mkdir('DeltaZ')
# saveas(gcf, ['DeltaZ\n=' num2str(n) '_alg' num2str(alg_case) 'initial_index' num2str(initial_index) '.eps']);
# end
# Costl(initial_index) = length(yE);
# Cost1l(initial_index) = size(xU, 2);
# end
# % keyboard
# Cost(ttt, alg_case) = mean(Costl);
# Cost1(ttt, alg_case) = mean(Cost1l);
# end
# end
