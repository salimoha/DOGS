import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt



# This script shows the alpha DOGS main code

# dimenstion is n
n = 1
# the noise level
sigma0=0.3

# truth function
funr = lambda x: (5*np.norm(x-0.3)**2)
fun = lambda x: (5*np.norm(x-0.3)**2+sigma0*np.random.rand())



# constants
iter_max = 300
plot_index = 1

#  interpolation strategy
inter_method = 1     #polyharmonic spline

# initial mesh grid size
Nm = 8

# AlphaDOGS search function constants
L0 = 1      #discrete 1 search function constant
K = 3           #continous search function constant

#initialization: initial triangulation points
xE = np.random.rand(n,n+1)
# quantize the points
xE = round(xE*Nm)/Nm



# % Calculate the function at initial points
for ii in range(xE.shape[1]):
    yE[ii]= fun(lb+(ub-lb)*xE[:,ii]);
    # yr[ii]=funr(lb+(ub-lb).*xE(:,ii));
    T[ii]=1

xU= bounds(np.zeros(n,1),np.ones(n,1),n)
# %K0=max(K0,0.1);

# % initialize Nm, L, K
L=L0


for k in range(iter_max):
    [inter_par, yp] = regressionparametarization(xE, yE, sigma0/ np.sqrt(T), inter_method);
    K0 = np.ptp(yE, axis=1)
    [yt, ind_out] = np.min(yp + sigma0*1.0/ np.square(T))
    sd = np.min(yp, 2 * yE - yp) - L * sigma0. / np.square(T);
    ypmin = np.min(yp)
    ind_min = np.argmin(yp)
    yd = np.min(sd)
    ind_exist = np.argmin(sd)
    xd = xE[:, ind_exist]





# % for k=1:iter_max
# % keyboard
# for k=1:iter_max
# [inter_par, yp] = regressionparametarization(xE, yE, sigma0. / sqrt(T), inter_method);
# % inter_par = interpolateparametarization(xi, yi, inter_method);
# K0 = range(yE);
# % Calculate
# the
# discrete
# function.
# [yt, ind_out] = min(yp + sigma0. / sqrt(T));
# sd = min(yp, 2 * yE - yp) - L * sigma0. / sqrt(T);
# [ypmin, ind_min] = min(yp);
# [yd, ind_exist] = min(sd);
# xd = xE(:, ind_exist);
# % ind_min = 1;
# ind_exist = 1;
# % yd = inf;
# if (ind_min~= ind_min ) % | | ind_min~= ind_out)
# yE(ind_exist)=((fun(xd))+yE(ind_exist) * T(ind_exist)) / (T(ind_exist)+1);
# T(ind_exist)=T(ind_exist)+1;
# else
#
# if sigma0./ sqrt(T(ind_exist)) < 0.01 * range(yE) * (max(ub-lb)) / Nm
# yd=inf;
# end
# % Calcuate the unevaluated function:
#     clear
# yu
# yu = [];
# if size(xU, 2)
# ~ = 0;
# for ii = 1:size(xU, 2)
# yu(ii) = (interpolate_val(xU(:, ii), inter_par)-min(yp)) / mindis(xU(:, ii), xE);
# end
# end
# if (size(xU, 2)
# ~ = 0 & & min(yu) < 0)
# [t, ind] = min(yu);
# xc = xU(:, ind); yc = -inf;
# xU(:, ind)=[];
# else
# % Minimize
# s_c ^ k(x)
# while 1
#     [xc yc] = tringulation_search_bound_constantK(inter_par, [xE xU], K * K0, ind_min);
# if interpolate_val(xc, inter_par) < min(yp)
#     xc = round(xc * Nm) / Nm;
#     break
# else
#     xc = round(xc. * Nm) / Nm;
# if mindis(xc, xE) < 1e-6
#     break
# end
# [xc, xE, xU, success] = points_neighbers_find(xc, xE, xU);
# if success == 1
#     break
# else
#     yu = [yu(interpolate_val(xc, inter_par) - min(yp)) / mindis(xc, xE)];
# end
# end
# end
# if size(xU, 2)~=0
# if min(yu) < (interpolate_val(xc, inter_par) - min(yp)) / mindis(xc, xE)
#     [t, ind] = min(yu);
#     xc = xU(:, ind); yc = -inf;
#     xU(:, ind)=[];
# end
# end
# end
# % if (Nm > 128 & & sigma0. / sqrt(max(T)) < range(yE) * (max(ub - lb)) / Nm)
#     % break
# % end
#
# % Minimize
# S_d ^ k(x)
# if (mindis(xc, xE) < 1e-6)
#     % if (max(T) > 200)
#         K = 2 * K;
#         Nm = 2 * Nm;
#         L = L + L0;
#     % else
#     % yd = -inf;
#     % end
# end
#
# if yc < yd
#     % keyboard
#     if mindis(xc, xE) > 1e-6
#         xE = [xE xc];
#         yE = [yE fun(lb + (ub - lb). * xc)];
#         T = [T 1];
#         yr = [yr funr(lb + (ub - lb). * xc)];
#     end
# else
#     yE(ind_exist) = ((fun(lb + (ub - lb). * xd)) + yE(ind_exist) * T(ind_exist)) / (T(ind_exist) + 1);
#     T(ind_exist) = T(ind_exist) + 1;
# end
# end
# % regret(ff, k) = funr(xE(:, ind_out));
# regret(ff, k) = min(yr);
# estimate(ff, k) = yE(ind_out);
# datalength(ff, k) = length(xE);
# mesh(ff, k) = Nm;
# % if Nm > 60
#     % break
# % end
#
# if plot_index
#     subplot(2, 1, 1)
#     errorbar(1:length(yE), yE, sigma0. / sqrt(T))
#     subplot(2, 1, 2)
#     plot(xE.
#     ')
#     drawnow
#     end
#     end
#     end
#
#
#
# 1.	Intializaiton of the vertices
# 2.	Construct the Delaunay triangulations of S
# 3.	Constrict an appropriate regression model of S. calculate the search function as sc(x) = p(x)-Ke(x).
#  evaluate the discrete search funciton as:
#           sd = min ( p(x) , yi + (yi-p(x))) - L * sigma(h,T) - sigma(Lx)
# 4.	If sc <= sd then evaluate the minimizer of the continuous search function.
# 5.       else:
#                calculate the loss function as:
#                    minimize_{N,T}     Loss = [ N log(N) ]^n * T * N
#                    such that          L * sigma(h,T)   <   eps
#                       where eps = min { (sd_i - sc),  (sd_i - sd{i-1}) }   [with minimum N value]
#           Improve the existing point accuracy with the N_new and T_new.

