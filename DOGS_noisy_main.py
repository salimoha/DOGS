import numpy as np
from scipy.linalg import norm
import Utils
import pandas as pd

np.set_printoptions(linewidth=200, precision=5, suppress=True)
pd.options.display.max_rows = 20
pd.options.display.expand_frame_repr = False

import pylab as plt



# This script shows the alpha DOGS main code

# dimenstion is n
n = 1
# the noise level
sigma0 = 0.3

# truth function
funr = lambda x: (5*norm(x-0.3)**2)
fun = lambda x: (5*norm(x-0.3)**2+sigma0*np.random.rand())
lb = 0 * np.ones((n,1))
ub = np.ones((n,1))

plot_index = 1
iter_max = 1  # maximum number of iterations:

# interpolation strategy:
inter_method = 1  # polyharmonic spline

# Calculate the Initial trinagulation points
Nm = 8  # initial mesh grid size
L0 = 1  # discrete 1 search function constant
K = 3  # continous search function constant

nff = 1

regret = np.zeros((nff, iter_max))
estimate = np.zeros((nff, iter_max))
datalength = np.zeros((nff, iter_max))
mesh = np.zeros((nff, iter_max))
for ff in range(nff):

    #initialization: initial triangulation points
    xE = np.random.rand(n,n+1)
    xE = np.round(xE*Nm)/Nm  # quantize the points

    # Calculate the function at initial points
    yE = np.zeros(xE.shape[1])
    yr = np.zeros(xE.shape[1])
    T = np.zeros(xE.shape[1])
    for ii in range(xE.shape[1]):
        yE[ii] = fun(lb+(ub-lb)*xE[:,ii])
        yr[ii] = funr(lb+(ub-lb)*xE[:,ii])
        T[ii] = 1

    SigmaT = sigma0 / np.square(T)
    xU = Utils.bounds(np.zeros(n,1),np.ones(n,1),n)

    # initialize Nm, L, K
    L=L0

    for k in range(iter_max):
        [inter_par, yp] = Utils.regressionparametarization(xE, yE, sigma0/ np.square(T), inter_method)

        K0 = np.ptp(yE, axis=0)

        #Calculate the discrete function.
        tmp = yp+SigmaT
        yt = np.amin(tmp, 0)
        ind_out = np.argmin(tmp, 0)
        sd = np.amin((yp, 2 * yE - yp) - L * SigmaT, 0)

        ypmin  = np.amin(yp, 0)
        ind_min = np.argmin(tmp, 0)

        yd = np.amin(sd, 0)
        ind_exist = np.argmin(sd, 0)

        xd = xE[:, ind_exist]

        if (np.array_equal(ind_min, ind_out)==False):
            yE[ind_exist] = ((fun(xd)) + yE(ind_exist) * T(ind_exist)) / (T(ind_exist)+1)
            T[ind_exist] = T(ind_exist)+1
        else:
            tmp1 = sigma0 / sd.square(T(ind_exist))
            tmp2 = 0.01 * range(yE) * (max(ub - lb)) / Nm
            if tmp1 < tmp2:
                yd = np.inf

            # Calcuate the unevaluated function:
            yu = np.empty( shape=(0, 0) ) #create empty array
            if (xU.shape[1]!=0):
                yu = np.zeros(xU.shape[1]);
                for ii in range(xU.shape[1]):
                    tmp = Utils.interpolate_val(xU[:, ii], inter_par)-np.amin(yp,0)
                    yu[ii] = tmp / Utils.mindis(xU[:, ii], xE)

            if (xU.shape[1]!=0 & np.amin(yu,0) < 0):
                t = np.amin(yu, 0)
                ind = np.argmin(yu, 0)
                xc = xU[:, ind]
                yc = -np.inf
                xU[:, ind] = np.empty( shape=(0, 0) ) #create empty array
            else:
                #Minimize s_c ^ k(x)
                while 1:
                    xc, yc = Utils.tringulation_search_bound_constantK(inter_par, [xE, xU], K * K0, ind_min);
                    if (Utils.interpolate_val(xc, inter_par) < min(yp)):
                        xc = np.round(xc * Nm) / Nm
                        break
                    else:
                        xc = np.round(xc * Nm) / Nm
                        if (Utils.mindis(xc, xE) < 1e-6):
                            break

                        xc, xE, xU, success = Utils.points_neighbers_find(xc, xE, xU)
                        if (success == 1):
                            break
                        else:
                            yu = [yu(Utils.interpolate_val(xc, inter_par) - min(yp)) / Utils.mindis(xc, xE)]

                if (xU.shape[1]!=0):
                    tmp = (Utils.interpolate_val(xc, inter_par) - min(yp)) / Utils.mindis(xc, xE)
                    if (np.amin(yu, 0)) < tmp:
                        t = np.amin(yu, 0)
                        ind = np.argmin(yu, 0)
                        xc = xU[:, ind]
                        yc = -np.inf
                        xU[:, ind] = np.empty(shape=(0, 0))  # create empty array

            # Minimize S_d ^ k(x)
            if (Utils.mindis(xc, xE) < 1e-6):
                K = 2 * K
                Nm = 2 * Nm
                L = L + L0

            if (yc < yd):
                # keyboard
                if (Utils.mindis(xc, xE) > 1e-6):
                    xE = [xE, xc]
                    yE = [yE, fun(lb + (ub - lb) * xc)]
                    T = [T, 1]
                    yr = [yr, funr(lb + (ub - lb) * xc)]
                    SigmaT = sigma0 / np.square(T)
                else:
                    yE[ind_exist] = ((fun(lb + (ub - lb) * xd)) + yE[ind_exist] * T[ind_exist]) / (T[ind_exist] + 1)
                    T[ind_exist] = T[ind_exist] + 1
                    SigmaT = sigma0 / np.square(T)

        regret[ff, k] = np.amin(yr, 0)
        estimate[ff, k] = yE[ind_out]
        datalength[ff, k] = np.shape(xE)
        mesh[ff, k] = Nm


################################################################################################################
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

