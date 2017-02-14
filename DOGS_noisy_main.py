import numpy as np
from scipy.linalg import norm
import Utils
import pandas as pd

np.set_printoptions(linewidth=200, precision=5, suppress=True)
pd.options.display.max_rows = 20
pd.options.display.expand_frame_repr = False

import pylab as plt

def interpolate_val(x, inter_par):
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        # Muhan modified on 2/10:
        # Usually x is a column extracted from xi, then x becomes a one-dimension vector, when xi is two-dimension matrix.
        # We need to convert x to be a two dimension vector. So I reshape x.
        n = xi.shape[0]  # Row of xi
        x = np.copy(x.reshape(n, 1))  # Similar to add a newaxis to x
        S = xi - x
        #             print np.dot(v.T,np.concatenate([np.ones((1,1)),x],axis=0)) + np.dot(w.T,np.sqrt(np.diag(np.dot(S.T,S))))**3
        return np.dot(v.T, np.concatenate([np.ones((1, 1)), x], axis=0)) + np.dot(w.T, (np.sqrt(np.diag(np.dot(S.T, S))) ** 3))



def regressionparametarization(xi, yi, sigma, inter_par):
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
        b = np.mean(np.divide(np.dot(V.T, w1) - yi.reshape(-1, 1), sigma) ** 2)
        wv = np.zeros([N + n + 1])
        if b < 1:
            wv[N:] = np.copy(w1.T)
            rho = 1000
            wv = np.copy(wv.reshape(-1, 1))
        else:
            rho = 1.1
            fun = lambda rho: smoothing_polyharmonic(rho, A, V, sigma, yi, n, N, 1)
            sol = optimize.fsolve(fun, rho)
            b, db, wv = smoothing_polyharmonic(sol, A, V, sigma, yi, n, N, 3)
        inter_par.w = wv[:N]
        inter_par.v = wv[N:]
        inter_par.xi = xi
        yp = np.zeros([N])
        while (1):
            for ii in range(N):
                yp[ii] = interpolate_val(xi[:, ii], inter_par)
            residual = np.max(np.divide(np.abs(yp - yi), sigma[0]))
            if residual < 2:
                break
            rho *= 0.9
            b, db, wv = smoothing_polyharmonic(rho, A, V, sigma, yi, n, N)
            inter_par.w = wv[:N]
            inter_par.v = wv[N:]
    return inter_par, yp


def smoothing_polyharmonic(rho, A, V, sigma, yi, n, N, num_arg):
    # Notice: num_arg = 1 will return b
    #         num_arg = else will return b,db,wv
    A01 = np.concatenate((A + rho * np.diag(sigma ** 2), np.transpose(V)), axis=1)
    A02 = np.concatenate((V, np.zeros(shape=(n + 1, n + 1))), axis=1)
    A1 = np.concatenate((A01, A02), axis=0)
    b1 = np.concatenate([yi.reshape(-1, 1), np.zeros(shape=(n + 1, 1))])
    wv = np.linalg.solve(A1, b1)
    b = np.mean(np.multiply(wv[:N], sigma) ** 2 * rho ** 2) - 1
    bdwv = np.concatenate([np.multiply(wv[:N], sigma.reshape(-1, 1) ** 2), np.zeros((n + 1, 1))])
    Dwv = np.linalg.solve(-A1, bdwv)
    db = 2 * np.mean(np.multiply(wv[:N] ** 2 * rho + rho ** 2 * np.multiply(wv[:N], Dwv[:N]), sigma ** 2))
    if num_arg == 1:
        return b
    else:
        return b, db, wv


# This script shows the alpha DOGS main code

# dimenstion is n
n = 2
# the noise level
sigma0 = 0.3

def fun(x, alpha=0.1):
    y = np.array((x[0, :] - 0.45) ** 2.0 ) +np.array((x[1, :] - 0.45) ** 2.0 )  + alpha*np.random.rand()
    return y.T

def funr(x, alpha=0.1):
    y = np.array((x[0, :] - 0.45) ** 2.0 ) +np.array((x[1, :] - 0.45) ** 2.0 )
    return y.T

# truth function
# funr = lambda x: (5*np.norm(x-0.3)**2)
# fun = lambda x: (5*np.norm(x-0.3)**2+sigma0*np.random.rand())
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
        # TODO: fix matrix
        # yE[ii] = fun(lb+(ub-lb)*xE[:,ii].T)
        # yr[ii] = funr(lb+(ub-lb)*xE[:,ii].T)
        T[ii] = 1

    yE = fun(xE)
    yi = yi[np.newaxis, :]
    yr = funr(xE)
    yr = yr[np.newaxis, :]
    T = np.ones((1,len(yE)))


    SigmaT = sigma0 / np.square(T)
    # Muhan Modiefied : I suppose that here we should generate a n by 1 vector full of zeros
    xU = Utils.bounds(np.zeros([n,1]),np.ones([n,1]),n)

    # initialize Nm, L, K
    L=L0
    inter_par = Utils.Inter_par(method="NPS")
    for k in range(iter_max):
        [inter_par, yp] = regressionparametarization(xE, yE, sigma0/ np.square(T), inter_par)

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
            tmp1 = np.divide(sigma0 , np.sqrt(T[ind_exist]))
            tmp2 = 0.01 * np.ptp(yE, axis=0) * (max(ub - lb)) / Nm
            if tmp1 < tmp2:
                yd = np.inf

            # Calcuate the unevaluated function:
            yu = np.empty( shape=[0, 0] ) #create empty array
            if (xU.shape[1]!=0):
                yu = np.zeros(xU.shape[1]);
                for ii in range(xU.shape[1]):
                    tmp = Utils.interpolate_val(xU[:, ii], inter_par)-np.amin(yp,0)
                    t1,_,_ = Utils.mindis(np.array([xU[:, ii]]), xE)
                    yu[ii] = tmp / t1

            if xU.shape[1]!=0 and np.min(yu) < 0:
                t = np.amin(yu, 0)
                ind = np.argmin(yu, 0)
                xc = xU[:, ind]
                yc = -np.inf
                xU[:, ind] = np.empty( shape=[1] ) #create empty array
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
            t1,_,_ = Utils.mindis(np.array([xc]), xE)
            if t1 < 1e-6:
                K = 2 * K
                Nm = 2 * Nm
                L = L + L0

            if (yc < yd):
                # keyboard
                t1,_,_ = Utils.mindis(np.array([xc]), xE)
                if t1 > 1e-6:
                    xE = np.concatenate([xE, np.array([xc])],axis=1)
                    yE = np.concatenate((yE, np.array([fun(lb + (ub - lb) * xc)])))
                    T = [T, 1]
                    yr = np.concatenate([yr, np.array([funr(lb + (ub - lb) * xc)])],axis=0)
                    SigmaT = sigma0 / np.square(T)
                else:
                    yE[ind_exist] = ((fun(lb + (ub - lb) * xd)) + yE[ind_exist] * T[ind_exist]) / (T[ind_exist] + 1)
                    T[ind_exist] = T[ind_exist] + 1
                    SigmaT = sigma0 / np.square(T)
        regret[ff, k] = np.min(yr)
        estimate[ff, k] = yE[ind_out]
        datalength[ff, k] = np.shape(xE)[1]
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
#%%

import matplotlib.pyplot as plt



xx = np.arange(0, 1.05, 0.05)
xx = pd.DataFrame(xx).values
yy= np.zeros((len(xx),1))
yr= np.zeros((len(xx),1))

for ii in range(len(xx)):
    yy[ii] = interpolate_val( xx[ii],inter_par)
yr = funr(xx.T)
# ground truth
# yr =
yr = yr[:,np.newaxis]

plt.figure()
plt.plot(xx, yy, '--')
plt.plot(xx, yr, '-')
plt.plot(xi, yi, '*')
# example variable error bar values
plt.errorbar(xi[0], yi[0], yerr=sigma[0], linestyle="None")
plt.title("modeling the error function")
plt.show()