import numpy as np
import scipy
from scipy.linalg import norm
import Utils
import pandas as pd

from matplotlib.pyplot import cm

import matplotlib.pyplot as pl
# %%
np.set_printoptions(linewidth=200, precision=5, suppress=True)
pd.options.display.max_rows = 20
pd.options.display.expand_frame_repr = False

import pylab as plt

# This script shows the alpha DOGS main code

# dimenstion is n
n = 2


# truth function
#x_star=np.ones((n,1))*0.3
# funr = lambda x: (5 * norm(x - 0.3) ** 2)
# fun = lambda x: (5 * norm(x - 0.3) ** 2 )#+ sigma0 * np.random.rand())

# schewfel
#x_star=np.ones((n,1))*0.8419
#fun = lambda x: 1.6759*n-sum(np.multiply(500 * x, np.sin(np.sqrt(abs(500 * x)))))[0] / 250 #+ sigma0 * np.random.rand()

# Rastriginn
x_star = np.ones((n,1))*0.7
fun = lambda x: Utils.rastriginn2(x)


lb = np.zeros((n, 1))
ub = np.ones((n, 1))

Ain = np.concatenate((np.identity((n)), -np.identity((n))), axis=0)
Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)

plot_index = 1
iter_max = 110  # maximum number of iterations:

# interpolation strategy:
inter_method = 1  # polyharmonic spline

# Calculate the Initial trinagulation points
Nm = 100  # initial mesh grid size
L0 = 1  # discrete 1 search function constant
K = 20 # continous search function constant

nff = 1

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=False)
fig.subplots_adjust(hspace=.5)


for ff in range(nff):

    # initialization: initial triangulation points
    #xE = np.random.rand(n, n + 1)
    #xE = np.round(xE * Nm) / Nm  # quantize the points
    #    xE = np.array([[0.125,0.375,0.375],[0.625,0.,0.9]])
    xE = Utils.bounds(np.zeros([n, 1]), np.ones([n, 1]), n)
    #xE = np.concatenate((xE, xU), axis=1)
    # Calculate the function at initial points
    yE = np.zeros(xE.shape[1])
    yr = np.zeros(xE.shape[1])
    T = np.zeros(xE.shape[1])
    for ii in range(xE.shape[1]):
        yE[ii] = fun(lb + (ub - lb) * xE[:, ii].reshape(-1, 1))


    # initialize Nm, K
    inter_par = Utils.Inter_par(method="NPS")
    for k in range(iter_max):
        # [inter_par, yp] = Utils.regressionparametarization(xE, yE, SigmaT, inter_par)
        inter_par = Utils.interpolateparameterization(xE, yE, inter_par)
        K0 = np.ptp(yE, axis=0)

        ind_exist = np.argmin(yE, 0)
        xd = xE[:, ind_exist]
        
        #ploting
        color = cm.rainbow(np.linspace(0, 1, xE.shape[1]))
        if plot_index:
            ax0.plot(range(1, len(yE) + 1), yE)
            ax0.set_title("function evalution and error per points added")
            for i, c in zip(range(xE.shape[0]), color):
                ax1.plot(range(1, len(yE) + 1), xE[i], c=c)
            ax1.set_title("x1 and x2")
        

        xc, yc = Utils.tringulation_search_bound_constantK(inter_par, xE, K*K0 , ind_exist)
        yc = yc[0, 0]
        #                    if Utils.interpolate_val(xc, inter_par) < min(yp):

        xc = np.round(xc * Nm) / Nm
        #                        break
        #                    else:
        #                        xc = np.round(xc * Nm) / Nm
        if Utils.mindis(xc, xE)[0] < 1e-6:
            break
        xE = np.concatenate([xE, xc.reshape(-1, 1)], axis=1)
        yE = np.concatenate((yE, np.array([fun(lb + (ub - lb) * xc)])))
        # T = np.hstack((T, 1))
        ind_exist = np.argmin(yE, 0)
        xd = xE[:, ind_exist]
        print("iter", k, ": distance from ||xk-x*|| = ", np.linalg.norm(xd-x_star), " and |yk-y0| = ", np.min(yE) )
pl.show()
        
        
        
