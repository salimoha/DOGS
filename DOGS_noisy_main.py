import numpy as np
import scipy
from scipy.linalg import norm
import Utils
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

#%%
np.set_printoptions(linewidth=200, precision=5, suppress=True)
pd.options.display.max_rows = 20
pd.options.display.expand_frame_repr = False

import pylab as plt



# This script shows the alpha DOGS main code

# dimenstion is n
n = 2
# the noise level
sigma0 = 0.1

# truth function quadratic:
funr = lambda x: 5*norm(x-0.3)**2
fun = lambda x: 5*norm(x-0.3)**2 + sigma0 * np.random.rand()

# schewfel
#funr = lambda x: -sum(np.multiply(500*x,np.sin(np.sqrt(abs(500*x))))) / 250
#fun = lambda x: -sum(np.multiply(500*x,np.sin(np.sqrt(abs(500*x))))) / 250 + sigma0 * np.random.rand()

lb = np.zeros((n,1))
ub = np.ones((n,1))

Ain = np.concatenate((np.identity((n)), -np.identity((n))), axis=0)
Bin = np.concatenate((np.ones((n,1)), np.zeros((n,1))),axis=0)

plot_index = 1
iter_max = 50  # maximum number of iterations:

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

fig, (ax0,ax1) = plt.subplots(nrows=2,sharex=False)
fig.subplots_adjust(hspace = .5)

for ff in range(nff):

    #initialization: initial triangulation points
#    xE = np.random.rand(n,n+1)
#    xE = np.round(xE*Nm)/Nm  # quantize the points
    xE = np.array([[0.125,0.375,0.375],[0.625,0.,0.9]])
    # Calculate the function at initial points
    yE = np.zeros(xE.shape[1])
    yr = np.zeros(xE.shape[1])
    T = np.zeros(xE.shape[1])
    for ii in range(xE.shape[1]):
        yE[ii] = fun(lb+(ub-lb)*xE[:,ii].reshape(-1,1))
        yr[ii] = funr(lb+(ub-lb)*xE[:,ii].reshape(-1,1))
        T[ii] = 1

    SigmaT = sigma0 / np.sqrt(T)
    # Muhan Modiefied : I suppose that here we should generate a n by 1 vector full of zeros 
    xU = Utils.bounds(np.zeros([n,1]),np.ones([n,1]),n)

    # initialize Nm, L, K
    L=L0
    inter_par = Utils.Inter_par(method="NPS")
    for k in range(iter_max):
        [inter_par, yp] = Utils.regressionparametarization(xE, yE, SigmaT, inter_par)
        K0 = np.ptp(yE, axis=0)
        # Calculate the discrete function.
        yt = np.amin(yp+SigmaT)
        ind_out = np.argmin(yp+SigmaT)
        # sd = np.amin((yp, 2 * yE - yp) - L * SigmaT, 0)
        sd = np.amin((yp, 2 * yE - yp),0) - L * SigmaT
        
        ypmin  = np.amin(yp)
        ind_min = np.argmin(yp+SigmaT)

        yd = np.amin(sd)
        ind_exist = np.argmin(sd)

        xd = xE[:, ind_exist]

        if ind_min != ind_out:
            yE[ind_exist] = ((fun(xd)) + yE[ind_exist] * T[ind_exist]) / (T[ind_exist]+1)
            T[ind_exist] = T[ind_exist] + 1
        else:
#            tmp1 = np.divide(sigma0 , np.sqrt(T[ind_exist]))
#            tmp2 = 0.01 * np.ptp(yE, axis=0) * (max(ub - lb)) / Nm
            if sigma0 / np.sqrt(T[ind_exist]) < 0.01 * np.ptp(yE, axis=0) * (max(ub - lb)) / Nm:
                yd = np.inf

            # Calcuate the unevaluated function:
            yu = np.zeros([1,xU.shape[1]]) 
            if xU.shape[1] != 0:
                for ii in range(xU.shape[1]):
                    tmp = Utils.interpolate_val(xU[:, ii], inter_par)-np.amin(yp)
                    yu[0,ii] = tmp / Utils.mindis(xU[:, ii], xE)[0]

            if xU.shape[1] != 0 and np.amin(yu) < 0:
                t = np.amin(yu)
                ind = np.argmin(yu)
                xc = xU[:, ind]
                yc = -np.inf
                xU = scipy.delete(xU,ind,1) #create empty array
            else:
                while 1:
                    xc, yc = Utils.tringulation_search_bound_constantK(inter_par, np.hstack([xE, xU]), K * K0, ind_min)
                    yc = yc[0,0]
                    if Utils.interpolate_val(xc, inter_par) < min(yp):
                        xc = np.round(xc * Nm) / Nm
                        break
                    else:
                        xc = np.round(xc * Nm) / Nm
                        if Utils.mindis(xc, xE)[0] < 1e-6:
                            break
                        xc, xE, xU, newadd, success = Utils.points_neighbers_find(xc, xE, xU,Bin,Ain)
                        if success == 1:
                            break
                        else:
                            yu = np.hstack([yu,(Utils.interpolate_val(xc, inter_par) - min(yp)) / Utils.mindis(xc, xE)[0]])

                if xU.shape[1] != 0:
                    tmp = (Utils.interpolate_val(xc, inter_par) - min(yp)) / Utils.mindis(xc, xE)[0]
                    if (np.amin(yu)) < tmp:
                        t = np.amin(yu)
                        ind = np.argmin(yu)
                        xc = xU[:, ind]
                        yc = -np.inf
                        xU = scipy.delete(xU,ind,1)  # create empty array

            # Minimize S_d ^ k(x)
            if Utils.mindis(xc, xE)[0] < 1e-6:
                K = 2 * K
                Nm = 2 * Nm
                L = L + L0
            if yc < yd:
                if Utils.mindis(xc, xE)[0] > 1e-6:
                    # TODO try np.append
                    xE = np.concatenate([xE, xc.reshape(-1,1)],axis=1)
                    yE = np.concatenate((yE, np.array([fun(lb + (ub - lb) * xc)])))
                    T = np.hstack((T,1))
                    yr = np.concatenate([yr, np.array([funr(lb + (ub - lb) * xc)])],axis=0)
                    SigmaT = sigma0 / np.sqrt(T)
            else:
                yE[ind_exist] = ((fun(lb + (ub - lb) * xd)) + yE[ind_exist] * T[ind_exist]) / (T[ind_exist] + 1)
                T[ind_exist] = T[ind_exist] + 1
                SigmaT = sigma0 / np.sqrt(T)
        regret[ff, k] = np.min(yr)
        estimate[ff, k] = yE[ind_out]
        datalength[ff, k] = np.shape(xE)[1]
        mesh[ff, k] = Nm

        color=cm.rainbow(np.linspace(0,1,xE.shape[1]))
        if plot_index:
            ax0.errorbar(range(1,len(yE)+1),yE,yerr=SigmaT)
            ax0.set_title("function evalution and error per points added")
            for i,c in zip(range(xE.shape[0]),color):
                ax1.plot(range(1,len(yE)+1),xE[i],c=c)
            ax1.set_title("x1 and x2")

if 1:
    plt.figure()
    plt.plot(range(1,iter_max+1),estimate[0],'-')
    plt.title("estimate")
    plt.figure()
    plt.plot(range(1,iter_max+1),regret[0],':')
    plt.title("regret")
    plt.figure()
    plt.plot(range(1,iter_max+1),datalength[0],'--')
    plt.title("datalength")
    
################################################################################################################
#square
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