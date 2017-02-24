import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt

from scipy import optimize

def regressionparametarization(xi,yi, sigma, inter_par):
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
        b = np.mean(np.divide(np.dot(V.T,w1)-yi.reshape(-1,1),sigma)**2)
        wv = np.zeros([N+n+1])
        if b < 1:
            wv[N:] = np.copy(w1.T)
            rho = 1000
            wv = np.copy(wv.reshape(-1,1))
        else:
            rho = 1.1
            fun = lambda rho:smoothing_polyharmonic(rho,A,V,sigma,yi,n,N,1)
            sol = optimize.fsolve(fun,rho)
            b,db,wv = smoothing_polyharmonic(sol,A,V,sigma,yi,n,N,3)
        inter_par.w = wv[:N]
        inter_par.v = wv[N:]
        inter_par.xi = xi
        yp = np.zeros([N])
        while(1):
            for ii in range(N):
                yp[ii] = interpolate_val(xi[:,ii],inter_par)
            residual = np.max(np.divide(np.abs(yp-yi),sigma[0]))
            if residual < 2:
                break
            rho *= 0.9
            b,db,wv = smoothing_polyharmonic(rho,A,V,sigma,yi,n,N)
            inter_par.w = wv[:N]
            inter_par.v = wv[N:]
    return inter_par, yp
    

def smoothing_polyharmonic(rho, A, V, sigma, yi, n, N,num_arg):
    # Notice: num_arg = 1 will return b
    #         num_arg = else will return b,db,wv
    A01 = np.concatenate((A + rho * np.diag(sigma ** 2), np.transpose(V)), axis=1)
    A02 = np.concatenate((V, np.zeros(shape=(n + 1, n + 1))), axis=1)
    A1 = np.concatenate((A01, A02), axis=0)
    b1 = np.concatenate([yi.reshape(-1,1), np.zeros(shape=(n + 1, 1))])
    wv = np.linalg.solve(A1, b1)
    b = np.mean(np.multiply(wv[:N],sigma)**2*rho**2) - 1
    bdwv = np.concatenate([np.multiply(wv[:N],sigma.reshape(-1,1)**2), np.zeros((n + 1, 1))])
    Dwv = np.linalg.solve(-A1, bdwv)
    db = 2 * np.mean(np.multiply(wv[:N]**2*rho + rho**2*np.multiply(wv[:N],Dwv[:N]),sigma**2))
    if num_arg == 1:
        return b
    else:
        return b,db,wv
    

def interpolate_val(x, inter_par):
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        # Muhan modified on 2/10:
        # Usually x is a column extracted from xi, then x becomes a one-dimension vector, when xi is two-dimension matrix.
        # We need to convert x to be a two dimension vector. So I reshape x.
        n = xi.shape[0]  # Row of xi
        x = np.copy(x.reshape(n,1)) # Similar to add a newaxis to x
        S = xi - x
        #             print np.dot(v.T,np.concatenate([np.ones((1,1)),x],axis=0)) + np.dot(w.T,np.sqrt(np.diag(np.dot(S.T,S))))**3
        return np.dot(v.T, np.concatenate([np.ones((1, 1)), x], axis=0)) + np.dot(w.T, (np.sqrt(np.diag(np.dot(S.T, S))) ** 3))

def fun(x,  alpha=0.1):
    y = np.array((x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0)
    return y.T
#    return (x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0

class Inter_par():
    def __init__(self,method="NPS", w=0, v=0, xi=0,a=0):
        self.method = "NPS"
        self.w=[]
        self.v=[]
        self.xi=[]
        self.a=[]
#%%
inter_par = Inter_par(method="NPS")
xi = np.array([[0,0.1000,0.2000,0.3000,0.5000,1]])
yi=np.array([[0.0049,0.2169,0.0473,0.0293,0.2529,0.8425]])
sigma = np.array([[0.0100,0.2000,0.0100,0.2000,0.0100,0.2000]])
inter_par,yp1 = regressionparametarization(xi,yi, sigma, inter_par)
print('yp',yp1)
print('inter_par.v')
print(inter_par.v)
print('inter_par.w')
print(inter_par.w)