import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt

import Utils as utl
reload(utl)

def interpolate_hessian(x, inter_par):
    if inter_par.method == "NPS" or self.method == 1:
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        n = x.shape[0]
        N = xi.shape[1]
        g = np.zeros((n))
        n = x.shape[0]

        H = np.zeros((n, n))
        for ii in range(N):
            X = x[:, 0] - xi[:, ii]
            if np.linalg.norm(X) > 1e-5:
                H = H + 3 * w[ii] * ((X * X.T) / np.linalg.norm(X) + np.linalg.norm(X) * np.identity(n))
    return H


# xi = pd.DataFrame([[0.5000 , 0.8000   , 0.5000,    0.2000,    0.5000],  [0.5000,    0.5000,    0.8000,    0.5000,    0.2000]])

xi = np.array([[0.5000 , 0.8000   , 0.5000,    0.2000,    0.5000],  [0.5000,    0.5000,    0.8000,    0.5000,    0.2000]])
x0 = np.array([[0.5], [0.53]]);
yi=fun(xi)
inter_par = Inter_par()
inter_par= interpolateparameterization(xi, yi, inter_par)
H = interpolate_hessian(x0,inter_par)
print("-------H-------")
print(H)