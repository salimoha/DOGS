import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt

import Utils as utl
reload(utl)

def interpolate_grad(x, inter_par):
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        n = x.shape[0]
        N = xi.shape[1]
        g = np.zeros((n))
        for ii in range(N):
            X = x[:, 0] - xi[:, ii]
            g = g + 3 * w[ii] * X.T * np.linalg.norm(X)
        # print("--------------")
        #                 print v[ii]
        #             print(g)
        #             print("--------------")
        #             print(v[1:])
        g = g + v[1:, 0]

        return g.T

# xi = pd.DataFrame([[0.5000 , 0.8000   , 0.5000,    0.2000,    0.5000],  [0.5000,    0.5000,    0.8000,    0.5000,    0.2000]])

xi = np.array([[0.5000 , 0.8000   , 0.5000,    0.2000,    0.5000],  [0.5000,    0.5000,    0.8000,    0.5000,    0.2000]])
x0 = np.array([[0.5], [0.53]]);
yi=fun(xi)
inter_par = Inter_par()
inter_par= interpolateparameterization(xi, yi, inter_par)
g0 = interpolate_grad(x0,inter_par)


print(g0)