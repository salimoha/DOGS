import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt

import Utils as utl
reload(utl)

def interpolate_val(x, inter_par):
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi

        S = xi - x
        #             print np.dot(v.T,np.concatenate([np.ones((1,1)),x],axis=0)) + np.dot(w.T,np.sqrt(np.diag(np.dot(S.T,S))))**3
        return np.dot(v.T, np.concatenate([np.ones((1, 1)), x], axis=0)) + np.dot(w.T, (np.sqrt(np.diag(np.dot(S.T, S))) ** 3))



# xi = pd.DataFrame([[0.5000 , 0.8000   , 0.5000,    0.2000,    0.5000],  [0.5000,    0.5000,    0.8000,    0.5000,    0.2000]])

xi = np.array([[0.5000 , 0.8000   , 0.5000,    0.2000,    0.5000],  [0.5000,    0.5000,    0.8000,    0.5000,    0.2000]])
x0 = np.array([[0.5], [0.53]]);
yi=fun(xi)
inter_par = Inter_par()
inter_par= interpolateparameterization(xi, yi, inter_par)
y0 = interpolate_val(x0,inter_par)
print(y0)