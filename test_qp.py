import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt

from cvxopt import matrix
from cvxopt.solvers import qp
P = matrix( [[ 13.0,  12, -2],
             [ 12,  17,  6],
             [ -2,   6,   12]] )

q = matrix([-22,-14.5,13])
G = matrix(0.0, (6,3))
G[0,0] =  G[1,1] =  G[2,2] = 1.0;
G[3,0] =  G[4,1] = G[5,2] = -1.0;
h = matrix(1.0, (6,1))
A = matrix(0.0, (1,3))
b = matrix(0.0)
xs = qp(P, q, G, h)
print xs['x']