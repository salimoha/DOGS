import numpy as np
from scipy.spatial import Delaunay

import Utils as utl

def direc_uncer(x,xi,tri):
    e=np.array([[0.0]]);
    n = x.shape[0]
#    print(n)
    for ind in range(tri.simplices.shape[0]):
        [R2,xC] = utl.circhyp(xi[:,tri.simplices[ind,:]],n)
        e = np.array([e,(R2- np.dot(np.transpose((x-xC)), x-xC)) ]).max()
    return e

N = 2;
x = np.array([[0.6443,    0.8116,    0.3507], [0.3786,    0.5328,    0.9390]]);
[R2,xC] = utl.circhyp(x, N)


x = np.array([[0.5000 , 0.8000   , 0.5000,    0.2000,    0.5000],  [0.5000,    0.5000,    0.8000,    0.5000,    0.2000]])
tri = Delaunay(x.T)
xx = np.array([[0.5], [0.5]]);
e = direc_uncer(xx,x,tri)
print("---Global e -----")
print(e)