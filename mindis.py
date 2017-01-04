import numpy as np
def mindis(x,xi):
# function [y,x1,index] = mindistance(x,xi)
# % calculates the minimum distance from all the existing points
# % xi all the previous points
# % x the new point
    y=float('inf')
    N=xi.shape[1]
    for i in range(N):
        y1 = np.linalg.norm(x[:,0]-xi[:,i])
        if y1<y:
            y=np.copy(y1)
            x1 = np.copy(xi[:,i])
            index=np.copy(i)
    return y,index,x1


xi = np.array([[0.5000 , 0.8000   , 0.5000,    0.2000,    0.5000],  [0.5000,    0.5000,    0.8000,    0.5000,    0.2000]])
xx = np.array([[0.5], [0.53]]);
[y,index,x1] = mindis(xx,xi)
print(y)
print(index)
print(x1)
