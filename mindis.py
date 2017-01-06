import numpy as np
import numpy.matlib
import Utils as utl
reload(utl)

xi = np.array([[0.5000 , 0.8000   , 0.5000,    0.2000,    0.5000],  [0.5000,    0.5000,    0.8000,    0.5000,    0.2000]])
xx = np.array([[0.5], [0.53]]);
[y,index,x1] = utl.mindis(xx,xi)
print(y)
print(index)
print(x1)
