import numpy as np
import numpy.matlib
import Utils as utl
#TEST
n = 3
bnd1 = np.zeros((n, 1))
bnd2 = np.ones((n, 1)) * 3
bnds = utl.bounds(bnd1,bnd2,n)
print(bnds)