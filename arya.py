import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt
import numpy.matlib

import Utils as utl
#TEST
n = 10
bnd1 = pd.DataFrame(np.zeros((n)))
bnd2 = pd.DataFrame(np.ones((n)) * 3)
bnd1.values
bnd2.values

bnds = utl.bounds(bnd1.values,bnd2.values,n)
print(bnds)