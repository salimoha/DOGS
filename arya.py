import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt
import numpy.matlib


def bounds(bnd1,bnd2,n):
#   find vertex of domain for a box domain.
#   INPUT: n: dimension, bnd1: lower bound, bnd2: upper bound.
#   OUTPUT: vertex of domain. 2^n number vector of n-D.
#   Example:
#           n = 3
#           bnd1 = np.zeros((n, 1))
#           bnd2 = np.ones((n, 1))
#           bnds = bounds(bnd1,bnd2,n)
#   Author: Shahoruz Alimohammadi
#   Modified: Dec., 2016
#   DELTADOGS package
	bnds = np.matlib.repmat(bnd2,  1, 2**n)
	for ii in range(n):
	    tt = np.mod(np.arange(2**n)+1, 2**(n-ii)) <= 2**(n-ii-1)-1
	    bnds[ii, tt] = bnd1[ii];
	return pd.DataFrame(bnds)
#TEST
n = 10
bnd1 = pd.DataFrame(np.zeros((n)))
bnd2 = pd.DataFrame(np.ones((n)) * 3)
bnd1.values
bnd2.values

bnds = bounds(bnd1.values,bnd2.values,n)
print(bnds)