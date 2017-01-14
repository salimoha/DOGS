import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt

from scipy import *
from scipy.optimize import minimize, rosen, rosen_der

def func(x):
  return x[0]-x[1], array(([1.0,-1.0]))

def fprime(x):
   return array(([1.0,-1.0]))

guess = 1.2, 1.3
bounds = [(-2.0,2.0), (-2.0,2.0) ]

[best, val, d ]= minimize.fmin_l_bfgs_b(func, guess, fprime,
approx_grad=True, bounds=bounds, iprint=2)

objfun = lambda x: ( costSearch(x,inter_par,xC,R2,y0) )
grad_objfun  = lambda x: kgradSearch(x,inter_par,xC,R2,y0)
# We first solve the problem without providing derivative info

res = minimize(objfun2, x0, method='L-BFGS-B', jac=grad_objfun2, options={'gtol': 1e-6, 'disp': True})