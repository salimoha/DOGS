import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt



# This script shows the alpha DOGS main code

# dimenstion is n
n = 1
# the noise level
sigma0=0.3

# truth function
funr = lambda x: (5*np.norm(x-0.3)**2)
fun = lambda x: (5*np.norm(x-0.3)**2+sigma0*np.random.rand())



# constants
iter_max = 300
plot_index = 1

#  interpolation strategy
inter_method = 1     #polyharmonic spline

# initial mesh grid size
Nm = 8

# AlphaDOGS search function constants
alpha0 = 1      #discrete 1 search function constant
K = 3           #continous search function constant

#initialization: initial triangulation points
xE = np.random.rand(n,n+1)
# quantize the points
xE = round(xE*Nm)/Nm



