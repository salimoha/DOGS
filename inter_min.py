import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt

def inter_min(x,inter_par, Ain=[], bin=[]):
    # %find the minimizer of the interpolating function starting with x
    rho = 0.9 # backtracking paramtere
    n = x.shape[0]
#     start the serafh method
    iter = 0
    x0 = np.zeros((n, 1))
    while iter < 10:
        H = np.zeros((n, n))
        g = np.zeros((n, 1))
        y = interpolate_val(x, inter_par)
        g = interpolate_grad(x, inter_par)
        H = interpolate_hessian(x, inter_par)
        # Perform the Hessian modification
        H = modichol(H, 0.01, 20);
        H = (H + H.T)/2.0
#         optimizaiton for finding hte right direction
        objfun3 = lambda x: (interpolate_val(x, inter_par))
        grad_objfun3 = lambda x:  interpolate_grad(x, inter_par)
        res = minimize(objfun3, x0, method='L-BFGS-B', jac=grad_objfun3, options={'gtol': 1e-6, 'disp': True})





# function[x
# y]=inter_min(x, inter_par)
# % find
# the
# minimizer
# of
# the
# interpolating
# function
# starting
# with x
#     global n
#     Ain
#     bin
#
# rho = 0.9; % parameters
# of
# backtracking
#
# % start
# the
# search
# with method
#     iter = 1;
#
# while iter < 10
#
# % Calculate
# the
# Newton
# direction
# H = zeros(n, n);
# g = zeros(n, 1);
# y = interpolate_val(x, inter_par);
# g = interpolate_grad(x, inter_par);
# H = interpolate_hessian(x, inter_par);
# % Perform
# the
# Hessian
# modification
# H = modichol(H, 0.01, 20);
# H = (H + H.
# ')/2;
# options = optimoptions('quadprog', 'Display', 'none');
# p = quadprog(double(H), double(g), Ain, bin - Ain * x, [], [], [], [], zeros(n, 1), options);
#
# if norm(p) < 1e-5
# break
# end
# a = 1;
# % Backtracking
# while 1
#     x1 = x + a * p;
#     y1 = interpolate_val(x1, inter_par);
#     if (y - y1) > 0
#         x = x1;
#     break
#     else
#     a = a * rho;
#     if norm(a * p) < 1e-4
#         break
#     end
#     end
#
# end
# iter = iter + 1;
# end
#

