import numpy as np


def interpolateparametarization(xi,yi,inter_method,interpolate_index):
    n = xi.shape[0]
    m = xi.shape[1]
    A= np.zeros(shape=(m,m))

    for ii in range(0,m,1): # for ii =0 to m-1 with step 1; range(1,N,1)
        for jj in range(0,m,1):
            A[ii,jj] = (np.transpose(xi[:,ii] - xi[:,jj],axes=None)*(xi[:,ii] - xi[:,jj]))**(3.0/2.0)

    print(A)