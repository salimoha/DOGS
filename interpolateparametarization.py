import numpy as np


def interpolateparametarization(xi,yi,inter_method=1,interpolate_index=0):
    n = xi.shape[0]
    m = xi.shape[1]
    A= np.zeros(shape=(m,m))

    for ii in range(0,m,1): # for ii =0 to m-1 with step 1; range(1,N,1)
        for jj in range(0,m,1):
#            A[ii,jj] = (np.transpose(xi[:,ii] - xi[:,jj],axes=None)*(xi[:,ii] - xi[:,jj]))**(3.0/2.0)

             A[ii,jj] = (np.dot(xi[:,ii] - xi[:,jj],xi[:,ii] - xi[:,jj]))**(3.0/2.0)

    V = np.concatenate((np.ones((1,m)), xi), axis=0) 
    A1 = np.concatenate((A, np.transpose(V)),axis=1)
    A2 = np.concatenate((V, np.zeros(shape=(n+1,n+1) )), axis=1)
    b = np.concatenate((np.transpose(yi), np.zeros(shape=(n+1,1) )), axis=0)
    A = np.concatenate((A1,A2), axis=0)
    wv = np.linalg.solve(A,b)

#    print(wv)
    print(V)


xi=np.random.rand(2,3)

yi=np.random.rand(1,3)

interpolateparametarization(xi,yi,inter_method=1,interpolate_index=0)
