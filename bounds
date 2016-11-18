import numpy.matlib
def bounds(bnd1,bnd2,n):
    bnds = np.matlib.repmat(bnd2,  2**n,1)
    print(bnds)
    for ii in range(n-1):
        tt= np.mod(np.arange(2**n)+1, 2**(n-ii)) <= 2**(n-ii-1)-1
#         print(tt)
        for jj in range(len(tt)):
#             bnds[ii,np.mod(np.arange(2**n), 2**(n-ii)) <= 2**(n-ii-1)-1]=bnd1[ii];
#             print(tt[jj])
            if tt[jj]==True:
                print(jj)
                bnds[jj,ii]=bnd1[ii];
#                 print(bnds[ii,jj])
    return bnds
    
