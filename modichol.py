import numpy as np


# def modichol(A,alpha,beta):
#     n = A.shape[1] # size of A
#     L = np.identity(n)
#     #################### EXTRA ???
#     D = np.zeros((n,1))
#     c = np.zeros((n,n)) #????
#     ######################
#     D[0] = np.max(np.abs(A[0,0]),alpha)
#     c[:,0]=A[:,0]
#     L[1:n,0]=c[1:n,0]/D[0]
#
#     for j in range(1,n-1):
#         c[j,j]=A[j,j]-(np.dot((L[j,0:j]**2).reshape(1,j), D[0:j]))[0, 0]
#         for i in range(j+1,n):
#             c[i,j]=A[i,j]-(np.dot((L[i,0:j]*L[j,0:j]).reshape(1,j), D[0:j]))[0, 0]
#         theta = np.max(c[j+1:n,j])
#         D[j]=np.array([(theta/beta)**2,np.abs(c[j,j]),alpha]).max()
#         L[j+1:n,j]=c[j+1:n,j]/D[j]
#     j=n-1;
#     c[j,j]=A[j,j]-(np.dot((L[j,0:j]**2).reshape(1,j), D[0:j]))[0, 0]
#     D[j]=np.max(np.abs(c[j,j]),alpha)
#     return np.dot(np.dot(L, (np.diag(np.transpose(D)[0]))), L.T)

A = np.array([[1.5756, 1.4679,0.4592], [1.4679, 1.5194, 0.7003], [0.4592, 0.7003,0.9425]])
# A = np.random.rand(4,4)
# A = A*A.T
import Utils as utl
A1 = utl.modichol(A,.3,0.1)
print(A1)

