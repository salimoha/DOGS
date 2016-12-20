import numpy as np

def circhyp(x,N):
#circhyp     Circumhypersphere of simplex
#   [xC, R2] = circhyp(x, N) calculates the coordinates of the circumcenter
#   and the square of the radius of the N-dimensional hypersphere
#   encircling the simplex defined by its N+1 vertices.
#   Author: Shahoruz Alimohammadi
#   Modified: Dec., 2016
#   DELTADOGS package
    
	test = np.sum(np.transpose(x)**2,axis=1)
	test = test[:, np.newaxis]
	m1 = np.concatenate(( np.matrix((x.T**2).sum(axis=1)), x))
	M = np.concatenate(( np.transpose(m1),   np.matrix(np.ones((N+1,1)))  ), axis=1)
	a = np.linalg.det(M[ :,1:N+2 ]  )
	c = (-1.0) ** (N+1) * np.linalg.det(M[:,0:N+1])
	D = np.zeros((N, 1))
	for ii in range(N):
		M_tmp = np.copy(M)
		M_tmp = np.delete(M_tmp, ii+1, 1)
		D[ii] = ((-1.0) ** (ii+1)) * np.linalg.det(M_tmp)
    #print(np.linalg.det(M_tmp))
	#print(D)
	xC = -D / (2.0 * a)
	print(xC)
	R2 = (np.sum(D**2,axis=0) - 4 * a * c) / (4.0 * a ** 2)
	print(R2)
	return R2, xC

x = np.array([[0.6443,    0.8116,    0.3507], [0.3786,    0.5328,    0.9390]]);
N = 2;

[R2,xC] = circhyp(x, N)
