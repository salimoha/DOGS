import numpy as np

class Interpolate:
   'Common base class for all employees'
   empCount = 0

   def __init__(self,inter_method):
      self.method = inter_method
            
   
   def parameterization(self,xi, yi):
      n = xi.shape[0]
      m = xi.shape[1]
      A= np.zeros(shape=(m,m))     
#      print(A)

      for ii in range(0,m,1): # for ii =0 to m-1 with step 1; range(1,N,1)
         for jj in range(0,m,1):
             A[ii,jj] = (np.dot(xi[:,ii] - xi[:,jj],xi[:,ii] - xi[:,jj]))**(3.0/2.0)

      V = np.concatenate((np.ones((1,m)), xi), axis=0) 
      A1 = np.concatenate((A, np.transpose(V)),axis=1)
      A2 = np.concatenate((V, np.zeros(shape=(n+1,n+1) )), axis=1)
      print(yi.T)
      print(np.zeros(shape=(n+1,1) ))
#      b = np.vstack([yi.T, np.zeros(shape=(n+1,1) )])
      b = np.concatenate([np.transpose(yi), np.zeros(shape=(n+1,1))]) 
#      b = np.concatenate((np.transpose(yi), np.zeros(shape=(n+1,1) )), axis=0)
      A = np.concatenate((A1,A2), axis=0)
      wv = np.linalg.solve(A,b)

      print(wv)
#      print(V)



#   def displayMethod(self):
#      print "Name : ", self.inter_method,  "




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
    print(yi)
    b = np.concatenate((np.transpose(yi), np.zeros(shape=(n+1,1) )), axis=0)
    A = np.concatenate((A1,A2), axis=0)
    wv = np.linalg.solve(A,b)

#    print(wv)
    print(V)

def fun(x,  alpha=0.001):
    y = np.array((x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0)
    return y.T
#    return (x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0

#xi = np.array([[0.5000 , 0.8000   , 0.5000,    0.2000,    0.5000],  [0.5000,    0.5000,    0.8000,    0.5000,    0.2000]])
xi=np.random.rand(2,3)
x=np.array([[0.5],[0.5]])
yi=np.random.rand(1,3)
yi=fun(xi)
#yi = np.array(yi)
print(yi.shape)
print(xi.shape)
inter_par = Interpolate("NPS")
inter_par.parameterization(xi,yi)




#interpolateparametarization(xi,yi,inter_method=1,interpolate_index=0)
