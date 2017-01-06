# class Interpolate:
# #    'Common base class for all employees'
# #    empCount = 0
#
#     def __init__(self,inter_method):
#         self.method = inter_method
#         self.w = 0
#         self.v = 0
#         self.xi = 0
# #        self.lambda = 0
#
#
#     def parameterization(self,xi, yi):
#         n = xi.shape[0]
#         m = xi.shape[1]
#         if self.method =='NPS':
#
#             A= np.zeros(shape=(m,m))
#
#             for ii in range(0,m,1): # for ii =0 to m-1 with step 1; range(1,N,1)
#                 for jj in range(0,m,1):
#                     A[ii,jj] = (np.dot(xi[:,ii] - xi[:,jj],xi[:,ii] - xi[:,jj]))**(3.0/2.0)
#
#             V = np.concatenate((np.ones((1,m)), xi), axis=0)
#             A1 = np.concatenate((A, np.transpose(V)),axis=1)
#             A2 = np.concatenate((V, np.zeros(shape=(n+1,n+1) )), axis=1)
#             yi = yi[np.newaxis,:]
#
#             print(yi.shape)
#             b = np.concatenate([np.transpose(yi), np.zeros(shape=(n+1,1))])
#     #      b = np.concatenate((np.transpose(yi), np.zeros(shape=(n+1,1) )), axis=0)
#             A = np.concatenate((A1,A2), axis=0)
#             wv = np.linalg.solve(A,b)
#             self.w = wv[:m]
#             self.v = wv[m:]
#             self.xi = xi
#     #      print(V)
#
#         if self.method == "MAPS" or self.method == "MAPS2":
#             a = np.ones(xi.shape[0],1)
#             #a = ones(size(xi,1),1);
#             lambda = 1;
#             for i in range (0,20):
#             # %TODO: FIX THE ITER MAX
#             #     %    [inter_par, a]  = Scale_interpar( xi,yi,a0, lambda); % method1
#             # [inter_par,a]  = Scale_interpar( xi,yi,a, lambda); %method2
#             # inter_par{1}=inter_method;
#             # lambda = lambda/2;
#             # end
#             # if inter_method==8
#             # %      keyboard
#             # epsFun = yi-y0;
#             # inter_par{8}=epsFun;
#             # end
#
#         # Scaled Polyharmonic Spline
#     def Scale_interpar(xi,yi,a0, lambda0):
#             #This function is for spinterpolation and finds the scaling factor for
#             #polyharmonic spline interpolation
#         self.lambda = lambda0;
#         n = xi.shape[0]
#         a0 = ones(n,1)
#         self.lambda =1;
#
#         return inter_par,a
#
import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt
class Inter_par():
    def __init__(self,method="NPS", w=0, v=0, xi=0,a=0):
        self.method = "NPS"
        self.w=[]
        self.v=[]
        self.xi=[]
        self.a=[]
        
def interpolateparameterization(xi, yi, inter_par):
    n = xi.shape[0]
    m = xi.shape[1]
    if inter_par.method =='NPS':
        A= np.zeros(shape=(m,m))
        for ii in range(0,m,1): # for ii =0 to m-1 with step 1; range(1,N,1)
            for jj in range(0,m,1):
                A[ii,jj] = (np.dot(xi[:,ii] - xi[:,jj],xi[:,ii] - xi[:,jj]))**(3.0/2.0)

        V = np.concatenate((np.ones((1,m)), xi), axis=0)
        A1 = np.concatenate((A, np.transpose(V)),axis=1)
        A2 = np.concatenate((V, np.zeros(shape=(n+1,n+1) )), axis=1)
        yi = yi[np.newaxis,:]
        # print(yi.shape)
        b = np.concatenate([np.transpose(yi), np.zeros(shape=(n+1,1))])
        #      b = np.concatenate((np.transpose(yi), np.zeros(shape=(n+1,1) )), axis=0)
        A = np.concatenate((A1,A2), axis=0)
        wv = np.linalg.solve(A,b)
        inter_par.w = wv[:m]
        inter_par.v = wv[m:]
        inter_par.xi = xi
        return inter_par
        #      print(V)

def fun(x,  alpha=0.01):
    y = np.array((x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0)
    return y.T
#    return (x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0


inter_par = Inter_par("NPS")
xi = np.array([[0.5000 , 0.8000   , 0.5000,    0.2000,    0.5000],  [0.5000,    0.5000,    0.8000,    0.5000,    0.2000]])
#xi=np.random.rand(2,3)
x=np.array([[0.5],[0.5]])
#yi=np.random.rand(1,3)
yi=fun(xi)
print yi
#yi = np.array(yi)
print(yi.shape)
print(xi.shape)

inter_par = interpolateparameterization(xi, yi, inter_par)