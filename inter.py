import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
#from __future__ import print_function    # (at top of module)


class Inter:
    #    'Common base class for all variable'
    
    def __init__(self,inter_method, y0=0.):
        self.method = inter_method
        self.w = 0
        self.v = 0
        self.xi = 0
        self.yi=0
        self.lamda = 0
        self.y0 = y0
        self.Dw = 0
        self.Dv = 0
    
    
    def parameterization(self,xi, yi):
        n = xi.shape[0]
        m = xi.shape[1]
        if self.method =='NPS': # 1
            A= np.zeros(shape=(m,m))
            for ii in range(0,m,1): # for ii =0 to m-1 with step 1; range(1,N,1)
                for jj in range(0,m,1):
                    A[ii,jj] = (np.dot(xi[:,ii] - xi[:,jj],xi[:,ii] - xi[:,jj]))**(3.0/2.0)
            
            V = np.concatenate((np.ones((1,m)), xi), axis=0)
            A1 = np.concatenate((A, np.transpose(V)),axis=1)
            A2 = np.concatenate((V, np.zeros(shape=(n+1,n+1) )), axis=1)
            yi = yi[np.newaxis,:]
            
            print(yi.shape)
            b = np.concatenate([np.transpose(yi), np.zeros(shape=(n+1,1))])
            #      b = np.concatenate((np.transpose(yi), np.zeros(shape=(n+1,1) )), axis=0)
            A = np.concatenate((A1,A2), axis=0)
            wv = np.linalg.solve(A,b)
            self.w = wv[:m]
            self.v = wv[m:]
            self.xi = xi
        #      print(V)
        #     7
        if self.method == "MAPS" or self.method == "MAPS2":
            a = np.ones(xi.shape[0],1)
            lamda = 1;
            for i in range (0,20):
                [inter_par,a] = Scale_interpar()
                lamda = lamda/2.0
                for jj in range(yi.shape[0]):
                    [inter_par,a]  = Scale_interpar(); #method2 %%% DO I NEED TO INCLUDE SELF INPUT????
    
    
    
    
    
    
    # inter_par{1}=inter_method;
    # lambda = lambda/2;
    #for jj=1:numel(yi)
    # ygps(jj)= interpolate_val(xi(:,jj),inter_par); % HERE --->
    # % equation 19 in MAPS
    # deltaPx  = abs(ygps(jj)-yi(jj));
    # DeltaFun = abs(yi(jj)-y0);
    # % keyboard
    # if deltaPx/DeltaFun > 0.1
    # break;
    # elseif jj==numel(yi)
    #     return;
    # end
    # end
    # if inter_method==8
    # %      keyboard
    # epsFun = yi-y0;
    # inter_par{8}=epsFun;
    # end
    
    
    
    
    
    ###########################IMPORTANT ##################################################
    def DiagonalScaleCost(self, a):
        xi = self.xi
        yi = self.yi
        w = self.w
        #         inter_par= interpolateparametarization_scaled(xi,yi,a,1, lamda)
        w = self.w # ????????
        cost = np.sum(w**2) # ????  Cost =sum(w.^2);
        return cost
    def DiagonalScaleCost_der( self, a):
        w = self.w
        Dw = self.Dw
        gradCost =2*Dw*w;
        return gradCost
    
    ###########################IMPORTANT ##################################################
    # Scaled
    def Scale_interpar(self):
        #This function is for spinterpolation and finds the scaling factor for
        #polyharmonic spline interpolation
        xi = self.xi
        yi = self.yi
        n = xi.shape[0]
        a0 = np.ones((n,1))
        
        self.lamda =1
        lb = np.zeros((n,1))
        ub = np.ones((n,1))*n  #No upper or lower bounds
        
        res = minimize(self.DiagonalScaleCost, a0, method='L-BFGS-B', jac=self.DiagonalScaleCost_der, options={'gtol': 1e-6, 'disp': True},
                       bounds=(lb,ub))
                       
                       
                       
        return inter_par,res.x
    
    
    
    
    #    def interpolateparametarization_scaled(self, xi1,yi1,a, inter_method,lamda,interpolate_index):
    def parameterization_scaled(self,xi, yi, a, inter_method,lamda, interpolate_index):
        w = self.w
        n = xi.shape[0]
        m = xi.shape[1]
        if self.method =='NPS':
            A= np.zeros(shape=(m,m))
            
            for ii in range(0,m,1): # for ii =0 to m-1 with step 1; range(1,N,1)
                for jj in range(0,m,1):
                    A[ii,jj] = (np.dot(xi[:,ii] - xi[:,jj],xi[:,ii] - xi[:,jj]))**(3.0/2.0)
                    dA[ii,jj,:] =3/2.* (xi[:,ii] - xi[:,jj])**2 *  ((np.transpose(xi[:,ii]-xi[:,jj]))*H*(xi[:,ii] - xi[:,jj]))**(1/2.0)
            
            V = np.concatenate((np.ones((1,m)), xi), axis=0)
            A = A + np.identity(m)*lamda
            
            A1 = np.concatenate((A, np.transpose(V)),axis=1)
            A2 = np.concatenate((V, np.zeros(shape=(n+1,n+1) )), axis=1)
            yi = yi[np.newaxis,:]
            
            
            b = np.concatenate([np.transpose(yi), np.zeros(shape=(n+1,1))])
            #  b = np.concatenate((np.transpose(yi), np.zeros(shape=(n+1,1) )), axis=0)
            A = np.concatenate((A1,A2), axis=0)
            wv = np.linalg.solve(A,b)
            
            # calculating the gradient
            Dw = []; Dv=[]
            #for kk in range(n):
            #np.concatenate((dA[:,:,kk] np.zeros((V.T.shape))),axis=)
            #np.concatenate(())
            # b{kk} = -[dA(:,:,kk) zeros(size(V'));zeros(size(V)) zeros(n+1,n+1)]*wv; ???
            # Dwv = np.linalg.solve(A, b{kk} )  # Dwv = pinv(A)* b{kk}; % solve the associated linear system
            # Dw = [Dw; Dwv(1:N)']; ????
            # Dv = [Dv; Dwv(N+1:end)']; ?????
            self.Dw = Dw
            self.Dv = Dv
            self.a = a
            self.w = wv[:m]
            self.v = wv[m:]
            self.xi = xi
    
    
    
    def pred(self,x):
        if self.method == "NPS":
            w = self.w
            v = self.v
            xi = self.xi
            
            S = xi - x
            #             print np.dot(v.T,np.concatenate([np.ones((1,1)),x],axis=0)) + np.dot(w.T,np.sqrt(np.diag(np.dot(S.T,S))))**3
            return np.dot(v.T,np.concatenate([np.ones((1,1)),x],axis=0)) + np.dot(w.T,(np.sqrt(np.diag(np.dot(S.T,S)))**3))
        
        if self.method == "MAPS":
            w = self.w
            v = self.v
            xi = self.xi
            
            S = xi - x
            #             print np.dot(v.T,np.concatenate([np.ones((1,1)),x],axis=0)) + np.dot(w.T,np.sqrt(np.diag(np.dot(S.T,S))))**3
            return np.dot(v.T,np.concatenate([np.ones((1,1)),x],axis=0)) + np.dot(w.T,(np.sqrt(np.diag(np.dot(S.T,S)))**3))
    
    def grad(self, x):
        if self.method == "NPS" or self.method==1:
            w = self.w
            v = self.v
            n=x.shape[0]
            xi = self.xi
            N=xi.shape[1]
            g = np.zeros((n))
            for ii in range(N):
                X = x[:,0]-xi[:,ii]
                g = g + 3*w[ii]*X.T*np.linalg.norm(X)
            #                 print("--------------")
            #                 print v[ii]
            #             print(g)
            #             print("--------------")
            #             print(v[1:])
            g=g+v[1:,0]
            
            return g
        
        if self.method == "MAPS" or self.method==7:
            
            w = self.w
            v = self.v
            N=x.shape[1]
            xi = self.xi
            g = np.zeros((n, 1))
            for ii in range(N):
                X = x-xi[:,ii]
                g = g + 3*w[ii]*X*np.linalg.norm(X)
            g=g+v[1:,0]
            return g
    
    
    def hessian(self,x):
        n = x.shape[0]
        if self.method =="NPS" or self.method ==1:
            w=self.w;
            xi = self.xi;
            N = xi.shape[1]
            H = np.zeros((n,n))
            for ii in range(N):
                X = x - xi[:,ii]
                if np.linalg.norm(X) > 1e-5:
                    H = H + 3*w[ii]*((X*X.T)/np.linalg.norm(X)  +  np.linalg.norm(X)*np.identity(n))
            return H






def fun(x,  alpha=0.001):
    y = np.array((x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0)
    return y.T
#    return (x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0

xi = np.array([[0.5000 , 0.8000   , 0.5000,    0.2000,    0.5000],  [0.5000,    0.5000,    0.8000,    0.5000,    0.2000]])
#xi=np.random.rand(2,3)
x=np.array([[0.5],[0.53]])
#yi=np.random.rand(1,3)
yi=fun(xi)
print(yi)
print(xi)
#yi = np.array(yi)
print(yi.shape)
print(xi.shape)
inter_par = Inter("NPS")
inter_par.parameterization(xi,yi)
inter_par.pred(x)
inter_par.w
# inter_par.Scale_interpar()
H = inter_par.hessian(x)
print("-------H-------")
print(H)
g= inter_par.grad(x)
print("-------g-------")
print(g)