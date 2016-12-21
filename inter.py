import numpy as np
# from scipy.optimize import minimize, rosen, rosen_der
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
            #a = ones(size(xi,1),1);
            lamda = 1;
            for i in range (0,20):
                [inter_par,a] = Scale_interpar(xi,yi,a0, lamda )
                lamda = lamda/2.0
                for jj in range(yi.shape[0]):
                    [inter_par,a]  = Scale_interpar( xi,yi,a, lamda); #method2 %%% DO I NEED TO INCLUDE SELF INPUT????
                     




    
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
    # Scaled
    def Scale_interpar(self, xi,yi,a0, lamda0):
        #This function is for spinterpolation and finds the scaling factor for
        #polyharmonic spline interpolation
        self.lamda = lamda0;
        n = xi.shape[0]
        a0 = ones(n,1)
        self.lamda =1;
        res = minimize(DiagonalScaleCost, x0, method='BFGS', jac=DiagonalScaleCost_der, options={'gtol': 1e-6, 'disp': True})
        #         res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'gtol': 1e-6, 'disp': True})
        res.x
        #         print(res.message)
        #         res.hess_inv
        
        
        # % options = optimoptions(@fmincon,'Algorithm','sqp','Display','iter-detailed' );
        # options = optimoptions(@fmincon,'Algorithm','sqp');
        # options = optimoptions(options,'GradObj','on');
        # lb = zeros(n,1); ub = ones(n,1)*n;   % No upper or lower bounds
        # % for ii=1:20
        # fung = @(a)DiagonalScaleCost(a,xi,yi);
        # % keyboard
        # [a,fval] = fmincon(fung,a0,[],[],ones(1,n),n,lb,ub,[],options);
        # % end
        # [ff,gf,inter_par] = DiagonalScaleCost(a,xi,yi);
        
        # end
        # %
        return inter_par,a
    
    
    ###########################IMPORTANT ##################################################
    def DiagonalScaleCost( self, a):
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
    
    # function [ Cost, gradCost, inter_par ] = DiagonalScaleCost( a, xi, yi)
    # %The Loss (cost) function that  how smooth the interpolating funciton is.
    # global lambda
    # % keyboard
    # inter_par= interpolateparametarization_scaled(xi,yi,a,1, lambda);
    # w = inter_par{2};
    # Cost =sum(w.^2);
    # % keyboard
    # if nargout>1
    # %The gradient of Loss (cost) function that indicates how smooth the interpolating funciton is.
    # Dw = inter_par{5};
    # gradCost =2*Dw*w;
    # end
    # inter_par{7}=a;
    # inter_par{1}=7;
    # %%%%%%%%%%%%%
    # end
    # %
    
    
    
    
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
            #      b = np.concatenate((np.transpose(yi), np.zeros(shape=(n+1,1) )), axis=0)
            A = np.concatenate((A1,A2), axis=0)
            wv = np.linalg.solve(A,b)
            
            # calculating the gradient
            Dw = []; Dv=[]
            for kk in range(n):
                #                 b{kk} = -[dA(:,:,kk) zeros(size(V'));zeros(size(V)) zeros(n+1,n+1)]*wv; ???
            #    Dwv = np.linalg.solve(A, b{kk} )  # Dwv = pinv(A)* b{kk}; % solve the associated linear system
            # Dw = [Dw; Dwv(1:N)']; ????
            # Dv = [Dv; Dwv(N+1:end)']; ?????
            self.Dw = Dw
            self.Dv = Dv
            self.a = a
            self.w = wv[:m]
            self.v = wv[m:]
            self.xi = xi
    
    # function inter_par= interpolateparametarization_scaled(xi1,yi1,a, inter_method,lambda,interpolate_index)
    # global xi yi y0 w
    # H= diag(a);
    # if nargin < 4
    # lambda = 0;
    # %lambda = 1e-3;
    # end
    # xi= xi1;
    # yi=yi1;
    # n=size(xi,1);
    # % keyboard
    # % polyharmonic spline interpolation
    # if inter_method==1
    #     N = size(xi,2); A = zeros(N,N);
    # for ii = 1 : 1 : N
    #     for jj = 1 : 1 : N
    #         A(ii,jj) = ((xi(:,ii) - xi(:,jj))' *H* (xi(:,ii) - xi(:,jj)))^(3 / 2);
    #         dA(ii,jj,:) =3/2.* (xi(:,ii) - xi(:,jj)).^2 *  ((xi(:,ii) - xi(:,jj))' *H* (xi(:,ii) - xi(:,jj)))^(1/2);
    #     end
    # end
    # % keyboard
    # V = [ones(1,N); xi1];
    # A = A + eye(N)*lambda;
    # A = [A V'; V zeros(n+1,n+1)];
    # %%%wv = pinv(A)* [yi.'; zeros(n+1,1)]; % solve the associated linear system
    # % keyboard
    # wv = A\[yi.'; zeros(n+1,1)];
    # %
    # % bb=[yi.'; zeros(n+1,1)], AA= A*A',  WV = AA\bb,   XX = A'*WV
    # % err = A*XX-[yi.'; zeros(n+1,1)]
    # inter_par{1}=1;
    # inter_par{2} = wv(1:N); inter_par{3} = wv(N+1:N+n+1);
    # inter_par{4}= xi1;
    # % calculating the gradient
    # Dw = []; Dv=[];
    # for kk=1:n
    # b{kk} = -[dA(:,:,kk) zeros(size(V')); zeros(size(V)) zeros(n+1,n+1)]*wv;
    # % Dwv = A \ b{kk}; % solve the associated linear system
    # % keyboard
    # Dwv = pinv(A)* b{kk}; % solve the associated linear system
    # Dw = [Dw; Dwv(1:N)'];
    # Dv = [Dv; Dwv(N+1:end)'];
    # end
    # inter_par{5} = Dw;
    # inter_par{6} = Dv;
    # inter_par{7} = a;
    # end
    # end
    # ###############################################################################
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


#y = v.T*np.concatenate([1, x]) + w.T*sqrt(diag(S' * S)) .^ 3
# ###############################################################################
      def interpolate_grad(self, x):
          if self.method == "NPS" || self.method==1:
             w = self.w 
             v = self.v
             N=x.shape[1]
             xi = self.xi 
             g = np.zeros((n, 1))
             for ii in range(N):
                 X = x-xi[:,ii]
                 g = g + 3*w[ii]*X*np.norm(X) # ??? norm???? 
             g = g + 3*w(ii)*X*np.norm(X)


          if self.method == "MAPS" || self.method==7:


# function g = interpolate_grad(x,inter_par)
# % Calculate te interpolatated value at points x
# % inter_par{1}=1 polyharmonic spline
# % inter_par{1}=2 Quadratic interpolation
# n=length(x);
# % polyharmonoic spline
# if inter_par{1}==1
#     w=inter_par{2}; v=inter_par{3};
#     xi=inter_par{4};
#     N = size(xi, 2);
#     g = zeros(n, 1);
# for ii = 1 : N
#     X = x - xi(:,ii);
#     g = g + 3 *w(ii)* X*norm(X);
# end
#     g=g+v(2:end);
# end
# % scaled polyharmonoic spline
# if inter_par{1}==7 || inter_par{1} == 8
#     w=inter_par{2}; v=inter_par{3};
#     xi=inter_par{4};  a = inter_par{7}; H = diag(a);
#     N = size(xi, 2);
#     g = zeros(n, 1);
# %     keyboard
# for ii = 1 : N
#     X = x - xi(:,ii);
# %     g = g + 3 *w(ii)* X'*norm(X);
#         g = g + 3*w(ii)*H*X*(X'*H*X).^(1/2);
#                term(ii,:)=  3*H*X*(X'*H*X).^(1/2);
# %       dA(ii,jj,:) =3/2.* (xi(:,ii) - xi(:,jj)).^2 *  ((xi(:,ii) - xi(:,jj))' *H* (xi(:,ii) - xi(:,jj)))^(1/2)
# end
#     g=g+v(2:end);
# end
# end

# ###############################################################################
# function H = interpolate_hessian(x,inter_par)
# n=length(x);
# % polyharmonoic spline
# if inter_par{1}==1
#     w=inter_par{2};
#     xi=inter_par{4};
#     N = size(xi, 2);
#     H = zeros(n);
# for ii = 1 : 1 : N
#     X = x - xi(:,ii);
#     if norm(X)>1e-5
#         H = H + 3 * w(ii) * ((X * X') / norm(X) + norm(X) * eye(n,n));
#     end
# end
# end
# % Scaled polyharmonoic spline
# if inter_par{1}==7 || inter_par{1} == 8
#     w=inter_par{2};
#     xi=inter_par{4};
#     a= inter_par{7}; S= diag(a);
#     N = size(xi, 2);
#     H = zeros(n);
# %     keyboard
# for ii = 1 : 1 : N
#     X = x - xi(:,ii);
#     if norm(X)>1e-5
# %         H = H + 3 * w(ii) * ((X * X') / norm(X) + norm(X) * eye(n,n));
#         H = H + 3 * w(ii) * ((S*X * X'*S) / (X'*S*X).^(1/2) + (X'*S*X).^(1/2) * eye(n,n));
#     end
# end
# end
# end

# function [x y]=inter_min(x, inter_par)
# %find the minimizer of the interpolating function starting with x
# global n Ain bin

# %keyboard
# rho=0.9; % parameters of backtracking

# % start the search with method
# iter=1;

# while iter<10

# % Calculate the Newton direction
# H=zeros(n,n); g=zeros(n,1);
# y=interpolate_val(x,inter_par);
# g=interpolate_grad(x,inter_par);
# H=interpolate_hessian(x,inter_par);
# % Perform the  modification
# H=modichol(H,0.01,20);
# H=(H+H.')/2;
# options=optimoptions('quadprog','Display','none');
# p=quadprog(double(H),double(g),Ain,bin-Ain*x,[],[],[],[],zeros(n,1),options);

# if norm(p)<1e-5
#     break
# end
# a=1;
# % Backtracking
# while 1
#     x1=x+a*p;
#         y1=interpolate_val(x1,inter_par);
#         if (y-y1)>0
#         x=x1;
#         break
#         else
#         a=a*rho;
#         if norm(a*p)<1e-4
#             break
#         end
#         end

# end
# iter=iter+1;
# end




def fun(x,  alpha=0.01):
    y = np.array((x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0)
    return y.T
#    return (x[0,:]-0.45)**2.0 + alpha*(x[1,:]-0.45)**2.0

xi = np.array([[0.5000 , 0.8000   , 0.5000,    0.2000,    0.5000],  [0.5000,    0.5000,    0.8000,    0.5000,    0.2000]])
#xi=np.random.rand(2,3)
x=np.array([[0.5],[0.5]])
#yi=np.random.rand(1,3)
yi=fun(xi)
print(yi)
#yi = np.array(yi)
print(yi.shape)
print(xi.shape)
inter_par = Inter("NPS")
inter_par.parameterization(xi,yi)
inter_par.pred(x)
inter_par.w
