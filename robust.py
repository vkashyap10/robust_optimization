import numpy as np
from sklearn import covariance
# rotational invariant estimator with iw regularization
from numpy import linalg as LA
import gc
from cvxopt import solvers, matrix
lookback = 250

# keep booksize to be 100X100 units (100 units for short sale and 100 units for long)
def fit(data):
    lookback = 250
    profit_chart = np.zeros((data.shape[1],1))
    mvoprofit = np.zeros((data.shape[1],1))
    import matplotlib.pyplot as plt
    booksize = 100.0

    # w stores portfoilio weights after robust optimization
    w = np.ones((data.shape[0],1))
    # w0 stores portfolio weights for baseline ( valina mvo )
    w0 = np.ones((data.shape[0],1))
    # loop for advancing in time
    for i in range(data.shape[1]-1):
        # we need lookback data
        if i <= lookback: continue
        # we fit the model to data after each 78 stock days to get updated weights which we use for next 78 days
        if i%78 == 0:
            print('Fitting on %d' % i)
            idx_start, idx_end = i-lookback, i

            # X stores stock data
            X = data[:,idx_start:idx_end].T
            # c is mean for mean variacne optimization
            c = np.mean(X, 0)
            # covariance for mean variance optimization using ledoit wolf for better estimation of covariance 
            #Q = compute_covariance_ledoit(X - np.mean(X,0).T)
            Q = np.cov((X - np.mean(X,0).T).T)
            x0 = initial_sol(Q,c)
            x_mvo = vanila_mvo(Q,c)
            print("vanila solution")
            print(x_mvo)
            #Q = compute_covariance(X - np.mean(X,0).T)
            del X
            gc.collect()
            # defining box constraints

            c = -1*c
            c_u,c_l = mean_limits(c)
            Q_u,Q_l = cov_limits(Q)
            # setting constraint Ax = b
            A = np.identity(c.shape[0])
            n,m = A.shape[1],A.shape[0]
            b = -np.ones((c.shape[0],1))
            #select alpha and beta and then set the initial values (given in report, algorithm)
            alpha,beta = 0.1/np.sqrt(n**2+4*n+m) , 0.1

            # setting weights for baseline model
            w0 = x_mvo[:,np.newaxis]
            print("sum of weights")
            print(np.sum(w0))
            #print("initial solution")
            #print(x0)
            # x0 y0 t0 to be selected, check condition
            # set initial value of t0 for interior point method
            t0 = 20
            # gradient and hessian calculation for newton step
            grad = gradient(x0,c,Q,t0,A,b,c_u,c_l,Q_u,Q_l)
            hess = hessian(x0,c,Q,t0,A,b,c_u,c_l,Q_u,Q_l)
            # calculating step using least square method
            delx = np.linalg.lstsq(hess[:1000],-grad[:1000])[0]
            # calculating function decrement, this has to be less for closeness to central path.
            print("function_decrement")
            #function_decrement = np.sqrt(np.matmul(np.matmul(grad.T,np.linalg.pinv(hess)),grad))
            F = -np.sum(np.log(abs(np.matmul(A,x0) - b)))
            upper = np.triu_indices(Q.shape[0])
            G = -np.sum(np.log(c_u-c)) - np.sum(np.log(c - c_l)) - np.sum(np.log(Q_u - Q)[upper])- np.sum(np.log(Q - Q_l)[upper]) - np.log(np.linalg.det(Q))
            function_decrement = t0*(np.matmul(c.T,x0) + np.matmul(np.matmul(x0.T,Q),x0)) + F -G

            if(function_decrement <= beta ):
                print("initial conditions satisfied")
                print(function_decrement)
                print(beta)
            else:
                print("intial conditions violated")
                print(function_decrement)
                print(beta)
                print(t0*(np.matmul(c.T,x0) + np.matmul(np.matmul(x0.T,Q),x0)))
                print(F)
                print(-G)
                print(np.sum(np.log(c_u-c)))
                print(np.sum(np.log(c - c_l)))
                print(np.sum(np.log(Q_u - Q)[upper]))
                print(np.sum(np.log(Q - Q_l)[upper]))
                print(np.log(np.linalg.det(Q)))
                break

            # storing to calculate sum-absolute change in weight distribution and defining tolerance for stopping
            prev = np.sum(x0)
            tol = 1e-3
            # epsilon accuracy for duality gap
            epsilon = 1
            # counter for iterations
            k = 0
            # this while condition checks if duality gap is less than epsilon(accuracy wanted)
            while(t0 < (1/epsilon)*(n**2+4*n+m)*(1+6*beta/np.sqrt(n**2+4*n+m))):
                t0 = (1.05+alpha)*t0
                grad = gradient(x0,c,Q,t0,A,b,c_u,c_l,Q_u,Q_l)
                hess = hessian(x0,c,Q,t0,A,b,c_u,c_l,Q_u,Q_l)
                # control step size if needed
                #step = 1.0/(k+1)**3
                step = 1
                newton = step*np.linalg.lstsq(hess[:1000],-grad[:1000])[0]
                del grad,hess
                gc.collect()

                upper = np.triu_indices(Q.shape[0])
                x0,c,Q[upper] = x0 + newton[0:n], c+newton[n:2*n], Q[upper]+newton[2*n:]
                Q.T[upper] = Q[upper]
                
                # sum absolute difference weights of 2 iterations
                curr = abs(np.sum(x0) - prev)
                #  can sotp according to curr
                #if(abs(np.sum(x0) - prev)<tol):
                #    break
                # updating iterate
                k = k + 1
                prev = np.sum(x0)
                print("sum absolute wieght difference between last two iterations ")
                print(curr) 
                print("duality gap")
                print((n**2+4*n+m)*(1+6*beta/np.sqrt(n**2+4*n+m))/t0)
            
            w = x0[:,np.newaxis]
            print("sum of weights")
            print(np.sum(abs(w)))
            print(w)

        # balance portfolio to fit booksize
        long_index = np.multiply(data[:,i][:,np.newaxis],w) > 0
        long_book = np.sum((data[:,i][:,np.newaxis]*w)[long_index])
        short_index = np.multiply(data[:,i][:,np.newaxis],w) < 0
        short_book = abs(np.sum((data[:,i][:,np.newaxis]*w)[short_index]))
        #w[long_index] = w[long_index]/long_book*booksize
        #w[short_index] = w[short_index]/short_book*booksize
        print(data[:,i][:,np.newaxis]*w)
        print(long_book)
        print(short_book)
        w = w/(long_book+short_book)*booksize

        # balancing for baseline
        long_index = np.multiply(data[:,i][:,np.newaxis],w0) > 0
        long_book = np.sum((data[:,i][:,np.newaxis]*w0)[long_index])
        short_index = np.multiply(data[:,i][:,np.newaxis],w0) < 0
        short_book = abs(np.sum((data[:,i][:,np.newaxis]*w0)[short_index]))
        #w0[long_index] = w0[long_index]/long_book*booksize
        #w0[short_index] = w0[short_index]/short_book*booksize
        print(data[:,i][:,np.newaxis]*w0)
        print(long_book)
        print(short_book)
        w0 = w0/(long_book+short_book)*booksize
        # calculating profits 
        #profit_chart[i] = np.sum((data[:,i+1]+data[:,i+1]*(np.random.rand(1)[0]-0.5)*10/100.0 - data[:,i])*w) + profit_chart[i-1]
        #mvoprofit[i] =np.sum((data[:,i+1]+data[:,i+1]*(np.random.rand(1)[0]-0.5)*10/100.0 - data[:,i])*w0) + mvoprofit[i-1]
        profit_chart[i] = np.sum((data[:,i+1]- data[:,i])*w) + profit_chart[i-1]
        mvoprofit[i] =np.sum((data[:,i+1]- data[:,i])*w0) + mvoprofit[i-1]
        print("profit of the day")
        print("robust case")
        print(profit_chart[i])
        print("vanila mvo")
        print(mvoprofit[i])
    return profit_chart,mvoprofit
        
        
   
# ledoit_wolf for true covariance estimation
def compute_covariance_ledoit(X):
    shrunk_cov = covariance.ledoit_wolf(X,assume_centered = False)
    shrunk_cov = np.array(shrunk_cov[0])
    return shrunk_cov

def mean_limits(c):
    limit = 3/100.0*c
    return c+abs(limit) , c-abs(limit)

def cov_limits(Q):
    limit = Q*3/100.0
    return Q+abs(limit),Q-abs(limit)
                      
def gradient(x,c,Q,t,A,b,c_u,c_l,Q_u,Q_l):
    n = x.shape[0]
    grad = np.zeros(int(2*n+n*(n+1)/2))
    xab = np.reciprocal(np.matmul(x,A) - np.squeeze(b))
    grad[0:n] = grad[0:n] + t*(c+np.matmul(Q,x)) - np.sum(A*xab,0)
    grad[n:2*n] = t*x + np.reciprocal(c_u - c) - np.reciprocal(c-c_l)
    temp = t*1/2*np.matmul(x,x.T) + np.reciprocal(Q_u-Q) - np.reciprocal(Q-Q_l) - np.linalg.pinv(Q)
    upper = np.triu_indices(temp.shape[0])
    triu = temp[upper]
    grad[2*n:] = triu
    del triu,upper,temp,xab,n
    gc.collect()
    return grad

def hessian(x,c,Q,t,A,b,c_u,c_l,Q_u,Q_l):
    n = x.shape[0]
    hess = np.zeros((int(2*n+n*(n+1)/2),int(2*n+n*(n+1)/2)))
    xab2 = np.square(np.reciprocal(np.matmul(x.T,A) - b))*A
    # grad x wrt everthing, in first n rows
    # wrt x
    hess[0:n,0:n] = t*Q - xab2*A
    #wrt c
    hess[0:n,n:2*n] = t*np.identity(n)
    #wrt Q
    temp = np.tile(x,(n,1))
    upper = np.triu_indices(n)
    triu = temp[upper]
    hess[0:n,2*n:] = hess[0:n,2*n:] + t*triu
    
    # row n to 2n
    #wrt x
    hess[n:2*n,0:n] = hess[n:2*n,0:n] + t*np.identity(n)
    #wrt c
    temp = np.reciprocal((c_u - c)**2) + np.reciprocal((c - c_l)**2)
    hess[n:2*n,n:2*n] = hess[n:2*n,n:2*n] + np.diag(temp)
    
    #row 2n and onwards
    #wrt x
    hess[2*n:,0:n] = hess[2*n:,0:n] + t*x.T
    
    #wrt Q
    temp = np.reciprocal((Q_u - Q)**2) + np.reciprocal((Q - Q_l)**2)
    upper = np.triu_indices(temp.shape[0])
    triu = temp[upper]
    kron = np.kron(Q,Q)[upper]
    hess[2*n:,2*n:] = hess[2*n:,2*n:] + np.diag(triu) - kron
    del triu,upper,temp,xab2
    gc.collect()
    
    return hess


    
def initial_sol(cov,mu):
    n = len(mu)
    P = 15*matrix(cov)
    q = matrix(-mu)
    w_min = -np.zeros(n)
    G = matrix(np.vstack([-np.eye(n)]))
    h = matrix(np.concatenate([w_min])) 
    A = matrix(np.array([[1.0]*n]))
    b = matrix(np.array([1.0]))
    solvers.options['show_progress'] = False
    solvers.options['refinement'] = 2
    solvers.options['feastol'] = 1e-12
    solvers.options['abstol'] = 1e-12
    solvers.options['reltol'] = 1e-12
    sol = solvers.qp(P, q , G,h,A,b)
    w = np.array(sol['x']).ravel()
    return w

def vanila_mvo(cov,mu):
    n = len(mu)
    P = 15*matrix(cov)
    q = matrix(-mu)
    w_min = np.ones(n)
    G = matrix(np.vstack([-np.eye(n)]))
    h = matrix(np.concatenate([w_min])) 
    A = matrix(np.array([[1.0]*n]))
    b = matrix(np.array([1.0]))
    solvers.options['show_progress'] = False
    solvers.options['refinement'] = 2
    solvers.options['feastol'] = 1e-12
    solvers.options['abstol'] = 1e-12
    solvers.options['reltol'] = 1e-12
    sol = solvers.qp(P, q , G,h)
    w = np.array(sol['x']).ravel()
    return w



"""



def hessian_approx(x,c,Q,t,A,b,c_u,c_l,Q_u,Q_l):
    n = x.shape[0]
    hess = np.zeros((int(2*n),int(2*n)))
    xab2 = np.square(np.reciprocal(np.matmul(x.T,A) - b))*A
    # grad x wrt everthing, in first n rows
    # wrt x
    hess[0:n,0:n] = t*Q - xab2*A
    #wrt c
    hess[0:n,n:2*n] = t*np.identity(n)
    #wrt Q
    #temp = np.tile(x,(n,1))
    #upper = np.triu_indices(n)
    #triu = temp[upper]
    #hess[0:n,2*n:] = hess[0:n,2*n:] + t*triu
    
    # row n to 2n
    #wrt x
    hess[n:2*n,0:n] = hess[n:2*n,0:n] + t*np.identity(n)
    #wrt c
    temp = np.reciprocal((c_u - c)**2) + np.reciprocal((c - c_l)**2)
    hess[n:2*n,n:2*n] = hess[n:2*n,n:2*n] + np.diag(temp)
    
    #row 2n and onwards
    #wrt x
    hess[2*n:,0:n] = hess[2*n:,0:n] + t*x.T
    
    #wrt Q
    temp = np.reciprocal((Q_u - Q)**2) + np.reciprocal((Q - Q_l)**2)
    upper = np.triu_indices(temp.shape[0])
    triu = temp[upper]
    #kron = np.kron(Q,Q)[upper]
    hess[2*n:,2*n:] = hess[2*n:,2*n:] + np.diag(triu) #- kron
    del triu,upper,temp,xab2
    gc.collect()
    
    return hess

def g_iw(z,q,k):
    lambda_p = ((1+q)*k + np.sqrt((2*k + 1)*(2*q*k + 1)))/k
    lambda_n = ((1+q)*k - np.sqrt((2*k + 1)*(2*q*k + 1)))/k
    return (z*(1+k) - k*(1-q) - np.sqrt(z-lambda_p)*np.sqrt(z- lambda_n))/(z*(z+2*q*k))

def rie(z,q,g):
    return z.real/(abs(1-q + q*z*g))

def denoising_rie(N,q,eigv):
    
    k = 2*eigv[-1]/((1-q-eigv[-1])**2 - 4*q*eigv[-1])
    k = 1
    alpha = 1/(1 + 2*q*k)
    xi = np.zeros(eigv.shape)
    for i in range(N):
        z = eigv[i] - 1j/np.sqrt(N)
        temp = 0
        for j in range(1,N):
            if(j==i):
                continue
            temp = temp + 1/(z - eigv[j])
        g = temp/(N-1)
        xi[i] = rie(z,q,g)
        g = g_iw(z,q,k)
        tau = (1+alpha*(eigv[i]-1))/rie(z,q,g)
        if(tau>1 and eigv[i]<1):
            xi[i] = tau*xi[i]
    s = np.sum(eigv)/np.sum(xi)
    return s*xi
    
def compute_covariance(X):
    N = X.shape[1]
    sample_cov = np.cov(X.T)
    eigv, eigvec = LA.eigh(sample_cov)
    eigvec = eigvec.T   # since eigenvector returned are column-wise, now they are row-wise
    sort_index = np.argsort(-eigv)  # negatiion for descending
    eigv = eigv[sort_index]
    eigvec = eigvec[sort_index]
    cov = np.zeros((N,N))
    q = N/X.shape[0]
    print("eig")
    print("--------------------")
    print(eigv)
    print("reig")
    xi = denoising_rie(N,q,eigv)
    print(xi)
    print("--------------------")
    for i in range(N):
        cov = cov + xi[i] * np.matmul(eigvec[i][:,np.newaxis],eigvec[i][:,np.newaxis].T)
    return cov
"""
data = np.load("first100.npy",allow_pickle = True)
data = data.astype(float)
robust,vanila = fit(data)

import matplotlib.pyplot as plt
t = t = np.linspace(0, robust.shape[0], robust.shape[0])
plt.plot(t[lookback:-2], robust[lookback:-2], 'r') # plotting t, a separately 
plt.plot(t[lookback:-2], vanila[lookback:-2], 'b')
plt.show()
