import time
import pickle
import numpy as np
import operators as op
from cvxopt import solvers, matrix
from sklearn import covariance
from scipy.sparse.linalg import lsmr
import gc
from sklearn.covariance import MinCovDet

class fit_function:
    
    def __init__(self):
        self.prod_alpha_postA_length = 1
        self.fit_start_date = 20150901
        self.lookback = 2048
        self.tvrlb = 512
        
    def resimulate_alpha(self,data, aid):
        print('Alpha {} is resimulated in {}'.format(aid, data['universe']))
        alpha = data['load_alpha'](aid)
        simres = op.op_simulate(data, alpha, data['delay'], slippage=0, exevwap=0, exeopen=0)
        return simres

    def fit(self, data, filter_matrix):
        assert(np.all(np.isfinite(filter_matrix)))
        target = op.ts_delay(data['ret1'], -1-data['delay'])
        alpha_indices = np.where(np.any(filter_matrix, 1))[0]
        #alphapnl = np.zeros((data['alpha_list'].size, data['numdates']))
        
        #for ai in alpha_indices:
         #   simres = self.resimulate_alpha(data, data['alpha_list'][ai])
          #  alpha = data['load_alpha'](data['alpha_list'][ai])
           # alphapnl[ai] = np.nansum(alpha * target, 0)

        model_dict = {}
        for di, date in enumerate(data['dates']):
            if date <= self.fit_start_date: continue
            ddi = di-data['delay']
            if data['rebalance_dates_mask'][di]:
                print('Fitting on %d' % date)
                selected_alpha_indices = np.where(filter_matrix[:, ddi] == True)[0]
                if(selected_alpha_indices.shape[0]==0):
                    continue
                idx_start, idx_end = ddi-1-self.lookback, ddi-1
                
                selected_alpha_indices = selected_alpha_indices[:20]
                selected_alpha = alphapnl[selected_alpha_indices,idx_start:idx_end]
                
                X = alphapnl[selected_alpha_indices,idx_start:idx_end].T
                c = np.mean(X, 0)
                Q = compute_covariance_ledoit(X - np.mean(X,0).T)
                del X
                gc.collect()
                c_u,c_l = mean_limits(c)
                Q_u,Q_l = cov_limits(Q)
                # no short sale conditions
                A = np.identity(c.shape[0])
                n,m = A.shape[1],A.shape[0]
                b = np.ones((c.shape[0],1))/(1e9*c.shape[0])
                #select alpha and beta and then set the initial values
                alpha,beta = 0.1/np.sqrt(n**2+4*n+m),0.1
                x0 = initial_sol(Q,c)

                # x0 y0 t0 to be selected, check condition
                t0 = 1e-2
                grad = gradient(x0,c,Q,t0,A,b,c_u,c_l,Q_u,Q_l)
                hess = hessian(x0,c,Q,t0,A,b,c_u,c_l,Q_u,Q_l)
                print(grad.shape)
                print(hess.shape)
                function_decrement = np.sqrt(np.matmul(np.matmul(grad.T,np.linalg.pinv(hess)),grad))
                if(function_decrement <= beta ):
                    print("initial conditions satisfied")
                    print(function_decrement)
                    print(beta)
                else:
                    print("intial conditions violated")
                    print(function_decrement)
                    print(beta)
                    break
                prev = np.sum(x0)
                
                epsilon = 1e-1
                print("conditions")
                print(t0)
                k = 0
                tol = 1e-3
                print((1/epsilon)*(n**2+4*n+m)*(1+6*beta/np.sqrt(n**2+4*n+m)))
                while(t0 < (1/epsilon)*(n**2+4*n+m)*(1+6*beta/np.sqrt(n**2+4*n+m)) and k < 10):
                    print("loop")
                    print(k)
                    t0 = (1+alpha)*t0
                    print("calculating gradient")
                    grad = gradient(x0,c,Q,t0,A,b,c_u,c_l,Q_u,Q_l)
                    print("calculating hessian")
                    hess = hessian_approx(x0,c,Q,t0,A,b,c_u,c_l,Q_u,Q_l)
                    print("calculating step")
                    print(hess.shape)
                    newton = np.linalg.lstsq(hess,-grad[:2*n])[0]
                    #newton = lsmr(hess, -grad, maxiter=20)[0]
                    #newton = -np.matmul(np.linalg.inv(hess),grad)
                    del grad,hess
                    gc.collect()
                    upper = np.triu_indices(Q.shape[0])
                    #x0,c,Q[upper] = x0 + newton[0:n], c+newton[n:2*n], Q[upper]+newton[2*n:]
                    x0,c = x0 + newton[0:n], c+newton[n:2*n]
                    #Q.T[upper] = Q[upper]
                    k = k + 1
                    if(abs(np.sum(x0) - prev)<tol):
                        break
                    prev = np.sum(x0)
                    print(x0)
                    
                print(x0)
                w = x0

                model_dump = {'w':w,
                              'selected_alpha_indices':selected_alpha_indices
                             }
                model_dict[date] = pickle.dumps(model_dump)
                
        return model_dict, filter_matrix.copy()
        
        
    def construct_preA(self, data, model_dict, mode):
 
        for k in model_dict:
            model_dict[k] = pickle.loads(model_dict[k])
        
        if mode == 'last':
            numstocks = data['numstocks']
            if data['region'] == 'GLOBAL':
                s, e = data['si_map'][data['primary_region']]
                numstocks = e - s + 1
            preA_ld = np.zeros(numstocks, dtype=np.float32, order='F')
            model_dump = list(model_dict.values())[0]
            w, selected_alpha_indices= model_dump['w'], model_dump['selected_alpha_indices']
            for i, ai in enumerate(selected_alpha_indices):
                we = w[i]
                if we == 0: continue
                alpha_postA = data['load_alpha'](data['alpha_list'][ai])[:,-1-data['delay']]
                preA_ld += op.at_nan2zero(alpha_postA) * we
            return preA_ld

        if mode == 'full':
            date_di_map = dict(zip(data['dates'], range(data['numdates'])))
            refit_indices = sorted(model_dict.keys())
            awm = np.full((data['alpha_list'].size, data['numdates']), np.nan, dtype=np.float32, order='F')
            for i in range(len(refit_indices) - 1):
                di = date_di_map[refit_indices[i]]
                di_next = date_di_map[refit_indices[i + 1]]
                model_dump = model_dict[refit_indices[i]]
                w, selected_alpha_indices = model_dump['w'], model_dump['selected_alpha_indices']
                awm[selected_alpha_indices, di-data['delay']:di_next-data['delay']] = w.reshape(-1,1)
            di = date_di_map[refit_indices[-1]]
            model_dump = model_dict[refit_indices[-1]]
            w, selected_alpha_indices = model_dump['w'], model_dump['selected_alpha_indices']
            awm[selected_alpha_indices, di-data['delay']:] = w.reshape(-1,1)
            awm = np.nan_to_num(awm, copy=False)
            preA = np.zeros((data['numstocks'], data['numdates']), dtype=np.float32, order='F')

            for i in range(data['alpha_list'].size):
                we = awm[i]
                if np.all(we == 0): continue
                print("loading alpha ",data['alpha_list'][i])
                alpha_postA = data['load_alpha'](data['alpha_list'][i])
                preA += op.at_nan2zero(alpha_postA) * we
            return preA

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
    #kron = np.kron(Q,Q)[upper]
    hess[2*n:,2*n:] = hess[2*n:,2*n:] + np.diag(triu) #- kron
    del triu,upper,temp,xab2
    gc.collect()
    
    return hess

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
    """
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
    """
    return hess
    
def initial_sol(cov,mu):
    n = len(mu)
    P = 1e-2 * matrix(cov)
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
