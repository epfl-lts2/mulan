# title:        MULAN: A Blind and Off-Grid Method for Multichannel Echo Retrieval
# authors:      helena.peic.tukuljac@gmail.com, antoine.deleforge@inria.fr
# year:         2018
# license:      GPL v3
# description:  contains baseline methods - Cross-relation and Constrained LASSO

import numpy as np
from scipy.linalg import toeplitz
from numpy.linalg import pinv
from sklearn import linear_model

import logging
logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)4s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

# the baseline method used for benchmarking the algorithm both require the following parameters:
# x - multichannel measurements in time domain
# L - length of the filter
# F - sampling frequency

def LASSO(x, L, K, F):
    logger.info('LASSO started!')
    M = x.shape[1]
    h_lasso = LASSO_approach(x, L)
    tau_lasso, alpha_lasso = pick_echoes(h_lasso, K)
    
    mintau = min(tau_lasso[:, 0])
    tau_lasso = tau_lasso - mintau
    alpha_lasso = alpha_lasso/alpha_lasso[0, 0]
    
    h_lasso = np.concatenate((h_lasso[mintau.astype(int):, :], np.zeros((mintau.astype(int),M))), axis = 0)
    h_lasso = h_lasso/alpha_lasso[0, 0]
    tau_lasso = tau_lasso/F
    logger.info('LASSO done!')
    return tau_lasso, alpha_lasso, h_lasso
    
def CR(x, L, K, F):
    logger.info('Cross-relation started!')
    M = x.shape[1]
    h_cr = cross_relations(x, L)
    tau_cr, alpha_cr = pick_echoes(h_cr, K)
    
    mintau = min(tau_cr[:, 0])
    tau_cr = tau_cr - mintau
    alpha_cr = alpha_cr/np.max(alpha_cr)
    
    h_cr = np.concatenate((h_cr[mintau.astype(int):, :], np.zeros((mintau.astype(int), M))), axis = 0)
    h_cr = h_cr/alpha_cr[0, 0]
    tau_cr = tau_cr/F
    logger.info('Cross-relation done!')
    return tau_cr, alpha_cr, h_cr   

def LASSO_approach(x, L):
    # Topelitz matrices should be formed out of the measurements
    # Input:
    # x (N*M) : M channels input signal
    # L (int) : desired filters' length (L<N)
    # Output:
    # h_est (L*M) : estimated filters
    # we need to generalize it for a case of M microphones
    L = L - 1
    M = x.shape[1]
    x1 = x[:, 0]
    x2 = x[:, 1]
    #Toeplitz matrices for each signal:
    T1 = toeplitz(x1[L:], x1[L::-1])
    T2 = toeplitz(x2[L:], x2[L::-1])
    # Combined matrix:
    A = np.concatenate((T2, -T1), axis=1)
    #A = np.reshape(Toep_matrices, )
    a1 = A[:, 0]
    A2 = A[:, 1:]
    # we solve the problem with LASSO
    clf = linear_model.Lasso(alpha=0.001)
    clf.fit(A2, -a1)
    y = clf.coef_
    y = np.reshape(y, (len(y), 1))
    x = np.concatenate((np.reshape(np.array([1]), (1,1)), y), axis=0) 
    h1 = x[0:(L+1)]
    h2 = x[(L+1):]
    h_est = np.concatenate((h1, h2), axis=1)
    return h_est

def cross_relations(x, L):
    # Input:
    # x (N*2) : 2 channels input signal
    # L (int) : desired filters' length (L<N)
    # Output:
    # h_est (L*2) : estimated filters
    x1 = x[:, 0]
    x2 = x[:, 1]
    #Toeplitz matrices for each signal:
    L = L - 1
    T1 = toeplitz(x1[L:], x1[L::-1])
    T2 = toeplitz(x2[L:], x2[L::-1])
    # Combined matrix:
    T = np.concatenate((T2, -T1), axis=1)
    tt = T[:, 0] # first column
    Tt = T[:, 1:] # rest
    # Estimate filters:
    ht = np.matmul(-pinv(Tt), tt)
    ht = np.reshape(ht, (len(ht), 1))
    h1 = np.concatenate((np.reshape(np.array([1]), (1,1)), ht[0:L]), axis=0)
    h2 = ht[L:]
    h_est = np.concatenate((h1, h2), axis=1)
    return h_est

def pick_echoes(h, K):
    # Input:
    # h (L*M) : some filters
    # Output:
    # tau (K*M) : the echo locations (in samples)
    # alpha (K*M) : the echo amplitudes
    L = h.shape[0]
    M = h.shape[1]
    tau = np.zeros((K,M))
    alpha = np.zeros((K,M), dtype=np.complex128)
    for m in range(M):
        # Pick all local maxima
        loc_argmax = np.array([0, L-1])#1, L])
        for l in range(1,L-1):
            if( h[l-1,m] < h[l,m] and h[l+1,m] < h[l,m] ):
                loc_argmax = np.append(loc_argmax, l)
        # Keep only K greatest
        loc_max = h[loc_argmax.astype(int), m]
        sorted_idx = (-loc_max).argsort()
        #ignore, sorted_idx = sort(loc_max,reverse = true)
        location = loc_argmax[sorted_idx[0:K]]
        weight = loc_max[sorted_idx[0:K]]
        tau[0:len(location), m] = location
        alpha[0:len(weight), m] = weight
        # Sort in time:
        sorted_idx = tau[:, m].argsort()
        tau[:, m] = np.sort(tau[:, m])
        alpha[:, m] = alpha[sorted_idx, m]
    return tau, alpha