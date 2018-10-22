# title:        MULAN: A Blind and Off-Grid Method for Multichannel Echo Retrieval
# authors:      helena.peic.tukuljac@gmail.com, antoine.deleforge@inria.fr
# year:         2018
# license:      GPL v3
# description:  contains Finite Rate of Innovation implementation 
# reference:    https://lcav.epfl.ch/research/topics/sampling_FRI.html

import numpy as np
import numpy.linalg as la

import noise_tools

def get_annihilating_filter(A, n_Cadzow = 0): 
    #A = noise_tools.denoise(A, K, n_Cadzow)
    try: # we need to handle possible exceptions
        U, s, V = la.svd(A, full_matrices = False)
    except np.linalg.LinAlgError as e:
        mu, sigma = 0, 1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, A.shape[0])
        return noise/la.norm(noise)
    A_coeff = np.conjugate(V[-1, :]).reshape(-1, 1) # find it in the nullspace
    
    return np.squeeze(A_coeff)

def get_locations(filter_zeros, T, frequency_step):
    return np.mod((-np.angle(filter_zeros)/(2*np.pi*frequency_step)), T)
      
def get_weights(X, filter_zeros, K, pos_freq = False):
    if(pos_freq):
        E = np.flipud(np.transpose(np.vander(filter_zeros, len(X))))
        X = X.reshape(len(X), 1)
        return np.abs(la.lstsq(E, X)[0].reshape(K,)) # for removing the influence of the offset
    else:        
        return np.squeeze(np.linalg.solve(np.flipud(np.vander(filter_zeros, K).T), X.reshape(K,1)))