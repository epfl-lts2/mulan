# title:        MULAN: A Blind and Off-Grid Method for Multichannel Echo Retrieval
# authors:      helena.peic.tukuljac@gmail.com, antoine.deleforge@inria.fr
# year:         2018
# license:      GPL v3
# description:  contains method for Cadzow denoising
# reference:    describes how to apply Cadzow denoising to FRI https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4472241

import numpy as np
import numpy.linalg as la
import scipy.linalg as lam

def add_white_noise(x, SNR_dB):
    '''
    Adds white Gaussian noise with the given SNR
    '''    
    # equations:
    # SNR_dB = 10log10(P_signal/P_noise)
    # P_noise = P_random_noise * scaling_factor
    # SNR_dB = 10log10(P_signal/(P_random_noise*scaling_factor))
    # 10^(SNR_dB/10) = P_signal/(P_random_noise*scaling_factor)
    # observe reciprocal values:
    # 10^(-SNR_dB/10) = (P_random_noise*scaling_factor)/P_signal
    # scaling_factor = (P_signal/P_random_noise)*10^(-SNR_dB/10)
    # required_noise = random_noise*sqrt(scaling_factor)
    L = len(x)
    random_noise = np.random.randn(1,L)
    random_noise = random_noise.reshape(x.shape)
    P_signal = sum(abs(x)**2)/L
    P_random_noise = sum(abs(random_noise)**2)/L
    scaling_factor = (P_signal/P_random_noise)*(10**(-float(SNR_dB)/10))
    required_noise = np.sqrt(scaling_factor)*random_noise
    if((np.real(x) == x).all()):
        x_with_noise = x + required_noise
    else:
        x_with_noise = x + required_noise*(1/np.sqrt(2) + 1j/np.sqrt(2))
    return x_with_noise

def denoise(A, K, n_Cadzow = 0):
    # run Cadzow denoising
    for cadzow_loop in range(n_Cadzow):
        # low-rank projection
        [U,s,V] = la.svd(A, full_matrices=False)
        s[-1] = 0
        A = U @ np.diag(s) @ V

        # enforce Toeplitz structure
        z = np.zeros((A.shape[0] + A.shape[1] - 1, 1), dtype=np.complex128)
        for i in range(z.shape[0]):
            z[i] = np.mean(np.diag(A, K - i))
        A = lam.toeplitz(z[K:], z[K::-1])
    return A