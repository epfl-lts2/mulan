# title:        MULAN: A Blind and Off-Grid Method for Multichannel Echo Retrieval
# authors:      helena.peic.tukuljac@gmail.com, antoine.deleforge@inria.fr
# year:         2018
# license:      GPL v3
# description:  contains core algorithm implementation, its helper functions and benchmark with baseline methods

import numpy as np
import numpy.linalg as la
from scipy.signal import convolve
import scipy.linalg as lam
from sklearn.metrics import mean_squared_error

from baseline_methods import CR
from baseline_methods import LASSO
import fri
import measurement_tools
import noise_tools

import logging
logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)4s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

### initialization types
from enum import Enum
class InitializationType(Enum):
    GroundTruthPerturbed, Beamformed, Inverse, InverseMean, Random = range(5)
    
def get_initial_value(init_option, m, f, ground_truth, tau_ground_truth):
    mu, sigma = 0, 1 # mean and standard deviation
    noise = np.random.normal(mu, sigma, len(f))
    noise = noise/la.norm(noise)
    
    if(init_option == InitializationType.GroundTruthPerturbed):
        perturbation_value = 0.2
        z = np.power(ground_truth, -1)
        z = z + noise*la.norm(z)*perturbation_value
    elif(init_option == InitializationType.Beamformed):
        z = np.zeros(m[:, 0].shape)
        for i in range(m.shape[1]):
            z = z + np.multiply(np.exp(1j*2*np.pi*f*(-1)*tau_ground_truth[0, i]), m[:, i])
        z = z/m.shape[1]
    elif(init_option == InitializationType.Inverse):
        z = np.power(m[:, 0], -1)
    elif(init_option == InitializationType.InverseMean):
        z = np.power(np.mean(np.sum(m, axis=1)), -1)
    else: # Random
        z = np.power(np.mean(np.sum(m, axis=1)), -1)
        z = noise*la.norm(z)
                             
    return z

def get_relative_delays_and_amplitudes(tau, tau_r, alpha, alpha_r, K, m, T):
    relative_delays = np.zeros(tau.shape)
    relative_delays_r = np.zeros(tau.shape)
    relative_amplitudes = np.zeros(tau.shape)
    relative_amplitudes_r = np.zeros(tau.shape)
    for i in range(tau.shape[1]):
        for k in range(K):
            if (tau_r[k, i] >= T/2):
                tau_r[k, i] = tau_r[k, i] - T
    tau = tau - np.amin(tau[:, 0])# % T
    tau_r = tau_r - np.amin(tau_r[:, 0]) #% T
    for mi in range(tau.shape[1]):
        relative_delays[:, mi] = np.sort(tau[:, mi])
        relative_delays_r[:, mi] = np.sort(tau_r[:, mi])
        indices = np.argsort(tau[:, mi])
        indices_r = np.argsort(tau_r[:, mi])
        relative_amplitudes[:, mi] = alpha[indices, mi]/np.max(abs(alpha))
        relative_amplitudes_r[:, mi] = alpha_r[indices_r, mi]/np.max(abs(alpha_r))
    return relative_delays, relative_delays_r, relative_amplitudes, relative_amplitudes_r

'''
Main algorithm parameters:
m - M microphone measurements in frequency domain (come as a result of chosed filter and input signal)
f - frequency set
K - level of sparsity
alpha - ground truth weights of Diracs
tau - ground truth delays of Diracs
init_option - current one is Random (other are available, but not reliable)
ground_truth - ground truth signal in frequency domain - used for some initialization schemes
'''
def mulan(m, f, frequency_offset, frequency_step, K, T, alpha, tau, init_option, filter_option, input_signal_option, ground_truth):
    logger.info('M = %d K = %d nF = %d - min: %.2f; step: %.2f [Hz]; max: %.2f' % (m.shape[1], K, len(f), min(f), f[1] - f[0], max(f)))
    PROGRESS_THRESHOLD = 1e-3
    ITERATION_THRESHOLD = 1000
    
    MAX_RANDOM_INITIALIZATIONS = 1
    if(init_option.value == InitializationType.Random.value):
        MAX_RANDOM_INITIALIZATIONS = 20
    
    global_obj_fun = 1000
    obj_fun_value = 1000
    global_a = np.zeros((K + 1, m.shape[1]), dtype=np.complex128)
    initialization_counter = 0   

    # start the algorithm
    while(initialization_counter < MAX_RANDOM_INITIALIZATIONS):
        initialization_counter = initialization_counter + 1
        old_obj_fun_value = 10000
        iter = 0 # we need to control the number of iterations
        
        z = get_initial_value(init_option, m, f, ground_truth, tau)
        # iterate while making progress, but not more than ITERATION_THRESHOLD iterations
        while((abs((old_obj_fun_value - obj_fun_value)/old_obj_fun_value) > PROGRESS_THRESHOLD) \
            & (iter < ITERATION_THRESHOLD)):
            iter = iter + 1
            # STEP 1: given the estimation of the input signal, estimate the filters
            a = estimate_filters_from_signal(m, z, K, T)
            # STEP 2: given the estimation of the filters, estimate the input signal
            z, z_toep = estimate_signal_from_filters(m, a, K)
            # STEP 3: keep track of the global optimum over the random initializations
            old_obj_fun_value = obj_fun_value
            obj_fun_value = 0
            for k in range(m.shape[1]):
                obj_fun_value = obj_fun_value + np.power(la.norm(z_toep[:, :, k] @ z), 2)
            obj_fun_value = obj_fun_value/(m.shape[0]*m.shape[1]) # normalize by the number of frequencies and number of microphones
            
        logger.info('Initialization counter: %d out of %d; objective function: %e' % (initialization_counter, MAX_RANDOM_INITIALIZATIONS, obj_fun_value))
        if(global_obj_fun > obj_fun_value):
            global_obj_fun = obj_fun_value
            global_a = a
        if(global_obj_fun < np.power(10, float(-13))):
            break

    # STEP 4: unpack the data in order to evaluate the performance against the ground truth
    logger.info('Lowest objective function value: %e', global_obj_fun)
    # a) we need to bring the roots back to the unit circle
    global_a = polyscale(global_a)
    global_z, z_toep = estimate_signal_from_filters(m, global_a, K)

    # b) recovery of the input signal and the (location, weight) pairs for filters from GLOBAL_A and GLOBAL_Z    
    tau_r = np.zeros((K, m.shape[1]))
    alpha_r = np.zeros((K, m.shape[1]))  
    for k in range(m.shape[1]):
        # global_a for recovery of the positions
        filter_zeros = np.roots(np.flipud(global_a[:, k]))
        tau_r[:,k] = fri.get_locations(filter_zeros, T, frequency_step)
        # global_z for recovery of the weights       
        h = np.multiply(m[:, k], global_z)
        alpha_r[:,k] = fri.get_weights(h, filter_zeros, K, pos_freq = True)
    
    relative_delays, relative_delays_r, relative_amplitudes, relative_amplitudes_r = \
        get_relative_delays_and_amplitudes(tau, tau_r, alpha, alpha_r, K, m, T)
       
    sf_r = np.power(global_z, -1)   
    return relative_delays, relative_delays_r, relative_amplitudes, relative_amplitudes_r, \
        tau, tau_r, alpha, alpha_r, ground_truth, sf_r
        
def estimate_filters_from_signal(measurements, signal, K, T):
    # we will have one annihilating filter per microphone
    filters = np.zeros((K + 1, measurements.shape[1]), dtype=np.complex128)
    for i in range(measurements.shape[1]):
        X = np.multiply(measurements[:, i], signal)
        filters_toep = lam.toeplitz(X[K:], X[K::-1])
        filters[:, i] = fri.get_annihilating_filter(filters_toep)#, n_Cadzow = 20)
        if(np.all(filters[:, i]) == 0):
            break                
        filters[:, i] = np.flipud(filters[:, i])
    return filters
    
def estimate_signal_from_filters(measurements, filters, K):
    nF = measurements.shape[0]
    filters_toep = np.zeros((nF - K, nF, measurements.shape[1]), dtype=np.complex128)
    for i in range(measurements.shape[1]):
        filters_toep[:, :, i] = lam.toeplitz(np.hstack((filters[0, i], np.zeros(nF - len(filters[:, i])))), \
                                             np.hstack((filters[:, i], np.zeros(nF - len(filters[:, i])))))
        
    signal_toep = np.zeros((nF - K, nF, measurements.shape[1]), dtype=np.complex128)
    for k in range(measurements.shape[1]):
        signal_toep[:, :, k] = filters_toep[:, :, k] @ np.diag(measurements[:, k])
    
    # we will need to find a signal that annihilates all the filters
    signal_toep_con = signal_toep[:, :, 0]              
    for k in range(1, measurements.shape[1]):
        signal_toep_con = np.concatenate((signal_toep_con, signal_toep[:, :, k]), axis=0)
    signal = fri.get_annihilating_filter(signal_toep_con)#, n_Cadzow = 20)
    
    return signal, signal_toep 

def polyscale(a):
    # we need to ensure that the roots of our filter have unit norm before computing z       
    # this ensures that the ratio of the input and output spectrum is a constant function
    for i in range(a.shape[1]):
        a[:, i] = np.flipud(a[:, i])
        roots = np.roots(a[:, i])
        a[:, i] = np.poly(np.divide(roots, np.abs(roots)))
        a[:, i] = np.flipud(a[:, i])
    return a

def run(Ts, K, mic_num, F, nF, SNR, init_option, filter_option, input_signal_option, file_number):
    m, f, frequency_offset, frequency_step, alpha, tau, sf, T = \
    measurement_tools.get_measurements(filter_option, input_signal_option, K, Ts, mic_num, F, nF, file_number) 
    
    # add noise with the given SNR
    for k in range(mic_num):
        m[:, k] = noise_tools.add_white_noise(m[:, k], SNR)
        
    # try to recover the filters and the spectrum of the input signal jointly
    return mulan(m, f, frequency_offset, frequency_step, K, T, alpha, tau, init_option, filter_option, input_signal_option, sf)

def get_reconstruction_error(m, ground_truth, reconstruction):
    reconstuction_error = 0
    for mic in range(m):
        reconstuction_error = reconstuction_error + \
        mean_squared_error(np.abs(ground_truth[:, mic]), np.abs(reconstruction[:, mic]))
    reconstuction_error = reconstuction_error/m
    return np.sqrt(reconstuction_error)   

def run_benchmark(M, K, T, F, nF, SNR, init_option, filter_option, input_signal_option, file_number):
    # Step 1: get the data in the temporal domain
    if (filter_option == measurement_tools.FilterType.Artificial): # ON GRID
        filters, alpha, tau = measurement_tools.generate_artificial_rirs(M, K, T, F) 
    else:                                        # OFF GRID
        filters, alpha, tau = measurement_tools.load_simulated_rirs(M, K, F, file_number)            
    if (input_signal_option == measurement_tools.InputSignalType.Artificial): # WHITE NOISE
        signal = measurement_tools.get_artificial_input_signal(T, F)
    else:                                                   # SPEECH
        signal = measurement_tools.load_input_signal(T, F, file_number)            
    L = filters.shape[0]
    Ts = len(signal)/F
    Tf = filters.shape[0]/F
    time_vector_convolution = np.linspace(1/F, Ts + Tf, len(signal) + filters.shape[0] - 1)
    x = np.zeros((len(time_vector_convolution), M), dtype=np.complex128) 
    for k in range(M):
        x[:, k] = convolve(signal, filters[:, k], 'full')    

    # Step 2: get the data in the frequency domain
    X, sf, f, frequency_offset, frequency_step, T_total = measurement_tools.get_spectral_coefficients(signal, K, filters, nF, F)

    # Step 3: evaluate the performance of different algorithms for the given data
    # x - measurements in time; X - measurements in frequency
    # L - filter length;        K - sparsity level
    tau_cr, alpha_cr, h_cr = CR(x, L, K, F)
    tau_lasso, alpha_lasso, h_lasso = LASSO(x, L, K, F)
    tau, tau_mulan, alpha, alpha_mulan, tau_gt, tau_r, alpha_gt, alpha_r, gt, sf_r = mulan(X, f, frequency_offset, frequency_step, K, T_total, alpha, \
                                                         tau, init_option, filter_option, input_signal_option, np.zeros((len(f), 1)))
    
    NUMBER_OF_CASES = 3 # because we have: CR, LASSO and MULAN
    location_reconstruction_error = np.zeros((NUMBER_OF_CASES, 1))
    weight_reconstruction_error = np.zeros((NUMBER_OF_CASES, 1))
    location_reconstruction_error[0] = get_reconstruction_error(M, tau, tau_cr)
    location_reconstruction_error[1] = get_reconstruction_error(M, tau, tau_lasso)
    location_reconstruction_error[2] = get_reconstruction_error(M, tau, tau_mulan)
    weight_reconstruction_error[0]  = get_reconstruction_error(M, alpha, alpha_cr)
    weight_reconstruction_error[1] = get_reconstruction_error(M, alpha, alpha_lasso)
    weight_reconstruction_error[2] = get_reconstruction_error(M, alpha, alpha_mulan)
    return location_reconstruction_error, weight_reconstruction_error