# title:        MULAN: A Blind and Off-Grid Method for Multichannel Echo Retrieval
# authors:      helena.peic.tukuljac@gmail.com, antoine.deleforge@inria.fr
# year:         2018
# license:      GPL v3
# description:  contains method for evaluating the performance of the algorithm together with storing
#               evaluation results as .pkl files, figures and latex scripts
#               experiments are executed in parallel with joblib library

import matplotlib
import matplotlib.pyplot as plt

import itertools
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
import os
import time
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count # for parallelizing the execution of the experiments

from algorithm import run
from algorithm import run_benchmark

FIGURE_FOLDER = "figures"
DATA_FOLDER = "data"
METADATA_FOLDER = "metadata"
LATEX_FOLDER = "latex"

def init_folder_structure():
    # make sure that all of the folders that will be used to keep the results exist
    if not os.path.exists(FIGURE_FOLDER):
        os.makedirs(FIGURE_FOLDER)
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    if not os.path.exists(METADATA_FOLDER):
        os.makedirs(METADATA_FOLDER)
    if not os.path.exists(LATEX_FOLDER):
        os.makedirs(LATEX_FOLDER)   

################################# benchmark measures #################################
# we will call this function twice - once for locations and once for weights
def get_reconstruction_error(m, ground_truth, reconstruction):
    reconstuction_error = 0
    for mic in range(m):
        reconstuction_error = reconstuction_error + \
        mean_squared_error(np.abs(ground_truth[:, mic]), np.abs(reconstruction[:, mic]))
    reconstuction_error = reconstuction_error/m
    return np.sqrt(reconstuction_error)   

def process_results(location_reconstruction_error, weight_reconstruction_error, F):
    LOCATION_SUCCESS_THRESHOLD = 1/F
    WEIGHT_SUCCESS_THRESHOLD = 1e-2
    experiment_success_counter = np.zeros(location_reconstruction_error.shape)
    for e in range(location_reconstruction_error.shape[0]):
        for method_index in range(location_reconstruction_error.shape[1]):
            if(location_reconstruction_error[e, method_index] <= LOCATION_SUCCESS_THRESHOLD):
            # we have decided to neglect the weight reconstruction for selecting the successful events
            #if(location_reconstruction_error[e, method_index] <= LOCATION_SUCCESS_THRESHOLD and \
            #   weight_reconstruction_error[e, method_index] <= WEIGHT_SUCCESS_THRESHOLD):
                experiment_success_counter[e, method_index] = 1
    experiment_mean_location = np.zeros((location_reconstruction_error.shape[1], 1))
    experiment_deviation_location = np.zeros((location_reconstruction_error.shape[1], 1))
    experiment_mean_weight = np.zeros((weight_reconstruction_error.shape[1], 1))
    experiment_deviation_weight = np.zeros((weight_reconstruction_error.shape[1], 1))
    experiment_success_percentage = np.zeros((weight_reconstruction_error.shape[1], 1))
    for method_index in range(location_reconstruction_error.shape[1]):
        if(np.count_nonzero(experiment_success_counter[:, method_index]) != 0):
            experiment_mean_location[method_index] = np.mean(location_reconstruction_error[experiment_success_counter[:, method_index] != 0, method_index])
            experiment_deviation_location[method_index] = np.std(location_reconstruction_error[experiment_success_counter[:, method_index] != 0, method_index])
            experiment_mean_weight[method_index] = np.mean(weight_reconstruction_error[experiment_success_counter[:, method_index] != 0, method_index])
            experiment_deviation_weight[method_index] = np.std(weight_reconstruction_error[experiment_success_counter[:, method_index] != 0, method_index])
            experiment_success_percentage[method_index] = np.sum(experiment_success_counter[:, method_index])/location_reconstruction_error.shape[0]*100 # in %
    return experiment_mean_location, experiment_deviation_location, experiment_mean_weight, experiment_deviation_weight, experiment_success_percentage   

########################### storing/retrieving results ##############################
# save figure to .png file and data to .pkl file
def save_figure_and_data(data, fig_title, x_title, y_title, x_values, y_values, E):
    # flipping in order to have the smallest experiment parameters in low left corner
    data = np.reshape(data, (len(y_values), len(x_values)))
    data = np.flipud(data)
    #data = data/E
    plt.figure()
    plt.imshow(data, cmap='gray', interpolation='none')
    plt.title(fig_title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    
    # adjusting the axis labels
    ax = plt.gca()
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_xticklabels(x_values)
    ax.set_yticklabels(list(reversed(y_values)))
    
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            plt.text(i, j, str(int(data[j, i]/E*100)) + "%", color='green', ha='center', va='center')  
    plt.colorbar()
    plt.clim(0, E)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(FIGURE_FOLDER, timestr + fig_title + '.png'))    
    with open(os.path.join(DATA_FOLDER, timestr + fig_title  + ".pkl"), 'wb') as f:
        pickle.dump([data, y_values, x_values], f)  
        
    return

def save_recovery_details(M, K, filter_option, input_signal_option, tau, tau_r, alpha, alpha_r):
    file_name = os.path.join(DATA_FOLDER, filter_option.name + "_" + input_signal_option.name + "_" + "M=" + str(M) + "_" + "K=" + str(K) + ".pkl")
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump([tau, alpha, tau_r, alpha_r], f)
        

def save_recovery_details_3_methods(nF, M, K, filter_option, input_signal_option, location_reconstruction_error, weight_reconstruction_error):
    file_name = os.path.join(DATA_FOLDER, "nF=" + str(nF) + "_" + filter_option.name + "_" + input_signal_option.name + "_" + "3_methods" + ".pkl")
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump([location_reconstruction_error, weight_reconstruction_error], f)
        
def save_results_to_latex(experiment_mean_location, experiment_deviation_location, experiment_mean_weight, experiment_deviation_weight, \
                          experiment_success_percentage, file_name):
    file_name = os.path.join(LATEX_FOLDER, file_name)
    import pandas as pd
    from matplotlib import rc
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)
    results_data = []
    for i in range(3):
        results_loc = ('%.2e' % experiment_mean_location[i]) + r'$\pm$' + ('%.2e' % experiment_deviation_location[i])
        results_wei = ('%.2e' % experiment_mean_weight[i]) + r'$\pm$' + ('%.2e' % experiment_deviation_weight[i])
        results_per = '%d' % experiment_success_percentage[i]
        results_data.append([results_loc, results_wei, results_per])
    df_tau = pd.DataFrame({r"$\alpha$ and $\tau$ reconstuction": ['cross-relation', 'constrainted LASSO', 'MULAN']}, index = [0, 1, 2])
    df_tau = pd.concat([df_tau, pd.DataFrame(results_data, columns=['locations ($\mu \pm \sigma$)', 'weights ($\mu \pm \sigma$)', 'success ($\%$)'])],
               axis=1)
    df_tau.style
    with open(file_name, 'w') as tf:
        tf.write(df_tau.to_latex(escape  = False))
        
def load_data(file_name):
    with open(file_name, 'rb') as f:
        data, param_y, param_x = pickle.load(f)
    return data, param_y, param_x

########################### experiment execution and logging ##############################
def print_experiment_parameters(init_option, filter_option, input_signal_option, M, K, T, SNR, F, nF):
    print("Exp: Init opt = %s\t Filter opt = %s\t Input signal opt = %s\t \n M = %d\t K = %d\t T = %.3f\t SNR = %d\t F = %d \tnF = %d" \
          % (init_option.name, filter_option.name, input_signal_option.name, M, K, T, SNR, F, nF))
    return

'''
We will evaluate the performance of the following:
a) TEMPORAL domain algorithms (input: measurements in time domain and filter length)
Cross-relation method
LASSO (Least Absolute Shrinkage and Selection Operator) method
b) FREQUENCY domain algorithm (input: measurements in frequency domain and filter sparsity)
MULAN (MULtichannel ANnihilation)
Output: the mean and the standard deviation of the recovery results for locations and weights
'''
def run_experiments_benchmark(E, M, K, T, F, nF, SNR, init_option, filter_option, input_signal_option):
    init_folder_structure()
    NUMBER_OF_CASES = 3 # because we have: CR, LASSO and MULAN
    location_reconstruction_error = np.zeros((E, NUMBER_OF_CASES))
    weight_reconstruction_error = np.zeros((E, NUMBER_OF_CASES))
    results = Parallel(n_jobs=cpu_count()-1) \
        (delayed(run_benchmark)(M, K, T, F, nF, SNR, init_option, filter_option, input_signal_option, e + 1) \
         for e in range(E))

    # we need to reorganize the results into Ex3 shape
    for e in range(E):
        results_of_current_experiment = np.squeeze(np.asarray(results[e]))
        location_reconstruction_error[e, :] = results_of_current_experiment[0]
        weight_reconstruction_error[e, :] = results_of_current_experiment[1]
    save_recovery_details_3_methods(nF, M, K, filter_option, input_signal_option, location_reconstruction_error, weight_reconstruction_error)
    experiment_mean_location, experiment_deviation_location, experiment_mean_weight, experiment_deviation_weight, experiment_success_percentage = \
    process_results(location_reconstruction_error, weight_reconstruction_error, F)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(DATA_FOLDER, filter_option.name + "_" + input_signal_option.name + "_" + timestr + ".pkl"), 'wb') as f:
        pickle.dump([experiment_mean_location, experiment_deviation_location, experiment_mean_weight, experiment_deviation_weight, experiment_success_percentage], f)
    return experiment_mean_location, experiment_deviation_location, experiment_mean_weight, experiment_deviation_weight, experiment_success_percentage
        
'''
For phase transition plots.
Experiments are run in parallel with 'joblib' library.
To track the progress of the iterations we have used 'tqmd' library.
'''
def run_experiments(E, M, K, T, F, nF, SNR, init_option, filter_option, input_signal_option, param1, param2, fig_title, x_label, y_label):
    init_folder_structure()
    data_loc = np.array([])  # location reconstuction buffer
    data_w = np.array([])    # weight reconstruction buffer
    LOCATION_SUCCESS_THRESHOLD = 0.5/F
    WEIGHT_SUCCESS_THRESHOLD = 1e-2
    
    for m, k in tqdm(itertools.product(M, K)):
        print_experiment_parameters(init_option, filter_option, input_signal_option, m, k, T, SNR, F, nF)
        results = Parallel(n_jobs=cpu_count()-1) \
        (delayed(run)(T, k, m, F, nF, SNR, init_option, filter_option, input_signal_option, e + 1) \
         for e in range(E))
        
        # analysis of the experimental results:
        result_matrix = np.matrix(results)
        # relative locations (up to a shift) and amplitudes (up to a scaling)
        ground_truth_relative_delays = np.squeeze(np.asarray(result_matrix[:, 0]))
        reconstructed_relative_delays = np.squeeze(np.asarray(result_matrix[:, 1]))
        ground_truth_relative_amplitudes = np.squeeze(np.asarray(result_matrix[:, 2]))
        reconstructed_relative_amplitudes = np.squeeze(np.asarray(result_matrix[:, 3]))
        # absolute locations and amplitudes (saved for case if we want to change the metric)        
        ground_truth_delays = np.squeeze(np.asarray(result_matrix[:, 4]))
        reconstructed_delays = np.squeeze(np.asarray(result_matrix[:, 5]))
        ground_truth_amplitudes = np.squeeze(np.asarray(result_matrix[:, 6]))
        reconstructed_amplitudes = np.squeeze(np.asarray(result_matrix[:, 7]))
        save_recovery_details(m, k, filter_option, input_signal_option, ground_truth_delays, reconstructed_delays, ground_truth_amplitudes, reconstructed_amplitudes)
        
        location_reconstruction_counter = 0
        weight_reconstruction_counter = 0
        
        for e in range(E):
            location_reconstruction_result = get_reconstruction_error(m, ground_truth_relative_delays[e], \
                                                                      reconstructed_relative_delays[e])
            weight_reconstruction_result = get_reconstruction_error(m, ground_truth_relative_amplitudes[e], \
                                                                      reconstructed_relative_amplitudes[e])
            location_reconstruction_counter = location_reconstruction_counter + \
            (location_reconstruction_result < LOCATION_SUCCESS_THRESHOLD)
            weight_reconstruction_counter = weight_reconstruction_counter + \
            (weight_reconstruction_result < WEIGHT_SUCCESS_THRESHOLD)
            
        data_loc = np.append(data_loc, location_reconstruction_counter)
        data_w = np.append(data_w, weight_reconstruction_counter)
        print('Number of successful location reconstruction: ', location_reconstruction_counter, ' for success threshold: ', LOCATION_SUCCESS_THRESHOLD)
        print('Number of successful weight reconstruction: ', weight_reconstruction_counter, ' for success threshold: ', WEIGHT_SUCCESS_THRESHOLD)
    save_figure_and_data(data_loc, fig_title + " LOC rec " + str(E) + "exp" + " Fs = " + str(F) + " nF = " + str(nF) + \
                         " T = " + str(T), x_label, y_label, param2, param1, E)
    save_figure_and_data(data_w, fig_title + " WEI rec " + str(E) + "exp" + " Fs = " + str(F) + " nF = " + str(nF) + \
                     " T = " + str(T), x_label, y_label, param2, param1, E)