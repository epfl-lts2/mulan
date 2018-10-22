# title:        MULAN: A Blind and Off-Grid Method for Multichannel Echo Retrieval
# authors:      helena.peic.tukuljac@gmail.com, antoine.deleforge@inria.fr
# year:         2018
# license:      GPL v3
# description:  contains definition of two basic types of experiments: (1) comparison with baseline
#               (2) phase transition diagrams

import numpy as np

from algorithm import InitializationType
from measurement_tools import InputSignalType
from measurement_tools import FilterType
import test_tools

'''
Parameters:
init_option         - random initalization (fixed to 20 for the moment)
filter_option       - artificial (on grid) or simulations (off grid)
input_signal_option - artificial (white noise) or speech
E                   - number of experiments
nF                  - number of frequencies
F                   - sampling frequency
T                   - lenght of the input signal

filter_option determines what kind of evaluation is performed:
1. if the filter is artificial, evaluation is ON grid
2. if the filter is simulated, evaluation is OFF grid
'''
def benchmark(init_option, filter_option, input_signal_option, E, nF, F, T, M, K):
    SNR = np.array([1000])
    return test_tools.run_experiments_benchmark(E, M, K, T, F, nF, SNR, init_option, filter_option, input_signal_option)
    
def phase_transition(init_option, filter_option, input_signal_option, E, nF, F, T):    
    M = np.array([2, 3, 4, 5, 6, 7])    # microphone number
    K = np.array([2, 3, 4, 5, 6, 7])    # Dirac number 
    SNR = np.array([1000])              # signal to noise ratio
    fig_title = " KvsM " + init_option.name + "," + filter_option.name  + "," + input_signal_option.name
    x_label = "K"
    y_label = "M"
    paramx = K
    paramy = M
    
    test_tools.run_experiments(E, M, K, T, F, nF, SNR, init_option, filter_option, input_signal_option, \
                               paramy, paramx, fig_title, x_label, y_label)