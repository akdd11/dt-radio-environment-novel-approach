"""Here, the anomaly detection is performed for the free space path loss (FSPL) model.

Configuration (e.g., model selection) needs to be done in the conf file.s"""

import hydra
import numpy as np
import os
import sys
from tqdm import tqdm
import warnings

# Own modules
repo_name = 'dt-radio-environment-novel-approach'
module_path = __file__[:__file__.find(repo_name)+len(repo_name)]
sys.path.append(os.path.abspath(module_path))

from src.anomaly_detection import anomaly_detection_utils

# Load config file which contains the parameterss into cfg object
hydra.initialize(version_base=None, config_path='conf')
cfg = hydra.compose(config_name='fspl_anomaly_detection')


print(f'Method: {cfg.method}')

if cfg.method not in ['one_class_svm', 'lof', 'unsupervised_threshold']:
    warnings.warn('Method not considered for the paper and may not be fully mature.')

# Further parameters

vary_parameter = cfg.vary_parameter # parameter that is changed in the simulation

if vary_parameter  == 'noise_std':
    grid_size = 10
    noise_std_list = np.arange(0, 11, 2)

elif vary_parameter == 'grid_size':
    noise_std = 2
    grid_size_list = np.arange(5, 25, 5)

else:
    raise NotImplementedError(f'Not implemented vary_parameter: {vary_parameter}')


# Calling function for anomaly detection in one configuration

if vary_parameter == 'noise_std':
    for noise_std in tqdm(noise_std_list):
        if cfg.verbosity > 0:
            print(f"\nFor noise std: {noise_std}")
        anomaly_detection_utils.fspl_anomaly_detection(cfg.method, cfg.outlier_probability,
                                                        noise_std, grid_size, cfg, cfg.probability,
                                                        cfg.verbosity)

elif vary_parameter == 'grid_size':
    for grid_size in tqdm(grid_size_list):
        if cfg.verbosity > 0:
            print(f"\nFor grid size: {grid_size}")
        anomaly_detection_utils.fspl_anomaly_detection(cfg.method, cfg.outlier_probability,
                                                       noise_std, grid_size, cfg, cfg.probability,
                                                       cfg.verbosity)
