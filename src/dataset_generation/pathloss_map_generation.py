'''For a given scene, this script generates a dataset of path loss maps for a given number of random transmitter positions.'''

import hydra
import os
import pickle as pkl
import sys
from tqdm import trange

# find the location of the repository and add it to the path to import the modules
repo_name = 'dt-radio-environment-novel-approach'
module_path = __file__[:__file__.find(repo_name)+len(repo_name)]
sys.path.append(os.path.abspath(module_path))

from src.utils.pl_utils import PathLossMap, PathLossMapCollection, create_fspl_sample, create_cov_matrix_corr_shadow

# Load config file which contains the parameterss into cfg object
hydra.initialize(version_base=None, config_path='conf')
cfg = hydra.compose(config_name='pathloss_map_generation')

# User defined parameters -------------------------------------------------------------------------

f_c = cfg.f_c       # carrier frequency in Hz

config = {'scene_size': [41, 41],               # in m
            'resolution': 1,                      # in m
            'f_c': f_c,                           # in Hz
            'noise_std': cfg.fspl.noise_std,      # std of the shadowing noise in dB
            'd_corr': 1,                          # correlation distance in meters
}
    

# Define paths ------------------------------------------------------------------------------------

dataset_path = os.path.join(os.path.dirname(__file__),'..','..','datasets',f'fspl_PLdataset{cfg.dataset_nr}.pkl') 

# Check if the file exists already
if os.path.exists(dataset_path):
    print(f'Dataset {dataset_path} already exists. Overwrite? (y/n)')
    answer = input()
    if answer == 'y':
        os.remove(dataset_path)
    else:
        sys.exit()

# Initialization ----------------------------------------------------------------------------------

plmc = PathLossMapCollection(config)

# Do the actual simulation loop ------------------------------------------------------------------

# Define covariance matrix for correlated shadowing
cov = create_cov_matrix_corr_shadow(config['scene_size'],
                                    config['d_corr'],
                                    config['noise_std'])

for idx in trange(cfg.nr_samples):
    plm = create_fspl_sample(config, cov=cov)
    plmc.pathlossmaps.append(plm)


# Save the results -------------------------------------------------------------------------------

with open(dataset_path, 'wb') as fout:
    pkl.dump(plmc, fout)

# Adding the description of the dataset to a text file    
with open(dataset_path[:-4]+'.txt', 'w') as f_descr:
    for k in config.keys():
        print(f'{k}: {config[k]}', file=f_descr)