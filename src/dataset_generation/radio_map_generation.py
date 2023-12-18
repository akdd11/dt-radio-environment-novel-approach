# import matplotlib.pyplot as plt
import hydra
import numpy as np
import os
import pickle as pkl
import sys
from tqdm import trange


repo_name = 'dt-radio-environment-novel-approach'
module_path = __file__[:__file__.find(repo_name)+len(repo_name)]
sys.path.append(os.path.abspath(module_path))

from src.utils.pl_utils import PathLossMap, PathLossMapCollection  # required to load pathloss map results from pickle file
from src.utils.radiomap_utils import RadioMap, Transmitter

# Load config file which contains the parameterss into cfg object
hydra.initialize(version_base=None, config_path='conf')
cfg = hydra.compose(config_name='radio_map_generation')


# Load the pathloss maps --------------------------------------------------------------------------- 

dataset_nr = cfg.dataset_nr
dataset_dir = os.path.join(module_path, 'datasets')

pl_filename = f'fspl_PLdataset{dataset_nr}.pkl'  # pathloss dataset (load)
rm_filename = f'fspl_RMdataset{dataset_nr}.pkl'  # radiomap dataset (save)

with open(os.path.join(dataset_dir, pl_filename), 'rb') as fin:
    pl_results = pkl.load(fin)

# Configuration ----------------------------------------------------------------------------------

tx_power = 20               # transmit power in dBm
range_num_tx = [10, 10]       # minimum and maximum number of transmitters
dist_num_tx = 'uniform'     # distribution of the number of transmitters

range_jam_power = [20, 20]   # minimum and maximum jamming power in dBm
dist_jam_power = 'uniform'  # distribution of the jamming power
range_num_jam = [0, 1]      # minimum and maximum number of jammers
dist_num_jam = 'uniform'    # distribution of the number of jammers

# Generate the radiomaps --------------------------------------------------------------------------

print(f'Path loss dataset filename: {pl_filename}')

radiomaps = []
for i in trange(cfg.num_radiomaps):
    radio_map = RadioMap(pl_results.config['scene_size'][:2], pl_results.config['resolution'])
    pl_map_idxs = list(range(len(pl_results.pathlossmaps))) # there can only be one transmitter or jammer at a certain position

    # Choose the number of transmitters and generate the corresponding radio maps
    num_tx = np.random.randint(range_num_tx[0], range_num_tx[1]+1)
    for i_tx in range(num_tx):
        # Choose a random path loss map
        idx = np.random.choice(pl_map_idxs)
        pl_map_idxs.remove(idx)

        radio_map.add_transmitter('tx', pl_results.pathlossmaps[idx].tx_pos, tx_power, pl_results.pathlossmaps[idx].pathloss)

    # Do the same for the jammers
    jam_power = np.random.uniform(range_jam_power[0], range_jam_power[1])
    num_jam = np.random.randint(range_num_jam[0], range_num_jam[1]+1)
    for i_jam in range(num_jam):
        # Choose a random path loss map
        idx = np.random.choice(pl_map_idxs)
        pl_map_idxs.remove(idx)

        radio_map.add_transmitter('jammer', pl_results.pathlossmaps[idx].tx_pos, jam_power, pl_results.pathlossmaps[idx].pathloss)
   
    # convert radio_map to dBm
    radiomaps.append(radio_map)

# Save the radio maps to a pickle file
with open(os.path.join(dataset_dir, rm_filename), 'wb') as fout:
    pkl.dump(radiomaps, fout)

# Adding the description of the dataset to a text file
config = pl_results.config
with open(os.path.join(dataset_dir,rm_filename[:-4]+'.txt'), 'w') as f_descr:
    for k in config.keys():
        print(f'{k}: {config[k]}', file=f_descr)