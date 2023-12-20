import hydra
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import seaborn as sns
import shutil
import sys
from tqdm import tqdm, trange

# find the location of the repository and add it to the path to import the modules
repo_name = 'dt-radio-environment-novel-approach'
module_path = __file__[:__file__.find(repo_name)+len(repo_name)]
sys.path.append(os.path.abspath(module_path))

from src.utils.pl_utils import PathLossMap, PathLossMapCollection  # required to load pathloss map results from pickle file
from src.utils.pl_utils import generate_fspl_map
from src.utils import description_file_utils, radiomap_utils

# Load config file which contains the parameterss into cfg object
hydra.initialize(version_base=None, config_path='conf')
cfg = hydra.compose(config_name='measurement_generation')

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'serif'

# Define configuration -----------------------------------------------------------------------------

measurement_method = cfg.measurement_method     # 'grid' or 'random'
if measurement_method == 'grid':
    grid_size = cfg.grid_size               # grid size of the measurements in meters


# can be used to add uncertainty to the measurements
add_noise_to_measurements = False
if add_noise_to_measurements:
    noise_var = 5       # variance of the shadowing noisey
    d_corr = 1          # correlation distance


pos_std = cfg.tx_pos_inaccuracy_std      # variance of the inaccuracy in tx position, see also README
pos_cov = [[pos_std**2, 0], [0, pos_std**2]]  # covariance matrix of the inaccuracy in tx position


# Load dataset -------------------------------------------------------------------------------------

rm_dataset_nr = cfg.rm_dataset_nr            # number of the path loss and radio map dataset to load (corresponds also to noise level)
meas_dataset_nr = cfg.meas_dataset_nr        # number of the measurements dataset to save
dataset_dir = os.path.join(module_path, 'datasets')

pl_filename = f'fspl_PLdataset{rm_dataset_nr}.pkl'  # pathloss dataset (load)
rm_filename = f'fspl_RMdataset{rm_dataset_nr}.pkl'  # radiomap dataset (load)
measurements_filenname = f'fspl_measurements{meas_dataset_nr}.pkl'  # measurements dataset (save)

# Check if the file already exists
if os.path.isfile(os.path.join(dataset_dir, measurements_filenname)):
    print(f'Dataset {measurements_filenname} already exists. Overwrite? (y/n)')
    answer = input()
    if answer == 'y':
        os.remove(os.path.join(dataset_dir, measurements_filenname))
    else:
        sys.exit()

print(f'Generating {measurements_filenname} ...')

with open(os.path.join(dataset_dir, pl_filename), 'rb') as fin:
    plmc = pkl.load(fin)

with open(os.path.join(dataset_dir, rm_filename), 'rb') as fin:
    radiomaps = pkl.load(fin)



# Find error between digital twin (without jammer) and the radio map (eventually containing jammer) measured

plot_difference = False
plot_dt_map = False

jammed_diffs = {'mean': [], 'median': []}
not_jammed_diffs = {'mean': [], 'median': []}

# Define the measurement points
meas_x, meas_y = radiomap_utils.generate_measurement_points(measurement_method, radiomaps[0].radio_map.shape,
                                                            grid_size=grid_size)


# Get resolution of radio maps
resolution = radiomaps[0].resolution


# A collection in which the measurements are stored
measurement_collection = radiomap_utils.MeasurementCollection(measurement_method, meas_x, meas_y, grid_size=grid_size)

for rm_orig in tqdm(radiomaps):

    rm_dt = radiomap_utils.RadioMap(rm_orig.radio_map.shape, resolution)
    for tx in rm_orig.transmitters:
        tx_pos_est = np.random.multivariate_normal(tx.tx_pos, pos_cov)   # estimated position of the transmitter
        # Recreate path loss map for estimated transmitter position
        # Shadowing (noise) is ignored now, because it can not be recreated
        pathlossmap = generate_fspl_map(plmc.config['scene_size'],
                                        plmc.config['resolution'],
                                        tx_pos_est,
                                        plmc.config['f_c'])

        rm_dt.add_transmitter('tx', tx_pos_est, tx.tx_power, pathlossmap)
    if plot_dt_map:
        # find minimum and maximum value for both radio maps
        v_min = np.min([np.min(rm_orig.radio_map), np.min(rm_dt.radio_map)])
        v_max = np.max([np.max(rm_orig.radio_map), np.max(rm_dt.radio_map)])
        rm_orig.show_radio_map(rm_type='orig', vmin=v_min, vmax=v_max)
        rm_dt.show_radio_map(rm_type='dt', vmin=v_min, vmax=v_max)

    if plot_difference:
        radiomap_utils.plot_radio_map_difference(rm_orig, rm_dt, plot_orig_tx=True, plot_dt_tx=True,
                                                meas_x=meas_x, meas_y=meas_y)
                

    measurements_orig = radiomap_utils.do_measurements(rm_orig, meas_x, meas_y)
    # Add shadowing noise to the measurements
    if add_noise_to_measurements:
        measurements_orig = radiomap_utils.add_correlated_noise_to_meas(meas_x, meas_y, measurements_orig, noise_var, d_corr)

    measurements_dt = radiomap_utils.do_measurements(rm_dt, meas_x, meas_y)
    measurement_collection.add_measurement(rm_orig.transmitters, rm_orig.jammers, measurements_orig, measurements_dt)

    if len(rm_orig.jammers) == 1:
        jammed_diffs['mean'].append(np.mean(measurements_orig - measurements_dt))
        
    else:
        not_jammed_diffs['mean'].append(np.mean(measurements_orig - measurements_dt))

with open(os.path.join(dataset_dir, measurements_filenname), 'wb') as f_out:
    pkl.dump(measurement_collection, file=f_out)

# save description file of measurements
if os.path.isfile(os.path.join(dataset_dir, measurements_filenname[:-4]+'.txt')):
    # if file already exists, shutil.copyfile raises exception -> delete first
    os.remove(os.path.join(dataset_dir, measurements_filenname[:-4]+'.txt'))
shutil.copyfile(os.path.join(dataset_dir, rm_filename[:-4]+'.txt'),
                 os.path.join(dataset_dir, measurements_filenname[:-4]+'.txt'))
with open(os.path.join(dataset_dir, measurements_filenname[:-4]+'.txt'), 'a') as f_out:
    print(f'\nmeasurement_method: {measurement_method}', file=f_out)
    if measurement_method == 'grid':
        print(f'grid_size: {grid_size}', file=f_out)
    else:
        raise NotImplementedError('Only grid measurement method is implemented.')
    print(f'tx_pos_inaccuracy_std: {pos_std}', file=f_out)


sns.kdeplot(jammed_diffs['mean'], label='Jammed')
sns.kdeplot(not_jammed_diffs['mean'], label='Not jammed')
plt.xlabel(r'$\overline{\Delta}$ $[\mathrm{dB}]$')
plt.grid()
plt.legend()
plt.tight_layout()
if cfg.save_density_plot:
    config = description_file_utils.get_config_from_file(os.path.join(dataset_dir, measurements_filenname[:-4]+'.txt'))
    plt.savefig(os.path.join(module_path, 'figures', 'fspl_anomaly_detection', f'difference_distr-noise_{config["noise_std"]}-grid_size{config["grid_size"]}.pdf'))
plt.show()