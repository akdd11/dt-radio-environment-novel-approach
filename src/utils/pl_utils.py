''' Contains some utilities to work with path loss maps'''

from itertools import product
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.distance import cdist
import seaborn as sns
from scanf import scanf


class PathLossMap:
    '''Class to save a path loss map.
    
    Contains the position of the transmitter and the path loss map.'''


    def __init__(self, tx_pos, pathloss):
        '''Initializes the PathLossMap object.

        tx_pos : tuple
            Position of the transmitter in meters.
        pathloss : numpy.ndarray
            Path loss map in dB.
        '''
        self.tx_pos = tx_pos
        self.pathloss = pathloss

    def show_pathloss_map(self, show_tx_pos=True):
        """Shows the path loss map.
        
        show_tx_pos : bool
            If True, the position of the transmitter is shown in the plot.
        """

        sns.heatmap(self.pathloss.T, square=True, cbar=True, cbar_kws={'label': 'Path loss [dB]'}) 
        if show_tx_pos:
            plt.plot(self.tx_pos[0], self.tx_pos[1], 'wv', markersize=10)
            plt.plot([], [], 'wv', markersize=10, label='Transmitter')

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()

class PathLossMapCollection:
    '''Class to save the results of the path loss map generation.

    Contains the configuration and an array of PathLossMap objects.'''
    def __init__(self, config):
        self.config = config
        self.pathlossmaps = []

    def pathlossmap_for_tx_pos(self, tx_pos):
        '''Returns the path loss map for a given transmitter position.
        
        tx_pos : tuple
            Position of the transmitter in meters.
        '''
        for plm in self.pathlossmaps:
            if np.array_equal(plm.tx_pos, tx_pos):
                return plm.pathloss

        raise ValueError('No path loss map found for the given transmitter position.')



def smooth_path_loss(path_loss, n=5):
    '''Sort out n biggest values and replace them with the n-th biggest value of the path loss map
    and returns the smoothed path loss map.

    In some earlier sionna version, extremely small values for the path loss in proximity to the 
    transmitter were generated. This function serves to cope with this problem.
    
    path_loss : numpy.ndarray
        Path loss map in dB.
    n : int
        Number of values to be sorted out.
    '''

    pl_threshold = np.sort(path_loss.flatten())[-n]
    path_loss[path_loss >= pl_threshold] = pl_threshold
    return path_loss


def create_fspl_sample(config, tx_pos=None, cov=None):
    """Returns one path loss map object created using free-space path loss.
    
    config : dict
        Configuration parameters.
    tx_pos : tuple
        2D Position of the transmitter in meters.
        If None, the transmitter position is randomly chosen.
    cov : np.ndarray
        Covariance matrix for correlated shadowing. If not specified, there is no random shadowing
    """

    if tx_pos == None:
        tx_pos = [np.random.randint(config['scene_size'][0]),
                np.random.randint(config['scene_size'][1])]
    path_loss = generate_fspl_map(config['scene_size'], 1, tx_pos, config['f_c'])

    if isinstance(cov, np.ndarray):
        noise = generate_corr_noise(config['scene_size'], cov)
        plm = PathLossMap(tx_pos, path_loss+noise)
    else:
        plm = PathLossMap(tx_pos, path_loss)

    return plm


def generate_fspl_map(scene_size, resolution, tx_pos, f_c, ple=2.0):
    """Generates a free space path loss map.
    
    scene_size : tuple
        Size of the scene in meters.
    resolution : float
        Resolution of the path loss map in meters.
    tx_pos : tuple
        Position of the transmitter in meters.
    f_c : float
        Carrier frequency in Hz.
    ple : float
        Path loss exponent.
    """

    # create a meshgrid of the scene
    x = np.arange(0, scene_size[0], resolution)
    y = np.arange(0, scene_size[1], resolution)
    x, y = np.meshgrid(x, y, indexing='ij')

    # calculate the distance between the transmitter and the receiver
    d = np.sqrt((x - tx_pos[0])**2 + (y - tx_pos[1])**2)

    # set a minimum distance > 0 to avoid log(0) when calculating the path loss
    d[d == 0] = 1

    # calculate the path loss
    path_loss = ple*10*np.log10(d) + 20*np.log10(f_c) - 147.55

    return path_loss


def create_cov_matrix_corr_shadow(scene_size, dcorr, std, resolution=1):
    """Create the covariance matrix for correlated shadowing.
    
    scene_size : tuple
        Size of the scene in meters.
    dcorr : float
        Correlation distance in meters.
    std : float
        std of the shadowing.
    resolution : float
        Resolution of the path loss map in meters.
    """
    
    # create a meshgrid of the scene
    x = np.arange(0, scene_size[0], resolution)
    y = np.arange(0, scene_size[1], resolution)
    points = [np.array(p) for p in product(x, y)]

    # built up the distance matrix between all points
    distance_list = cdist(points, points, 'euclidean')

    cov = std**2 * np.exp(-distance_list/dcorr)

    return cov


def generate_corr_noise(scene_size, cov):
    """Generate shadowing noise.
    
    scene_size : tuple
        Size of the scene in meters. 
    cov : np.ndarray
        Covariance matrix.
    """
    
    # generate a multivariate normal distribution
    noise = np.random.multivariate_normal(np.zeros(scene_size[0]*scene_size[1]), cov)
    noise = noise.reshape(scene_size[0], scene_size[1])

    return noise


def get_dataset_nr_for_noise_std_fspl(dataset_dir, std):
    """Returns the dataset number for a given noise variance.

    dataset_dir : str
        Directory where the datasets are saved.
    std : float
        Standard deviation of the shadowing for which the file is requested.
    """

    for description_fname in glob.glob(os.path.join(dataset_dir, 'fspl_PLdataset*.txt')):
        with open(description_fname, 'r') as description_file:
            lines = description_file.readlines()
            for l in lines:
                A = scanf("noise_std: %f", l)
                if A == None:
                    continue
                if A[0] == var:
                    return int(description_fname[len(dataset_dir)+len('fspl_PLdataset')+1:-4])
                
    raise ValueError('No dataset found for the given noise variance.')


if __name__ == "__main__":
    import os
    import sys

    # find the location of the repository and add it to the path
    repo_name = 'dt-radio-environment-novel-approach'
    module_path = __file__[:__file__.find(repo_name)+len(repo_name)]
    sys.path.append(os.path.abspath(module_path))

    dataset_dir = os.path.join(module_path, 'datasets')

    var = 5
    dataset_nr = get_dataset_nr_for_noise_std_fspl(dataset_dir, var=var)

    print(f'Var: {var} -- Dataset nr: {dataset_nr}')

