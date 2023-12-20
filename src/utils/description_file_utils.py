""" Different utitlities for handling the description files (.txt) of the dataset files."""

from glob import glob
import os
import pandas as pd
from scanf import scanf
import sys

repo_name = 'dt-radio-environment-novel-approach'
module_path = __file__[:__file__.find(repo_name)+len(repo_name)]
sys.path.append(os.path.abspath(module_path))


def get_config_from_file(path_to_file):
    """Reads the config from a file and returns it as a dictionary.
    
    path_to_file: str
        Path to the text file containing the config.
    """
    config = {}
    with open(path_to_file, 'r') as f_in:
        lines = f_in.readlines()

    def read_line_with_array(l):
        splitted_line = l.split(':') # split into key and array
        splitted_line[1] = splitted_line[1].translate(str.maketrans('', '', '[]\n ')) # only comma remains to separate the values
        config[splitted_line[0]] = [float(x) for x in splitted_line[1].split(',')] # comma-separated values to array


    for l in lines:

        if 'scene_size' in l:
            read_line_with_array(l)
            continue

        A = scanf("%s: %s", l)
        if A == None:
            continue

        # if possible, convert value to float
        try:
            config[A[0]] = float(A[1])
        except ValueError:
            config[A[0]] = A[1]

    return config


def get_filename_by_params(**kwargs):
    """Get the filename for the specified parameters.
    
    The filename_table can either be passed or pl_model and data_type have to be provided.
    
    filename_table : pd.DataFrame
        Table which contains the relation between parameters and filename for all files of the specified type.
    pl_model : str
        'fspl' or 'sionna'. Required if filename_table is not provided.
    data_type : str
        'path_loss' or 'radiomap' or 'measurements'. Required if filename_table is not provided.
    **kwargs : dict
        Parameters for which the filename should be returned.
    """

    # load the filename table
    # after the filename_table is loaded, the arguments required for this are removed from kwargs
    # to only keep the parameters for which the filename should be returned
    if 'filenames_table' not in kwargs:
        if 'pl_model' not in kwargs or 'data_type' not in kwargs:
            raise ValueError('Either filenames_table or pl_model and data_type have to be provided.')
        filenames_table = get_param_by_filename_table(kwargs['pl_model'], kwargs['data_type'])
        del kwargs['pl_model'], kwargs['data_type']
    else:
        filenames_table = kwargs['filenames_table']
        del kwargs['filenames_table']


    # get the row which contains the specified parameters
    for param in kwargs:
        filenames_table = filenames_table[filenames_table[param] == kwargs[param]]

    if len(filenames_table) == 0:
        raise ValueError('No filename found for the specified parameters.')
    elif len(filenames_table) > 1:
        raise ValueError('Multiple filenames found for the specified parameters.')

    return filenames_table['filename'].values[0]


def get_param_by_filename_table(pl_model, data_type):
    """For all available datasets in the datasets folder, return a table which contains
    the relation between parameters and filename for all files of the specified type.

    Note: the corresponding filename is returned without the file ending, i.e., without '.txt' or '.pkl'.
    
    pl_model : str
        'fspl' or 'sionna'
    data_type : str
        'path_loss' or 'radiomap' or 'measurements'
    """

    if pl_model != 'fspl':
        raise NotImplementedError('Only fspl is implemented so far.')

    datasets_path = os.path.join(module_path, 'datasets')

    if data_type == 'path_loss':
        params_for_table = ['noise_std']
        filename_regex = os.path.join(datasets_path, 'fspl_PLdataset*.txt')
    elif data_type == 'radiomap':
        params_for_table = ['noise_std']
        filename_regex = os.path.join(datasets_path, 'fspl_RMdataset*.txt')
    elif data_type == 'measurements':
        params_for_table = ['noise_std', 'grid_size', 'tx_pos_inaccuracy_std']
        filename_regex = os.path.join(datasets_path, 'fspl_measurements*.txt')

    params_dict = {param: [] for param in params_for_table}
    params_dict['filename'] = []

    for file in glob(filename_regex):
        config = get_config_from_file(file)
        config['filename'] = file
        for param in params_for_table:
            params_dict[param].append(config[param])
        params_dict['filename'].append(file[:-4])

    return pd.DataFrame(params_dict)


if __name__ == '__main__':

    test_cases = ['get_config_from_file', 'get_param_by_filename_table']

    if 'get_config_from_file' in test_cases:
        datasets_path = module_path+"\\datasets"
        filename = 'fspl_measurements1.txt'
        path_to_file = os.path.join(datasets_path, filename)

        config = get_config_from_file(path_to_file)
        print(config)


    if 'get_param_by_filename_table' in test_cases:
        for data_type in ['path_loss', 'radiomap', 'measurements']:
            print(data_type)
            print(get_param_by_filename_table('fspl', data_type).to_string(index=False))
            print('\n')