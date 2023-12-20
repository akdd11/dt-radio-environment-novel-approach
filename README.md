**WORK IN PROGRESS: Please note, that the code upload and documentation is not yet finished.** 

# Digital Twin of the Radio Environment: A Novel Approach for Anomaly Detection in Wireless Networks

This code is used for the paper:

 A. Krause, M.D. Khursheed, P. Schulz, F. Burmeister and G. Fettweis, "Digital Twin of the Radio Environment: A Novel Approach for Anomaly Detection in Wireless Networks." arXiv preprint arXiv:2308.06980 (2023).

 It can be found here [https://arxiv.org/abs/2308.06980](https://arxiv.org/abs/2308.06980).

## Workflow

The workflow is split into two parts: dataset generation and anomaly detection on the generated data.

### Data Formats

The following data formats are used in the process of results generation. *fspl* indicates that the free-space path loss model was to used to generate the path loss maps.

* **Path loss map:** Contains a collection of path loss maps. File name is *fspl_PLdataset\<nr\>.pkl*. Each path loss map indicates the path loss between a specific transmitter location and each pixel on the map in dB.
* **Radio map:** Contains a collection of radio maps. File name is *fspl_RMdataset\<nr\>.pkl*. Each radio maps combines one or several transmitters (regular or jammer) with a specific power and path loss and adds everything up, so that the result is the total RSS at each pixel on the map in dBm.
* **Measurements:** Contains a collection of measurements. File name is *fspl_measurements\<nr\>.pkl*. Each measurement entity is a collection of values, whereby each value is the difference between the measured (original) RSS and the RSS expected from the digital twin at the same location.
* **Results:** Contains the results of anomaly detecion for a given measurement data set. File name *fspl_results\<nr\>.pkl*. Each file contains a dictionary with the following entries: ```y_test```, ```y_hat``` and ```jammer```. ```jammer``` refers to an array, in which the transmitters of type jammer are saved or an empty array, respectively in case there is no jammer.

### Dataset Generation
The dataset generation consists of three steps. For each script, there is a corresponding `yaml` file in the `conf` folder, which contains the parameters for each step.

1. Create pathloss map dataset using `src\dataset_generation\pathloss_map_generation.py`. In this step, for random transmitter positions path loss maps including shadowing are generated.
2. Create a radio map dataset using `src\dataset_generation\radio_map_generation.py`. First, the number of regular transmitters $N_{reg}$ is specified. Then, one pathlossmap is randomly chosen for each transmitter (i.e., random locations of the transmitters). Each pathloss map is converted into a radiomap by taking the transmit power into account($P_{rx} = p_{tx} - L). The received power are added up to obtain the total received power at each pixel on the map. For some maps (according to the specified jammer probability), a jammer is added to the map. The jammer is also based on one of the pre-generated pathloss maps.
3. Create a measurement dataset using `src\dataset_generation\measurement_generation.py`. In this step, for each of the radio map from step 2, a digital twin radio map is generated (see Section IV.A in the paper for more information). Subsequently, from the sensing unit positions the RSS values are extracted from both the physical twin and the digital twin radio map and the difference is calculated.

Typically, there is one pathloss map dataset with a given shadowing variance. From this, one radio map dataset is generated for a specified number of transmitters. From the radio map different measurement datasets for different grid sizes / number of sensing units can be generated.

### Anomaly Detection

The script `src\anomaly_detection\fspl_anomaly_detection.py` executes the anomaly detection. The script takes the measurement datasets as input. For each measurement dataset, the anomaly detection is executed and the results are saved in a results file. The results file contains for each method and each sample a score/probability, whether the sample is an anomaly or not (as long as in `fspl_anomaly_detection.yaml` the parameter `probability` is set to `True`).

In the file `fspl_anomaly_detection.yaml` in the `conf` folder, the parameters for the anomaly detection can be specified. The most important parameters are:
* `vary_parameter`: The parameter over which is iterated when executing `fspl_anomaly_detection.py`. Can be either `noise_std` or `grid_size`.
* `method`: The anomaly detection algorithm.

You can find more information in the `yaml` file.

After the results are generated, they can be visualized using the Jupyter notebook `notebooks\roc_curves_fspl.ipynb`. 

## Methods for Anomaly Detection

### Conventions

* Binary classification:
  * 0: Normal
  * 1: Anomality

* The indexing is [x, y], even though numpy uses matrix indexing [i, j]. This has to be considered when plotting.

### Unsupervised Methods

The best performing unsupervised methods are:

* One-class SVM
* Local outlier factor
* Adapted energy detector (in the code, it is referred to as unsupervised_threshold)

Yet, there are more unsupervised and even supervised methods for comparison implemented. Please note, that they may not be fully mature.

## Other stuff

### Positioning inaccuracy

See paper > IV.A Building the Digital Twin for more information on how the poisitioning inaccuracy is modeled.