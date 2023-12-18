# Digital Twin of the Radio Environment: A Novel Approach for Anomaly Detection in Wireless Networks

This code is used for the paper:

 A. Krause, M.D. Khursheed, P. Schulz, F. Burmeister and G. Fettweis, "Digital Twin of the Radio Environment: A Novel Approach for Anomaly Detection in Wireless Networks." arXiv preprint arXiv:2308.06980 (2023).

 It can be found here [https://arxiv.org/abs/2308.06980](https://arxiv.org/abs/2308.06980).

## Workflow

The workflow is as follows:

1. Create pathloss map dataset using `src\dataset_generation\pathloss_map_generation.py`.
2. Create a radio map dataset using `src\dataset_generation\radio_map_generation.py`.
3. Create a measurement dataset using `src\dataset_generation\measurement_generation.py`.

### Data Formats

*fspl* indicates free space path loss model was used calculcate the path loss.

* **Path loss map:** Contains a collection of path loss maps. File name is *fspl_PLdataset\<nr\>.pkl*. Each path loss map indicates the path loss between a specific transmitter location and each pixel on the map in dB.
* **Radio map:** Contains a collection of radio maps. File name is *fspl_RMdataset\<nr\>.pkl*. Each radio maps combines one or several transmitters (regular or jammer) with a specific power and path loss and adds everything up, so that the result is the total RSS at each pixel on the map in dBm.
* **Measurements:** Contains a collection of measurements. File name is *fspl_measurements\<nr\>.pkl*. Each measurement entity is a collection of values, whereby each value is the difference between the measured (original) RSS and the RSS expected from the digital twin at the same location.
* **Results:** Contains the results of anomaly detecion for a given measurement data set. File name *fspl_results\<nr\>.pkl*. Each file contains a dictionary with the following entries: ```y_test```, ```y_hat``` and ```jammer```. ```jammer``` refers to an array, in which the transmitters of type jammer are saved or an empty array, respectively in case there is no jammer.

## Methods for generation of path loss maps

### Free space path loss

$$ L = 10 \alpha \log \left( \frac{d}{d_0} \right)  + \beta + 10 \gamma \left( \frac{f}{f_0} \right)$$

with $d_0 = 1\text{m}$ and $f_0 = 1\text{Hz}$.

Defaults:
* $\alpha = 2$
* $\beta = - 147.55\text{dB}$
* $\gamma = 2$

Correlated shadowing noise can be added to the original map which has a covariance matrix with the entries

$$[\mathbf{C}]_{a,b} = \sigma_{\text{dB}}^2 \exp\left( -\frac{d(\mathbf{x}_a, \mathbf{x}_b)}{d_{\text{cor}}} \right)$$

for each points $\mathbf{x}_a$ and $\mathbf{x}_b$. $d_{\text{cor}}$ defaults to $1 \text{m}$.

## Methods for outlier detection

### Conventions

* Binary classification:
  * 0: Normal
  * 1: Anomality

* The indexing is [x, y], even though numpy uses matrix indexing [i, j]. This has to be considered when plotting.

### Unsupervised

* One-class SVM
* Local outlier factor
* Adapted energy detector


## Other stuff

### Positioning inaccuracy

See paper > IV.A Building the Digital Twin for more information on how the poisitioning inaccuracy is modeled.