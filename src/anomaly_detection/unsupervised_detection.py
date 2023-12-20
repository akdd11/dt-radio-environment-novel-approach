import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
from scipy.stats import wasserstein_distance, skew
import scipy as sp
import warnings
import sys
import os

from keras.models import Model
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, UpSampling2D

from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KernelDensity,LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM


# Implemented methods for unsupervised anomaly detection
unsupervised_methods = ['one_class_svm', 'unsupervised_threshold', 'cnn_autoencoder',
                         'unsupervised_density','isolation_forest', 'dbscan','lof',
                         'kmeans', 't_test', 'elliptic_envelope']


def sigmoid(x):
    """Sigmoid function.
    
    x: float or array_like
        Argument of the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


class CNNAutoencoder():
    """Convolutional neural network autoencoder.
    
    It uses a convolutional neural network to learn a lower dimensional representation of the
    input data (measurements), from which the measurements are reconstructed. The reconstruction
    error is used to detect anomalies.

    RMSE is used as reconstruction error metric.

    The classification is done based on Rajendran et al., "SAIFE: Unsupervised Wireless Spectrum
    Anomaly Detection with Interpretable Features", 2018. (p. 5)
    """

    def __init__(self, input_shape, n_sigma=3, verbose=0):
        """Initialize the autoencoder.
        
        Properties:
        input_shape : tuple
            Shape of the input data.
        n_sigma : int
            Number of standard deviations to use as threshold.
        verbose : int
            Verbosity level (0: not verbose, 1: verbose).
        """

        if not (input_shape[0] == 9 and input_shape[1] == 9 and len(input_shape) == 2):
            raise NotImplementedError('Only input shape [9, 9] is implemented.')
        
        self.n_sigma = n_sigma
        self.verbose = verbose

        input = Input(shape=(input_shape[0], input_shape[1], 1))
        x = Conv2D(64, kernel_size=3, activation='relu')(input)
        x = MaxPooling2D((2, 2))(x)
        x = UpSampling2D((3, 3))(x)
        x = Conv2D(filters=1, kernel_size=1)(x)

        model = Model(inputs=input, outputs=x)
        model.compile(optimizer='adam', loss='mse')

        if verbose > 0:
            model.summary()

        self.model = model

    def fit(self, X_train, epochs=10):
        """Train the classifier.
        
        Parameters:
        X_train : numpy.ndarray
            Training data.
        y_train : numpy.ndarray
            Training labels.
        """

        # train autoencoder (input=output)
        self.model.fit(X_train, X_train, epochs=epochs, verbose=self.verbose)

        # predict training data to evaluate goodness of fit
        X_pred = self.model.predict(X_train, verbose=self.verbose)
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]))

        # Calculate reconstruction error (RMSE) for each sample
        reconstruction_error = np.sqrt(np.mean(np.square(X_train - X_pred), axis=(1, 2)))

        # Save mean and standard deviation of reconstruction error to later on define,
        # which value of reconstruction error is considered an anomaly
        self.mean_reconstruction_error = np.mean(reconstruction_error)
        self.std_reconstruction_error = np.std(reconstruction_error)

    def predict(self, X_test):
        """Predict anomalies.
        
        Parameters:
        X_test : numpy.ndarray
            Test data.
        """

        if not hasattr(self, 'model'):
            raise NotFittedError('The model has not been trained yet.')

        X_pred = self.model.predict(X_test)
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]))
        reconstruction_error = np.sqrt(np.mean(np.square(X_test - X_pred), axis=(1, 2)))

        return reconstruction_error > self.mean_reconstruction_error + self.n_sigma * self.std_reconstruction_error


class UnsupervisedDensityClassifier:
    """Estimates the probability density function of the differences between DT and original measurements.
    Samples with a high deviation from the probability density function are classified as anomaly.

    The probability density is estimated using kernel density estimation (KDE). During the training
    phase the average probability density is calculated and a threshold for the anomaly is set based
    on the variance in the training data.

    For the prediction, the probability density is estimated for each sample and compared to the
    probability density function from the test data. If the deviation is to big, the sample is
    classified as anomaly.
    """

    def __init__(self, n_sigma=3, probability=False, verbosity=0):
        """Initialize the classifier.
        
        Properties:
        n_sigma : int
            Number of standard deviations to use as threshold.
        probability : bool
            If True, the probability density is returned instead of the binary classification.
        verbosity : int
            Verbosity level (0: not verbose, 1: verbose).
        """

        self.n_sigma = n_sigma
        self.probability = probability
        self.verbosity = verbosity
        
        if verbosity > 0:
            from tqdm import trange
            self.range_fun = trange
        else:
            self.range_fun = range


    def fit(self, X_train):
        """Train the classifier.
        
        X_train : numpy.ndarray
            Training data.
        """

        if self.verbosity > 0:
            print('Fitting overall KDE for normal samples...', end='  ', flush=True)

        # Estimate the overall probability density function of the measurement differences
        self.normal_kde = KernelDensity(bandwidth='scott').fit(X_train.ravel().reshape(-1, 1))

        if self.verbosity > 0:
            print('FINISHED')

        # Calculate the variance on of the deviation between the overall probability density function
        # and the probability density function of each sample
        eval_range = np.linspace(np.percentile(X_train, 0.1), np.percentile(X_train, 99.9), 100)
        eval_probabilities = np.exp(self.normal_kde.score_samples(eval_range.reshape(-1, 1)))

        if self.verbosity > 0:
            print('Evaluating training samples...', end='  ', flush=True)

        diff_list = np.zeros(X_train.shape[0])
        for i in self.range_fun(X_train.shape[0]):
            kde_sample = KernelDensity(bandwidth='scott').fit(X_train[i].ravel().reshape([-1, 1]))
            sample_probabilities = np.exp(kde_sample.score_samples(eval_range.reshape(-1, 1)))
            diff_list[i] = wasserstein_distance(sample_probabilities, eval_probabilities)
            
         # as the diff_list contains values from a metric, every value is positive -> no abs needed
        self.threshold = np.mean(diff_list) + self.n_sigma * np.std(diff_list)
        self.eval_range = eval_range    # values on which to score the PDF
        self.eval_probabilites = eval_probabilities    # PDF of the overall data


    def predict(self, X_test):
        """Predict anomalies.
        
        X_test : numpy.ndarray
            Test data.
        """

        if not hasattr(self, 'threshold'):
            raise NotFittedError('The classifier has not been trained yet.')
        
        if self.verbosity > 0:
            print('Predicting anomalies...', end='  ', flush=True)

        y_test = np.zeros(X_test.shape[0])
        for i in self.range_fun(X_test.shape[0]):
            kde_sample = KernelDensity(bandwidth='scott').fit(X_test[i].ravel().reshape([-1, 1]))
            sample_probabilities = np.exp(kde_sample.score_samples(self.eval_range.reshape(-1, 1)))
            diff = wasserstein_distance(sample_probabilities, self.eval_probabilites)
            if self.probability:
                y_test[i] = sigmoid(diff - self.threshold)
            else:
                y_test[i] = diff > self.threshold

        return y_test



class UnsupervisedThresholdClassifier:
    """Threshold-based classifier.
    
    It uses the a percentile of the mean or median difference between original data and the
    digital twin data as a threshold. If the difference is bigger than the threshold, the
    measurement is classified as an anomaly.

    This means, that there will be always false positives, i.e., anomalies are detected, even
    if there are no anomalies in the data (and also the other way around is likely to happen).
    """

    def __init__(self, percentile=0.9, probability=False, statistic='mean', use_absolute_value=False):
        """Initialize the classifier.
        
        Properties:
        percentile : float
            Percentile to use for the threshold (0-1).
        probability : bool
            If True, the probability of the measurement being an anomaly is returned instead of
            the binary classification. 
        statistic : str
            Statistic to use for the threshold. Can be 'mean' or 'median'.
        use_absolute_value : bool
            If True, the absolute value of the statistic is used.
        """

        self.percentile = percentile*100
        self.probability = probability

        if statistic in ['mean', 'median','skewness','var']:
            self.statisitc = statistic
        else:
            raise ValueError('Statistic has to be either mean or median.')
        
        self.use_absolute_value = use_absolute_value
        self.threshold = np.nan

    def fit(self, X_train):
        """Train the classifier.
        
        Parameters:
        X_train : numpy.ndarray
            Training data.
        """

        if self.use_absolute_value:
            X_train = np.abs(X_train)
        
        if self.statisitc == 'mean':
            X_train = np.mean(X_train, axis=1)
        elif self.statisitc == 'median':
            X_train = np.median(X_train, axis=1)
        elif self.statisitc == 'skewness':
            X_train = skew(X_train,axis=1)
        elif self.statisitc == 'var':
            X_train = np.var(X_train,axis=1)

        self.threshold = np.percentile(X_train, self.percentile)

    def predict(self, X_test):
        """Predict anomalies.
        
        Parameters:
        X_test : numpy.ndarray
            Test data.
        """

        if np.isnan(self.threshold):
            raise NotFittedError('The classifier has to be fitted first.')

        if self.use_absolute_value:
            X_test = np.abs(X_test)
        
        if self.statisitc == 'mean':
            X_test = np.mean(X_test, axis=1)
        elif self.statisitc == 'median':
            X_test = np.median(X_test, axis=1)
        elif self.statisitc == 'skewness':
            X_test = skew(X_test,axis=1)
        elif self.statisitc == 'var':
            X_test = np.var(X_test,axis=1)

        if self.probability:
            return sigmoid(X_test - self.threshold)
        else:
            return X_test > self.threshold
        

    def visualize_distributions(self, X_train, X_test, y_train, y_test):
        """Visualize the distributions of the training and test data.
        
        Parameters:
        X_train : numpy.ndarray
            Training data.
        X_test : numpy.ndarray
            Test data.
        y_train : numpy.ndarray
            Training labels.
        y_test : numpy.ndarray
            Test labels.
        """

        if self.use_absolute_value:
            X_train = np.abs(X_train)
            X_test = np.abs(X_test)
        
        if self.statisitc == 'mean':
            X_train = np.mean(X_train, axis=1)
            X_test = np.mean(X_test, axis=1)
        elif self.statisitc == 'median':
            X_train = np.median(X_train, axis=1)
            X_test = np.median(X_test, axis=1)

        # Plot the distributions
        sns.kdeplot(X_train[y_train == 0], color='C0', linestyle='-')
        sns.kdeplot(X_train[y_train == 1], color='C0', linestyle='--')
        sns.kdeplot(X_test[y_test == 0], color='C1', linestyle='-')
        sns.kdeplot(X_test[y_test == 1], color='C1', linestyle='--')

        # Create the legend entries
        plt.plot(np.nan, np.nan, color='black', linestyle='-', label='Normal')
        plt.plot(np.nan, np.nan, color='black', linestyle='--', label='Anomaly')
        plt.plot(np.nan, np.nan, color='C0', linestyle='-', label='Train')
        plt.plot(np.nan, np.nan, color='C1', linestyle='-', label='Test')

        # Plot the threshold
        if np.isnan(self.threshold):
            warnings.warn('The classifier has not yet been fitted. Hence, no threshold is shown.')
        else:
            plt.axvline(self.threshold, color='red', linestyle=':', label='Threshold')


        plt.xlabel('Measurement difference (original - digital twin) [dB]')
        plt.legend()
        plt.show()

class IsolationForestClassifier:
    """IsolationForest

    It used binary trees for anamoly detection and has linear time complxity. It splits the data
    space with parallel lines and score each data-point on number of splits required to spearate it.
    The fewer the slits the higher the anamoly score (the probablity of that data-point being an anamoly.)
    """

    def __init__(self,n_estimators=100,max_samples=256,contamination='auto',verbose=0):
        """Initialization

            n_estimator : int
                no of base estimatiors (trees) in the ensemble.
            max_samples : int or float
                no of samples to consider for training each base estimator.
            contamination : float
                propotion of outliers in the dataset.
            verbose : 1 or 0
        """
        self.n_estimator = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.verbose = verbose

        self.cls = IsolationForest(n_estimators=n_estimators,
                            max_samples=self.max_samples,
                            contamination=self.contamination,
                            verbose=self.verbose)

    def fit(self,X_train):
        """Train the model

        Parameter:
        X_train : np.ndarray
            Training data
        """  

        self.cls.fit(X_train)

    def predict(self,X_test):
        """Make anamoly predictions

        Parameter
        X_test : np.ndarray
            Test data
        """
        return self.cls.predict(X_test)

class DBSCANCLearning:
    """Density based spatial clustering of application
    It is a density-based clustering non-parametric algorithm. It groups together points 
    that are closely packed together (points with many nearby neighbors), 
    marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).
    """

    def __init__(self, eps=0.5, min_samples=5, probability=False, preprocessing=None, grid_size=None):
        """Initialization

        eps : float
            Maximum distance between two samples for one to be considered as in the  neighborhood of the other.
        min_samples : int
            No. of samples in a neighborhood for a point to be considered as a core point (inclusive).
        probability : bool
            If True, the probability of the measurement being an anomaly is returned instead of
            the cluster number for each point or -1 for anomalies, respectively.
        preprocessing : {None, 'sub-avg', 'full-avg', 'sort','sort++'}
            Type of preprocessing to be performed of data point before training or predicting.
        grid_size : None or int
            Distance between two transmitters placed in space.
        """

        self.eps = eps
        self.min_samples = min_samples
        self.probability = probability
        self.preprocessing = preprocessing
        self.grid_size = grid_size

        self.cls = DBSCAN(eps=self.eps,
                           min_samples=self.min_samples)

    def fit(self, X_train):
        """Train the model
        
        Parameter
        X_train : np.ndarray
            Training data
        """
        if self.preprocessing == 'full-avg':
            X_train = np.mean(X_train, axis=1).reshape(-1,1)

        elif self.preprocessing == 'sub-avg':
            if self.grid_size == None:
                raise NotImplementedError("Grid size not specified. \n"
                                          "Grid size must be specified when preprocessing of 'sub-avg' type.")
            X_train = sub_processing_2D(X_train,'mean',self.grid_size)

        elif self.preprocessing == 'sort':
            X_train = np.sort(X_train,axis=1)

        elif self.preprocessing == 'sort++':
            if self.grid_size == None:
                raise NotImplementedError("Grid size not specified. \n"
                                          "Grid size must be specified when preprocessing of 'sort++' type.")

            X_train = sub_processing_2D(X_train, 'max', self.grid_size)

        self.cls.fit(X_train)

        if len(np.unique(self.cls.labels_)) > 2:
            warnings.warn('More than one cluster detected.')


    def predict(self,X_test):
        """Make anamoly predictions.

        Parameter
        X_test : np.ndarray
            Test data
        """
        if self.preprocessing == 'full-avg':
            X_test = np.mean(X_test, axis=1).reshape(-1,1)

        elif self.preprocessing == 'sub-avg':
            if self.grid_size == None:
                raise NotImplementedError("Grid size not specified. \n"
                                          "Grid size must be specified when preprocessing of 'sub_avg' type.")
            X_test = sub_processing_2D(X_test,'mean',self.grid_size)

        elif self.preprocessing == 'sort':
            X_test = np.sort(X_test, axis=1)

        elif self.preprocessing == 'sort++':

            if self.grid_size == None:
                 raise NotImplementedError("Grid size not specified. \n"
                                           "Grid size must be specified when preprocessing of 'sort++' type.")
        #   X_test = np.sort(X_test, axis=1)
            X_test = sub_processing_2D(X_test,'max',self.grid_size)

        nr_samples = X_test.shape[0]
        if self.probability:
            y_hat = np.empty(shape=nr_samples, dtype=float)
        else:
            y_hat = np.ones(shape=nr_samples, dtype=int) * -1 # -1 indicates an outlier

        # label the points according to the closest core point if the distance is less than eps
        # or calculate the probability of the point being an outlier if probability is True
        for i in range(nr_samples):
            diff = self.cls.components_ - X_test[i, :]  # NumPy broadcasting, components_ are all core samples
            dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
            shortest_dist_idx = np.argmin(dist)  # find closest core sample
            if self.probability:
                y_hat[i] = sigmoid(dist[shortest_dist_idx] - self.cls.eps)
            else:
                if dist[shortest_dist_idx] < self.cls.eps:
                    y_hat[i] = self.cls.labels_[self.cls.core_sample_indices_[shortest_dist_idx]]
        return y_hat


class OCSVMClassifier:
    """Wrapper for the sklearn one-class SVM classifier."""

    def __init__(self, probability=False, **kwargs):
        """Initialize the classifier.
        
        probability : bool
            If True, the probability of the measurement being an anomaly is returned instead of
            the binary classification.
        **kwargs : dict
            Additional arguments for the sklearn one-class SVM classifier.
        """

        self.probability = probability
        self.cls = OneClassSVM(**kwargs)

    def fit(self, X_train):
        """Train the classifier.
        
        Parameters:
        X_train : numpy.ndarray
            Training data.
        """

        self.cls.fit(X_train)

    def predict(self, X_test):
        """Predict anomalies.
        
        Parameters:
        X_test : numpy.ndarray
            Test data.
        """

        if self.probability:
            return self.cls.decision_function(X_test)
        else:
            return self.cls.predict(X_test)

class LocalOutlierFactorLearning:
    """LocalOutlierFactorLearing

     It measures the local deviation of the density of a given sample with respect to its neighbors.
     Based on this, it calculates an anomaly score, the Local Outlier Factor (LOF), for each sample.
    """

    def __init__(self, n_neighbors=20, algorithm='auto', contamination='auto', probability=False):
        """Initialization

            n_neighbors : int
                no of neighbors (nearest) to consider for density calculation.
            alogrithm : {auto,ball_tree,kd_tree, brute}
                algorithm to use for calculating nearest neighbors.
            probability : bool
                If True, the probability of the measurement being an anomaly is returned instead of
                the binary classification.
        """

        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.contamination = contamination
        self.probability = probability

        self.cls = LocalOutlierFactor(n_neighbors=self.n_neighbors,
                                      algorithm=self.algorithm,
                                      contamination=self.contamination,
                                      novelty=True)


    def fit(self, X_train):
        """Train the model

        Parameter:
        X_train : np.ndarray
            Training data
        """  

        self.cls.fit(X_train)

    def predict(self, X_test):
        """Make anamoly predictions

        Parameter
        X_test : np.ndarray
            Test data
        """
        if self.probability:
            # a minus sign is added, as the decision function return bigger values for inliers
            return sigmoid(-self.cls.decision_function(X_test))
        else:
            return self.cls.predict(X_test)


class KMeansLearning:
    """K-Means clustering

                <INSERT DEFINITION HERE>
    """

    def __init__(self,n_clusters=8,n_init=1,max_iter=300):
        """Initialization

            n_clusters : int
                no. of clusters to form (no of centroids).
            n_init : 'auto' or int
                no. of times k-means algorithm is run with different centroid seeds.
            max_iter : int
                max no. of iteration of algorithm for a single run.

        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter

        self.cls = KMeans(n_clusters=self.n_clusters,
                          n_init=self.n_init,
                          max_iter=self.max_iter)


    def fit(self,X_train):
        """Train the model

        Parameter:
        X_train : np.ndarray
            Training data
        """
        X_train = np.mean(X_train,axis=1).reshape(-1,1)
        self.cls.fit(X_train)

    def predict(self,X_test):
        """Make anamoly predictions

        Parameter
        X_test : np.ndarray
            Test data
        """
        X_test = np.mean(X_test,axis=1).reshape(-1,1)
        return self.cls.predict(X_test)


class EllipticEnvelopeClassifier:
    """ Elliptic Envelope outlier detection.
    
    Identifies outliers in a Gaussian distributed dataset.
    Purely sklearn-based implementation.
    """

    def __init__(self, probability=False) -> None:
        """Initialization of the sklearn model.
        
        probability : bool
            If True, the probability of the measurement being an anomaly is returned instead of
            the binary classification.
        """

        self.probability = probability

        self.model = EllipticEnvelope(contamination=1e-9)
        # note: very small contamination, as training is only on normal data

    def fit(self, X_train):
        """Train the model.
        
        X_train : np.ndarray
            Training data.
        """

        self.model.fit(X_train)

    def predict(self, X_test):
        """ Make outlier prediction.
        
        X_test : np.ndarray
            Test data.
        """
        
        if self.probability:
            return self.model.decision_function(X_test)
        else:
            return self.model.predict(X_test)


class TtestClassifier:
    """T-test classifier.

    Assumption is that the mean of the differences is 0 if no jammer
    is present. This is tested using the T test.
    https://de.wikipedia.org/wiki/Einstichproben-t-Test#Einseitiger_Test
    """

    def __init__(self, alpha=0.95):
        """Initialization.
        
        alpha : float
            Confidence level.
        """
        self.alpha = alpha

    def fit(self, X_train):
        """Train the model. Dummy function, as no training is required.

        
        X_train : np.ndarray
            Training data
        """
        pass

    def predict(self, X_test):
        """Make anamoly predictions using one-sided t-test.

        X_test : np.ndarray
            Test data
        """

        nr_samples = X_test.shape[0]    # number of samples to detect anomalies
        n_meas = X_test.shape[1]        # number of measurements per sample

        y_hat = np.zeros(shape=nr_samples)

        critical_value = sp.stats.t.ppf(self.alpha, n_meas - 1)

        for i in range(nr_samples):

            # Calculate the t value
            t_value = np.sqrt(n_meas) * (np.mean(X_test[i, :]) / np.std(X_test[i, :]))

            # Compare the t value with the critical value
            if t_value < -critical_value:
                y_hat[i] = 1

def sub_processing_2D(dist,statistic,grid_size):
    """Perform preprocessing after dividing data into sub part.
    Done before training and testing.

    dist : np.ndarray
        Train or Test data
    statistic : {'max', 'mean'}
        Type of operation to perform on each sub parts
    grid_size : int
        Distance between two transmitters placed in space
    """

    grid_len = (40//grid_size) + 1
    res = 2
    output = []

    for sample in dist:
        sub_div = []
        col = 0
        grid = np.array(sample).reshape(grid_len,grid_len).T

        for col in range(0,grid_len,res):
            for i in range(0,grid_len,res):
                if statistic == 'max':
                    sub_div.append(np.max(grid[i:i + res, col:col + res]))
                elif statistic == 'mean':
                    sub_div.append(np.mean(grid[i:i + res, col:col + res]))
        output.append(sub_div)
    return np.asarray(output)