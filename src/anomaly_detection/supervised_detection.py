import matplotlib.pyplot as plt
import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D,AveragePooling2D,Dropout

from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KernelDensity,KNeighborsClassifier

# Implemented methods for supervised anomaly detection
supervised_methods = ['cnn_classifier', 'supervised_threshold', 'knn_classifier']


class CNNClassifier():

    def __init__(self, input_shape, verbose=0):
        """Initialize the classifier.
        
        Properties:
        input_shape : tuple
            Shape of the input data.
        verbose : int
            Verbosity level (0: not verbose, 1: verbose).
        """

        self.verbose = verbose

        input = Input(shape=(input_shape[0], input_shape[1], 1))
        x = Conv2D(2, kernel_size=(3,3),activation='relu')(input)           #Output shape : ((input_shape - kernel_size )+1)**2 * filters
        x = MaxPooling2D(pool_size=(3, 3),strides=1,padding='same')(x)      #Output shape : ((prev_output - pool_size)+1)**2 * filters
        x = Flatten()(x)
        x = Dense(49, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=input, outputs=x)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',CNNClassifier.F1_score])

        if verbose > 0:
            model.summary()

        self.model = model

    def fit(self, X_train, y_train, epochs=75):
        """Train the classifier.
        
        Parameters:
        X_train : numpy.ndarray
            Training data.
        y_train : numpy.ndarray
            Training labels.
        """

        # TODO: can **kwargs be passed directly to model.fit?

        self.model.fit(X_train, y_train, epochs=epochs, verbose=self.verbose)

    def predict(self, X_test):
        """Predict anomalies.
        
        Parameters:
        X_test : numpy.ndarray
            Test data.
        """

        return self.model.predict(X_test, verbose=self.verbose) > 0.5

    def F1_score(y_true, y_pred):  # taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val


class SupervisedThresholdClassifier:
    """Threshold-based classifier.
    
    Estimates the distributions of the differences between the measurements from the original
    data and the digital twin data. The difference hereby refers to the mean or median difference
    of all samples in one scenario. For a given value, decides depending on the estimated
    distributions, whether the value is an anomaly or not.
    """

    def __init__(self, statistic, use_absolute_value=False):
        """Initialize the classifier.
        
        Properties:
        statistic : str
            Statistic to use for the threshold. Can be 'mean' or 'median'.
        use_absolute_value : bool
            If True, the absolute value of the statistic is used.
        """

        if statistic in ['mean', 'median']:
            self.statistic = statistic
        else:
            raise ValueError('Statistic has to be either mean or median.')
        
        self.use_absolute_value = use_absolute_value

    def fit(self, X_train, y_train):
        """Train the classifier.
        
        Parameters:
        X_train : numpy.ndarray
            Training data.
        y_train : numpy.ndarray
            Training labels.
        """

        if self.use_absolute_value:
            X_train = np.abs(X_train)
        
        # Calculating one value per scenario from all measurements
        if self.statistic == 'mean':
            X_train = np.mean(X_train, axis=1)
        elif self.statistic == 'median':
            X_train = np.median(X_train, axis=1)

        
        # seperate the data into two groups: normal and anomaly
        # normal: y_train == 0
        # anomaly: y_train == 1

        X_train_normal = X_train[y_train == 0]
        X_train_anomaly = X_train[y_train == 1]

        # save the data ranges for visualization
        self.value_range_normal = (np.min(X_train_normal), np.max(X_train_normal))
        self.value_range_anomaly = (np.min(X_train_anomaly), np.max(X_train_anomaly))

        # estimate the distributions
        self.distribution_normal = KernelDensity(bandwidth='scott').fit(X_train_normal.reshape(-1, 1))
        self.distribution_anomaly = KernelDensity(bandwidth='scott').fit(X_train_anomaly.reshape(-1, 1))

    def predict(self, X_test):
        """Predict anomalies.
        
        Parameters:
        X_test : numpy.ndarray
            Test data.
        """

        if self.use_absolute_value:
            X_test = np.abs(X_test)
        
        if self.statistic == 'mean':
            X_test = np.mean(X_test, axis=1)
        elif self.statistic == 'median':
            X_test = np.median(X_test, axis=1)

        # estimate the log probability density for each value
        log_prob_normal = self.distribution_normal.score_samples(X_test.reshape(-1, 1))
        log_prob_anomaly = self.distribution_anomaly.score_samples(X_test.reshape(-1, 1))

        # return 1 if the value is an anomaly, 0 otherwise
        return (log_prob_anomaly > log_prob_normal).astype(int)
    
    def visualize_distributions(self):
        """Visualize the estimated distributions.
        """

        if not hasattr(self, 'distribution_normal'):
            raise NotFittedError('The classifier has to be fitted first.')
        
        range_extension = 0.5

        # plot the distributions
        x = np.linspace(self.value_range_normal[0]-range_extension, self.value_range_normal[1]+range_extension, 100)
        log_prob_normal = self.distribution_normal.score_samples(x.reshape(-1, 1))
        plt.plot(x, np.exp(log_prob_normal), label='normal')

        x = np.linspace(self.value_range_anomaly[0]-range_extension, self.value_range_anomaly[1]+range_extension, 100)
        log_prob_anomaly = self.distribution_anomaly.score_samples(x.reshape(-1, 1))
        plt.plot(x, np.exp(log_prob_anomaly), label='anomaly')

        plt.legend()
        plt.show()

class KNNClassifier:
    """K-nearest neighbors clustering

                <INSERT DEFINATION HERE>
    """

    def __init__(self,n_neighbors=5,weights='uniform'):
        """Initialization

            n_neighbors : int
                no. of neighbors to consider
            weights : 'uniform' or 'distance'
                impact of each neighbor on deciding the label.
        """
        self.n_neighbor = n_neighbors
        self.weights = weights
     
        self.cls = KNeighborsClassifier(n_neighbors=self.n_neighbor,
                                        weights=self.weights)


    def fit(self,X_train,y_train):
        """Train the model

        Parameter:
        X_train : np.ndarray
            Training data
        """  

        self.cls.fit(X_train,y_train)

    def predict(self,X_test):
        """Make anamoly predictions

        Parameter
        X_test : np.ndarray
            Test data
        """
