import numpy as np
from numpy.linalg import inv as inv
from numpy.linalg import multi_dot as multi_dot
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin


def relu(x):
    my_relu = np.maximum(x, 0)
    return my_relu


def sigmoid(x):
    my_sigmoid = 1 / (1 + np.exp(-x))
    return my_sigmoid


class ExtremeLearningMachine(BaseEstimator, ClassifierMixin):
    """
        Extreme Learning Machine for imbalanced learning
        Klasyfikator Extreme Learning Machine na potrzeby klasyfikacji niezbalansowanej
        """

    def __init__(self, C, hidden_units=1000):
        self.hidden_units = hidden_units
        self.C = C

    def fit(self, X, y):

        # dorobić walidaotr
        self.X_, self.y_ = check_X_y(X, y)
        self.training_samples = self.X_.shape[0]
        self.input_units = self.X_.shape[1]
        self.input_weights = np.random.uniform(size=[self.hidden_units, self.input_units])
        self.biases = np.random.uniform(size=[self.hidden_units])

        input_dot_product = np.dot(self.X_, self.input_weights.T)
        add_biases = input_dot_product + self.biases
        self.H = sigmoid(add_biases)

        self.W = np.zeros((self.training_samples, self.training_samples))
        positive_samples = np.count_nonzero(self.y_ == 1)
        # użycie y, bo dla self.y_ woła "can’t assign to function call Solution"
        negative_samples = self.training_samples - positive_samples
        positive_ratio = 1 / positive_samples
        negative_ratio = 1 / negative_samples

        for i in range(self.training_samples):
            if self.y_[i] == 1:
                self.W[i, i] = positive_ratio
            else:
                self.W[i, i] = negative_ratio

        if self.training_samples < self.hidden_units:
            inverse_complicated_part = inv(np.eye(self.training_samples) / self.C + multi_dot([self.W, self.H, self.H.T]))
            self.output_weights = multi_dot([self.H.T, inverse_complicated_part, self.W, self.y_])
        else:
            inverse_complicated_part = inv(np.eye(self.hidden_units) / self.C + multi_dot([self.H.T, self.W, self.H]))
            self.output_weights = multi_dot([inverse_complicated_part, self.H.T, self.W, self.y_])

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        input_dot_product = np.dot(X, self.input_weights.T)
        add_biases = input_dot_product + self.biases
        prediction_H = sigmoid(add_biases)
        output = np.dot(prediction_H, self.output_weights)
        return np.sign(output)

