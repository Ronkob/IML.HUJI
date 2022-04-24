from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        assert (int(X.shape[0]) == int(len(y)))

        self.classes_, class_inverse, class_counts = np.unique(y, return_counts=True, return_inverse=True)

        self.pi_ = class_counts / len(y)  # vector of size classes
        self.mu_ = np.zeros((len(self.classes_), X.shape[1]))  # mu of size (classes_, features)

        for i, class_value in enumerate(self.classes_):
            self.mu_[i] = sum(X[class_inverse == i, :]) / class_counts[i]  # each class in mu is a row of feature means

        self.vars_ = np.zeros(shape=(len(self.classes_), X.shape[1]))  # variance of size (classes, features)

        for class_number, class_value in enumerate(self.classes_):
            sigma_k = np.zeros(shape=(X.shape[1], X.shape[1]))
            for i, sample in enumerate(X[y == class_value]):
                c = sample - self.mu_[class_number]
                sigma_k += np.outer(c, c)
            sigma_k = sigma_k / class_counts[class_number]  # diagonal variance matrix for class k
            self.vars_[class_number, :] = np.diagonal(sigma_k)



    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        all_bayes = self.likelihood(X)
        predictions = np.argmax(all_bayes, axis=1)
        y_pred = np.array(list([self.classes_[pred_class_number] for pred_class_number in predictions]))
        return y_pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = np.zeros((X.shape[0], len(self.classes_)))  # should be a (samples, classes_) matrix
        for i, sample in enumerate(X):
            for class_value in self.classes_:
                mu_k = self.mu_[self.classes_ == class_value]
                cov = np.diagflat(self.vars_[self.classes_ == class_value, :])
                z = np.sqrt(np.power((2 * np.pi), X.shape[1]) * np.linalg.det(cov))
                c = sample - mu_k
                exponent = -0.5 * np.dot(c, np.dot(np.linalg.pinv(cov), c.T).flatten())
                likelihoods[i, class_value] = self.pi_[self.classes_ == class_value] * np.exp(exponent) / z

        return likelihoods


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
