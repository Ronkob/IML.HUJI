from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        assert (int(X.shape[1]) == int(len(y)), f'X and y sizes dont match up, X size {X.shape}, y size {y.shape}')

        self.classes, class_inverse, class_counts = np.unique(y, return_counts=True, return_inverse=True)
        print(f'print from lda fit. \n  this is y {y[1:5]} and this is self.classes {self.classes}')
        self.mu_ = np.zeros((len(self.classes), X.shape[1]))  # mu of size (classes, features)
        print(f'print from lda fit. \n  this is mu \n   {self.mu_}')
        for i, class_value in enumerate(self.classes):
            self.mu_[i] = sum(X[class_inverse == i, :]) / class_counts[i] # each class in mu is a row of feature means

        self.cov_ = np.zeros(shape=(X.shape[1], X.shape[1]))  # cov of size (features, features)
        for i, sample in enumerate(X):
            c = sample - self.mu_[self.classes == y[i]]
            self.cov_ += np.outer(c, c)
        print(f'print from lda fit. \n  this is self_cov \n  {self.cov_}')
        self.cov_ = self.cov_ / (X.shape[0] - len(self.classes))  # unbiased estimator for the covariance

        self.pi_ = class_counts / len(y)  # vector of size k

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
        y_pred = np.array(self.classes_[pred_class_number] for pred_class_number in predictions)
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

        likelihoods = np.zeros((X.shape[0], self.classes_))
        for sample in X:
            for class_value in self.classes_:
                mu_k = self.mu_[self.classes == class_value]
                a_k = np.array(self._cov_inv * mu_k)
                b_k = np.array(np.log(self.pi_) - 0.5*mu_k.T*self._cov_inv*mu_k)
                likelihoods[sample, class_value] = a_k @ sample + b_k
                print("Printed in likelihood comp in LDA class")

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

