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
        assert (int(X.shape[0]) == int(len(y)))
        
        self.classes_, class_inverse, class_counts = np.unique(y, return_counts=True, return_inverse=True)
        # print(f'print from lda fit. \n  this is y {y[1:5]} and this is self.classes_ {self.classes_}')
        self.mu_ = np.zeros((len(self.classes_), X.shape[1]))  # mu of size (classes_, features)
        # print(f'print from lda fit. \n  this is mu \n   {self.mu_}')
        for i, class_value in enumerate(self.classes_):
            self.mu_[i] = sum(X[class_inverse == i, :]) / class_counts[i] # each class in mu is a row of feature means

        self.cov_ = np.zeros(shape=(X.shape[1], X.shape[1]))  # cov of size (features, features)
        for i, sample in enumerate(X):
            c = sample - self.mu_[self.classes_ == y[i]]
            self.cov_ += np.outer(c, c)
        # print(f'print from lda fit. \n  this is self_cov \n  {self.cov_}')
        self.cov_ = self.cov_ / (X.shape[0] - len(self.classes_))  # unbiased estimator for the covariance
        self._cov_inv = np.linalg.pinv(self.cov_)
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
        y_pred = np.array(list([self.classes_[pred_class_number] for pred_class_number in predictions]))
        return y_pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes_.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, classes_)
            The likelihood for each sample under each of the classes_

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = np.zeros((X.shape[0], len(self.classes_)))  # should be a (samples, classes_) matrix
        for i, sample in enumerate(X):
            for class_value in self.classes_:
                mu_k = self.mu_[self.classes_ == class_value]
                # print(f'self._cov_inv * mu_k {np.dot(self._cov_inv, mu_k.T)}')
                a_k = np.dot(np.array(np.dot(self._cov_inv, mu_k.T)).flatten(), sample)
                # print(f'Printed in likelihoods comp \n  self.cov_inv size: {self._cov_inv.shape}'
                #      f'\n  mu_k: {mu_k} \n  mu_k_transposed: {mu_k.T} \n self_classes {self.classes_}, \n self_mu {self.mu_}, \n self_pi {self.pi_}')
                b_k = (np.log(self.pi_[self.classes_ == class_value]))
                # print(f'Printed in likelihoods comp \n {np.dot(self._cov_inv, mu_k.T)}')
                b_k -= 0.5 * np.dot(mu_k,np.dot(self._cov_inv, mu_k.T).flatten())
                # print(f'Printed in likelihoods comp \n  a_k = {a_k}\n   b_k = {b_k}')
                likelihoods[i, class_value] = a_k + b_k
                # print("Printed in likelihood comp in LDA class::: ", likelihoods[i, class_value] )
        # print(f'likelihood matrix: {likelihoods[0:5,:]}')
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

