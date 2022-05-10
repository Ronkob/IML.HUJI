from __future__ import annotations
import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn
from ..metrics.loss_functions import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations

    self.D_: ndarray of shape (n_samples, )
        Distribution to measure samples
        
    self.weights_: List[float]
        Weight for each model


    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = [], [], None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # self.D_ = np.array([[1/X.shape[0]]*X.shape[0]]).flatten()  # uniform distribution for all iterations
        self.D_ = np.ones(X.shape[0])/X.shape[0]
        for iteration_number in range(self.iterations_):
            model = self.wl_()
            print(f'iteration {iteration_number} and first few D: {self.D_[:10]}, is null {np.isnan(self.D_).any()} and'
                  f' sum D is {self.D_.sum()}')
            # choosing a random sample with replacement from X in consideration for the current distribution
            t_sample_index = np.random.choice(np.arange(X.shape[0]), size=X.shape[0], replace=True,
                                              p=self.D_)

            predications = model.fit_predict(X, y*self.D_)
            self.models_.append(model)

            # calculating model weight
            # misclassification_indicator = np.abs(predications - y[t_sample_index])/2  # misclass = 0
            # epsilon_t = np.dot(self.D_, misclassification_indicator)  # should be a scalar
            epsilon_t = np.sum((np.abs(predications - y) / 2) * self.D_)
            print(f'iteration {iteration_number} and epsilon is {epsilon_t}')
            weight_t = 0.5 * np.log((1.0/epsilon_t)-1)
            print(f'iteration {iteration_number} and something is {weight_t*y*predications.shape}')
            self.weights_.append(weight_t)

            # calculating next samples distribution
            self.D_ *= np.exp((-1)*weight_t*y*predications)
            self.D_ /= np.sum(self.D_)



    def _predict(self, X):
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
        predictions = np.zeros(X.shape[0])
        for iteration_number in range(self.iterations_):
            predictions += self.models_[iteration_number].predict(X)*self.weights_[iteration_number]
        return np.sign(predictions)

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
        return misclassification_error(y, self._predict(X))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        predictions = np.zeros(X.shape[0])
        for iteration_number in range(T):
            predictions += self.models_[iteration_number].predict(X) * self.weights_[iteration_number]
        return np.sign(predictions)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.partial_predict(X, T))
