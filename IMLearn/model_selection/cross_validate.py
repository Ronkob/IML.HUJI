from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    assert X.shape[0] == y.shape[0]

    train_loss = []
    test_loss = []

    X_splited = np.asarray(np.array_split(X, cv))
    y_splited = np.asarray(np.array_split(y, cv))

    for fold in range(cv):
        b = np.array([True if i != fold else False for i in range(cv)])
        train_X = np.concatenate(X_splited[b])
        test_X = np.concatenate(X_splited[~b])
        train_y = np.concatenate(y_splited[b])
        test_y = np.concatenate(y_splited[~b])

        estimator.fit(train_X, train_y)
        train_predictions = estimator.predict(train_X)
        test_predictions = estimator.predict(test_X)

        train_loss.append(scoring(train_predictions, train_y))
        test_loss.append(scoring(test_predictions, test_y))

    return np.mean(np.asarray(train_loss)), np.mean(np.asarray(test_loss))

