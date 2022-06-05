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
    assert 1 == 1

    shuffle_idx = np.arange(X.shape[0])
    np.random.shuffle(shuffle_idx)
    X_shuffled = X[shuffle_idx]
    y_shuffled = y[shuffle_idx]
    train_loss = []
    test_loss = []

    for i in range(cv):
        print(len(X_shuffled[0:int(i * X.shape[0] / cv)]))
        train_X = np.append(X_shuffled[0:int(i * X.shape[0] / cv)], X_shuffled[int((i + 1) * X.shape[0] / cv):])
        train_y = np.append(y_shuffled[0:int(i * y.shape[0] / cv)], y_shuffled[int((i + 1) * y.shape[0] / cv):])
        test_X = X_shuffled[int(i * X.shape[0] / cv):int((i + 1) * X.shape[0] / cv)].flatten()
        test_y = y_shuffled[int(i * y.shape[0] / cv):int((i + 1) * y.shape[0] / cv)].flatten()
        estimator.fit(train_X, train_y)
        train_predictions = estimator.predict(train_X)
        test_predictions = estimator.predict(test_X)
        train_loss.append(scoring(train_predictions, train_y))
        test_loss.append(scoring(test_predictions, test_y))

    return np.mean(train_loss), np.mean(test_loss)
