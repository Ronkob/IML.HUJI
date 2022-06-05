import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    assert (y_pred.shape == y_true.shape), "Y_pred and Y_true are not from the same shape"
    n = len(y_true)
    res = y_pred-y_true
    return np.mean(res**2)


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    y_true = np.asarray(y_true).reshape(-1, 1)

    assert (y_true.shape == y_pred.shape)
    b = y_true - y_pred
    misclass_index = np.nonzero(y_true - y_pred)[0]
    if len(y_true) > 0 and len(y_pred) > 0:
        if not normalize:
            return len(misclass_index)
        else:
            a = len(misclass_index)/len(y_pred)
            return a
    else:
        return 0

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    return 1-misclassification_error(y_true, y_pred)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()
