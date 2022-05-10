import numpy as np
from typing import Tuple
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners.adaboost import AdaBoost
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    #y[0.2*X[:, 0]+X[:, 1] > 0.2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def q1(train_X, train_y, test_X, test_y) -> AdaBoost:
    model = AdaBoost(DecisionStump, 250)
    predictions = model.fit_predict(train_X, train_y).astype(int)
    # plotting the model's prediction on the generated data
    go.Figure(go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                         marker=dict(size=10, opacity=0.9, color=train_y, colorscale=class_colors(3),
                                     symbol=class_symbols[1 + predictions]))).show()
    # plotting the
    training_error = []
    test_error = []
    for t in range(1,250):
        training_error.append(model.partial_loss(train_X, train_y, t))
        test_error.append(model.partial_loss(test_X, test_y, t))
    go.Figure(data=[go.Scatter(x=list(range(250)), y=training_error, mode='markers+lines', name=f'train error'),
              go.Scatter(x=list(range(250)), y=test_error, mode='markers+lines', name=f'test error')]).show()

    return model

def q2(model, iterations, lims):
    pass


def fit_and_evaluate_adaboost(noise: float, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = q1(train_X, train_y, test_X, test_y)

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    q2(model, T, lims)

    # Question 3: Decision surface of best performing ensemble
    raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0.4)
