from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
# from sklearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def q1(n_samples, noise):
    X = np.linspace(-3, 2, n_samples)
    epsilons = np.random.normal(0, noise, n_samples)
    y = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    y_noisy = y(X) + epsilons
    data, response = pd.DataFrame(X), pd.Series(y_noisy)
    train_X, train_y, test_X, test_y = split_train_test(data, response, train_proportion=0.67)

    train_trace = go.Scatter(x=train_X.loc[:, 0], y=y(train_X.loc[:, 0]), mode='markers', name='Train samples',
                             marker=dict(size=10, opacity=0.9, color=1, colorscale=class_colors(3)))
    test_trace = go.Scatter(x=test_X.loc[:, 0], y=y(test_X.loc[:, 0]), mode='markers', name='Test samples',
                            marker=dict(size=10, opacity=0.9, color=2, colorscale=class_colors(3)))
    fig = go.Figure(data=[train_trace, test_trace],
                    layout=dict(title=rf"$\text{{(1) True Model | noise: {noise}, n_samples: {n_samples} }}$",
                                xaxis=dict(ticklen=5, zeroline=False)))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

    fig.show()

    return train_X, train_y, test_X, test_y


def q2(train_X, train_y, n_samples, noise):
    losses = []
    from sklearn.model_selection import cross_val_score
    for k in range(0, 11):
        model = PolynomialFitting(k)
        avg_train, avg_test = cross_validate(model, train_X, train_y, scoring=mean_square_error)
        losses.append((avg_train, avg_test))
    losses = np.asarray(losses)
    fig = go.Figure(data=[go.Scatter(x=list(range(0, 11)), y=losses[:, 0], name='train avg loss'),
                          go.Scatter(x=list(range(0, 11)), y=losses[:, 1], name='test avg loss')],
                    layout=dict(
                        title=rf"$\text{{(2) Avg train and validation errors | "
                              rf"noise: {noise}, n_samples: {n_samples}}}$"))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

    fig.show()

    return losses[:, 1].argmin(), losses[:, 1].min(), losses


def q3(k, previous_loss, train_X, train_y, test_X, test_y, n_samples, noise):
    model = PolynomialFitting(k).fit(train_X, train_y)
    loss = mean_square_error(test_y, model.predict(test_X))
    predictions = model.predict(np.append(train_X, test_X))

    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    fig = go.Figure(data=[go.Scatter(x=np.append(train_X, test_X), y=predictions,
                                     name='predictions', mode='markers'),
                          go.Scatter(x=np.append(train_X, test_X), y=f(np.append(train_X, test_X)),
                                     name='test avg loss', mode='markers')],
                    layout=dict(title=rf"$\text{{(3) best degree - ({k}) polynomial fit | noise: "
                                      rf"{noise}, n_samples: {n_samples} }}$"))
    fig.add_annotation(xref='paper', yref='paper', x=0, y=0.95, align='left',
                       text=f'best validation loss: {np.round(previous_loss, 2)} - '
                            f'test error: {np.round(loss)}',
                       font=dict(size=16), showarrow=False)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.show()


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    train_X, train_y, test_X, test_y = q1(n_samples=n_samples, noise=noise)
    train_X, train_y, test_X, test_y = np.asarray(train_X).flatten(), np.asarray(train_y), np.asarray(
        test_X).flatten(), np.asarray(test_y)
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    best_k, best_loss, losses = q2(train_X, train_y, n_samples, noise)

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    q3(best_k, best_loss, train_X, train_y, test_X, test_y, noise, n_samples)


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree(n_samples=100, noise=5)
    # select_polynomial_degree(n_samples=100, noise=0)
    # select_polynomial_degree(n_samples=1500, noise=10)
