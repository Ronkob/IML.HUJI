from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

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


def q6(n_samples):
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    train_X, train_y, test_X, test_y = split_train_test(X, y, train_proportion=n_samples / X.shape[0])
    return np.asarray(train_X), np.asarray(train_y), np.asarray(test_X), np.asarray(test_y)


def q7(train_X, train_y, n_evaluations):
    ridge_losses = []
    lasso_losses = []
    lasso_lamdas = np.linspace(0, 2, n_evaluations)
    ridge_lamdas = np.linspace(0, 1, n_evaluations)

    for i in range(n_evaluations):
        lasso_model = Lasso(alpha=lasso_lamdas[i])
        avg_train_lasso, avg_test_lasso = cross_validate(lasso_model, train_X, train_y, scoring=mean_square_error)
        lasso_losses.append((avg_train_lasso, avg_test_lasso))
        ridge_model = RidgeRegression(lam=ridge_lamdas[i])
        avg_train_ridge, avg_test_ridge = cross_validate(ridge_model, train_X, train_y, scoring=mean_square_error)
        ridge_losses.append((avg_train_ridge, avg_test_ridge))

    ridge_losses = np.asarray(ridge_losses)
    lasso_losses = np.asarray(lasso_losses)

    fig = go.Figure(data=[go.Scatter(x=lasso_lamdas, y=lasso_losses[:, 0],
                                     name='lasso train avg loss', mode='markers'),
                          go.Scatter(x=lasso_lamdas, y=lasso_losses[:, 1],
                                     name='lasso test avg loss', mode='markers'),
                          go.Scatter(x=ridge_lamdas, y=ridge_losses[:, 0],
                                     name='ridge train avg loss', mode='markers'),
                          go.Scatter(x=ridge_lamdas, y=ridge_losses[:, 1],
                                     name='ridge test avg loss', mode='markers')],
                    layout=dict(
                        title=rf"$\text{{(7) train and validation errors along different regularization terms | }}$"))
    fig.add_annotation(xref='paper', yref='paper', x=0, y=0.95, align='left',
                       text=f'Best regularization paramater for Ridge is {ridge_lamdas[np.argmin(ridge_losses[:, 1])]}\n'
                            f'Best regularization paramater for Lasso is {lasso_lamdas[np.argmin(lasso_losses[:, 1])]}',
                       font=dict(size=16), showarrow=False)
    fig.show()

    return ridge_lamdas[np.argmin(ridge_losses[:, 1])], lasso_lamdas[np.argmin(lasso_losses[:, 1])]


def q8(lamda_ridge, lamda_lasso, train_X, train_y, test_X, test_y):
    ls_model = LinearRegression().fit(train_X, train_y)
    ridge_model = RidgeRegression(lam=lamda_ridge).fit(train_X, train_y)
    lasso_model = Lasso(alpha=lamda_lasso).fit(train_X, train_y)

    ls_model_predictions = (ls_model.predict(test_X))
    ridge_model_predictions = (ridge_model.predict(test_X))
    lasso_model_predictions = (lasso_model.predict(test_X))

    ls_model_loss = mean_square_error(ls_model_predictions, test_y)
    ridge_model_loss = mean_square_error(ridge_model_predictions, test_y)
    lasso_model_loss = mean_square_error(lasso_model_predictions, test_y)

    print('ls_model_loss is: ' + str(ls_model_loss))
    print('ridge_model_loss is: ' + str(ridge_model_loss))
    print('lasso_model_loss is: ' + str(lasso_model_loss))


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
    train_X, train_y, test_X, test_y = q6(n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lamda_ridge, lamda_lasso = q7(train_X, train_y, n_evaluations)

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    q8(lamda_ridge, lamda_lasso, train_X, train_y, test_X, test_y)


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree(n_samples=100, noise=5)
    # select_polynomial_degree(n_samples=100, noise=0)
    # select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter(50, 500)
