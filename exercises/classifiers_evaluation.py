import numpy as np
import sys

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from typing import NoReturn
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Loss of Linearly Separable data',
                                        'Loss of Linearly Inseparable data',
                                        'prediction of Linearly Separable data',
                                        'prediction of Linearly Inseparable data'))

    for i, (n, f) in enumerate([("Linearly Separable", "../datasets/linearly_separable.npy"),
                                ("Linearly Inseparable", "../datasets/linearly_inseparable.npy")]):
        # Load dataset
        data = load_dataset(f)
        fig.append_trace(go.Scatter(x=data[0][:, 0], y=data[0][:, 1], mode='markers', name=f'{n} Perceptron prediction',
                                    marker=dict(size=10, color=data[1], line=dict(color="black", width=1))), row=2,
                         col=i + 1)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback_loss(fit: Perceptron, x: np.ndarray, y: int) -> NoReturn:
            losses.append(fit.loss(data[0], data[1]))

        perceptron = Perceptron(callback=callback_loss)
        perceptron.fit(data[0], data[1])

        # Plot figure of loss as function of fitting iteration
        fig.append_trace(go.Scatter(x=np.arange(len(losses)), y=losses, mode='lines+markers', line={'color': 'black'},
                                    name=n), row=1, col=i + 1)

        # Plot figure of the classification rule on the data
        lim = np.array([data[0].min(axis=0), data[0].max(axis=0)]).T + np.array([-.5, .5])
        w = perceptron.coefs_[1:]
        yy = (-w[0] / w[1]) * lim[0] - (perceptron.coefs_[0] / w[1])

        fig.append_trace(go.Scatter(x=lim[0], y=[yy[0], yy[1]], mode='lines', line_color="black", showlegend=False),
                         row=2, col=i + 1)
        fig.update_xaxes(title_text="number of iterations", row=1)
        fig.update_yaxes(title_text="Misclassification Loss", row=1)
        fig.layout.update(title_text=f'Perceptron Classifier', title_x=0.5, title_font_size=20, margin=dict(t=120))
    fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, showlegend=False, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    # Create subplots

    for i, f in enumerate(["../datasets/gaussian1.npy", "../datasets/gaussian2.npy"]):
        # Load dataset
        data = load_dataset(f)
        fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                             subplot_titles=("LDA Predictions, Accuracy:",
                                             "Gaussian Naive Bayes Predictions, Accuracy:"))
        model_names = ["LDA", "Gaussian Naive Bayes"]
        for j, model in enumerate([LDA(), GaussianNaiveBayes()]):
            # Fit models and predict over training set
            model.fit(data[0], data[1])
            predictions = model.predict(X=data[0])  # should be an m length vector of classes

            # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA
            # predictions on the right. Plot title should specify dataset used and subplot titles should specify
            # algorithm and accuracy

            # plotting the correct data
            # fig2.append_trace(go.Scatter(x=data[0][:, 0], y=data[0][:, 1], mode='markers',
            #                              marker=dict(size=10, opacity=0.9, color=data[1],
            #                                          symbol=class_symbols[1],
            #                                          line=dict(color="black", width=1))), row=1, col=3)

            from IMLearn.metrics import accuracy
            lims = np.array([data[0].min(axis=0), data[0].max(axis=0)]).T + np.array([-.2, .2])
            fig2.append_trace(go.Scatter(x=data[0][:, 0], y=data[0][:, 1], mode='markers',
                                         text=predictions, name =f'{model_names[j]} predictions',
                                         marker=dict(size=10, opacity=0.9, color=predictions, colorscale=class_colors(3),
                                                     symbol=class_symbols[1 * (np.asarray(predictions == data[1]))],
                                                     line=dict(color="black", width=1))), row=1, col=j + 1)
            fig2.append_trace(decision_surface(model.predict, lims[0], lims[1], showscale=False), row=1, col=j + 1)
            model_accuracy = accuracy(data[1], predictions)
            fig2.layout.annotations[j].update(text=f'{fig2.layout.annotations[j].text}{model_accuracy}')

            # Add `X` dots specifying fitted Gaussians' means
            fig2.append_trace(
                go.Scatter(x=model.mu_[:, 0], y=model.mu_[:, 1], mode='markers', showlegend=False,
                           marker=dict(size=30, opacity=0.9, color='black', symbol=class_symbols[1])), row=1, col=j + 1)

            # Add ellipses depicting the covariances of the fitted Gaussians
            if j == 0:
                for class_number in model.classes_:
                    fig2.append_trace(
                        get_ellipse(model.mu_[class_number], model.cov_), row=1, col=j + 1)
            if j == 1:
                for class_number in model.classes_:
                    fig2.append_trace(
                        get_ellipse(model.mu_[class_number], np.diagflat(model.vars_[class_number])), row=1, col=j + 1)

        fig2.layout.update(title_text=f'Gaussians {i + 1} Dataset', title_x=0.5, title_font_size=20, margin=dict(t=120))
        fig2.show()


if __name__ == '__main__':
    sys.path.append('../datasets/')
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
