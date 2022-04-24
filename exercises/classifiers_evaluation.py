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
                        subplot_titles=('Perceptron Loss of Linearly Seperable data',
                                        'Perceptron Loss of Linearly Ineperable data',
                                        'Perceptron prediction of Linearly Seperable data',
                                        'Perceptron prediction of Linearly Inseperable data'))

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

        print(perceptron.coefs_)
        lim = np.array([data[0].min(axis=0), data[0].max(axis=0)]).T + np.array([-.5, .5])
        w = perceptron.coefs_[1:]
        yy = (-w[0] / w[1]) * lim[0] - (perceptron.coefs_[0] / w[1])
        # Plot figure of loss as function of fitting iteration
        fig.append_trace(go.Scatter(x=lim[0], y=[yy[0], yy[1]], mode='lines', line_color="black", showlegend=False),
                         row=2, col=i + 1)
        fig.update_xaxes(title_text="number of iterations", row=1)
        fig.update_yaxes(title_text="Misclassification Loss", row=1)
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

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    fig2 = make_subplots(rows=2, cols=3,
                         subplot_titles=(f'Gaussians1 dataset', f'Gaussians1 LDA predictions, accuracy:',
                                         "Gaussians1 Gaussian_Naive_Bayes predictions",
                                         f'Gaussians2 dataset', f'Gaussians2 LDA predictions, accuracy:',
                                         "Gaussians2 Gaussian_Naive_Bayes predictions"))

    for i, f in enumerate(["../datasets/gaussian1.npy", "../datasets/gaussian2.npy"]):
        # Load dataset
        data = load_dataset(f)

        # Fit models and predict over training set
        LDA_model = LDA()
        LDA_model.fit(data[0], data[1])
        LDA_predictions = LDA_model.predict(X=data[0])  # should be an m length vector of 1 and -1

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        fig2.append_trace(
            go.Scatter(x=data[0][:, 0], y=data[0][:, 1], mode='markers', marker=dict(size=10, opacity=0.6, color=data[1],
                       line=dict(color="black", width=1))), row=1+i, col= 1)

        fig2.append_trace(
            go.Scatter(x=data[0][:, 0], y=data[0][:, 1], mode='markers', marker=dict(size=10, opacity=0.6, color=LDA_predictions,
                       line=dict(color="black", width=1))), row=1+i, col= 2)
        LDA_accuracy = accuracy(data[1], LDA_predictions)
        fig2.layout.annotations[(3*i)+1].update(text=f'{fig2.layout.annotations[(3*i)+1].text}{LDA_accuracy}')



    fig2.show()
    # Add traces for data-points setting symbols and colors

    # Add `X` dots specifying fitted Gaussians' means

    # Add ellipses depicting the covariances of the fitted Gaussians


if __name__ == '__main__':
    sys.path.append('../datasets/')
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
