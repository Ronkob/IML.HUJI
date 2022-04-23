import sys

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from typing import NoReturn
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
from sklearn.linear_model import Perceptron as SK_perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

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

    fig = make_subplots(rows=1, cols=2)

    for n, f in [("Linearly Separable", "../datasets/linearly_inseparable.npy")]:
        # Load dataset
        data = load_dataset("C:/Users/PC/Documents/ACADEMY/4th Semester/IML/IML.HUJI/datasets/linearly_inseparable.npy")


        model = LDA()
        model.fit(data[0], data[1])
        print(model.coefs_)
        lim = np.array([data[0].min(axis=0), data[0].max(axis=0)]).T + np.array([-.5, .5])
        w = model.coefs_[0]
        yy = (-w[0] / w[1]) * lim[0] - (model.intercept_[0] / w[1])
        # # Plot figure of loss as function of fitting iteration
        fig.append_trace(go.Scatter(x=data[0][:,0], y=data[0][:,1], mode='markers',
                            marker=dict(size=10, color=data[1], line=dict(color="black", width=1))), row=1, col=1)
        fig.append_trace(go.Scatter(x=lim[0], y=[yy[0], yy[1]], mode='lines', line_color="black", showlegend=False),row=1,col=1)
    fig.show()


if __name__ == '__main__':
    sys.path.append('../datasets/')
    np.random.seed(0)
    run_perceptron()