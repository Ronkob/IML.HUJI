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
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def q1(train_X, train_y, test_X, test_y, n_learners, function_calls=[1]):
    model = AdaBoost(DecisionStump, n_learners)
    predictions = model.fit_predict(train_X, train_y).astype(int)
    # plotting the model's prediction on the generated data
    go.Figure(data=[go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                               marker=dict(size=10, opacity=0.9, color=train_y, colorscale=class_colors(3),
                                           symbol=class_symbols[1 + predictions]))],
              layout=dict(title="Model predictions")).show()

    # plotting the test and train errors to the num_learners
    training_error = []
    test_error = []
    for t in range(1, n_learners):
        training_error.append(model.partial_loss(train_X, train_y, t))
        test_error.append(model.partial_loss(test_X, test_y, t))
    go.Figure(data=[go.Scatter(x=list(range(1, n_learners)), y=training_error, mode='markers+lines', name=f'train error'),
                    go.Scatter(x=list(range(1, n_learners)), y=test_error, mode='markers+lines', name=f'test error')],
              layout=dict(title=f'({function_calls[0]}.1) Model error rates',
                          title_x=0.5, title_font_size=20, margin=dict(t=120))).show()
    function_calls[0] += 1
    idx = np.argmin(np.asarray(test_error))
    return model, idx, test_error[idx]


def q2(model, T, lims, X, y, function_calls=[1]):
    iterations = T
    fig2 = make_subplots(rows=2, cols=2, horizontal_spacing=0.1,
                         subplot_titles=(f'{iterations[0]} training iterations',
                                         f'{iterations[1]} training iterations',
                                         f'{iterations[2]} training iterations',
                                         f'{iterations[3]} training iterations'))
    boundarys = [decision_surface(lambda X: model.partial_predict(X, t), lims[0], lims[1], showscale=False)
                 for t in iterations]
    for i in range(4):
        fig2.append_trace(boundarys[i], row=int(i/2)+1, col=i % 2+1)
        fig2.append_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                          marker=dict(size=10, opacity=0.9, color=y, colorscale=class_colors(3)), showlegend=False),
                          row=int(i/2)+1, col=(i % 2)+1)
    fig2.layout.update(title_text=f'({function_calls[0]}.2) Decision Boundary learned from different number of training iterations',
                       title_x=0.5, title_font_size=20, margin=dict(t=120))
    fig2.show()
    function_calls[0] += 1


def q3(model, best_t, loss, lims, X, y, function_calls=[1]):
    boundary = decision_surface(lambda X: model.partial_predict(X, best_t), lims[0], lims[1], showscale=False)
    go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                          marker=dict(size=10, opacity=0.9, color=y, colorscale=class_colors(3)), showlegend=False),
                    boundary],
              layout=dict(title=f'({function_calls[0]}.3) Best ensemble size model predictions'
                                f' (Ensemble size {best_t}, accuracy {1-loss}) ',
                          title_x=0.5, title_font_size=20, margin=dict(t=120))).show()
    function_calls[0] += 1


def q4(model, n_learners, lims, X, y, function_calls=[1]):
    D = model.D_
    D = (D/np.max(D)) * 20
    boundary = decision_surface(lambda x: model.partial_predict(x, n_learners), lims[0], lims[1], showscale=False)
    go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(size=D, opacity=0.9, color=y,
                                                                                 colorscale=class_colors(3))),
                    boundary],
              layout=dict(title=f'({function_calls[0]}.4) Training data in proportion to their final weights',
                          title_x=0.5, title_font_size=20, margin=dict(t=120))).show()
    function_calls[0] += 1


def fit_and_evaluate_adaboost(noise: float, n_learners=250, train_size=5000, test_size=500, to_answer=np.arange(4)+1):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    if 1 in to_answer:
        model, best_t, loss = q1(train_X, train_y, test_X, test_y, n_learners)
        print("this best t: ", best_t)
        T = [5, 50, 100, 250]
        lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    if 2 in to_answer:
        assert 1 in to_answer
        # Question 2: Plotting decision surfaces
        q2(model, T, lims, test_X, test_y)

    if 3 in to_answer:
        assert 1 in to_answer
        # Question 3: Decision surface of best performing ensemble
        q3(model, best_t, loss, lims, test_X, test_y)

    if 4 in to_answer:
        assert 1 in to_answer
        # Question 4: Decision surface with weighted samples
        q4(model, n_learners, lims, train_X, train_y)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0.4)

