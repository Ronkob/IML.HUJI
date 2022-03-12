from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    MU = 10
    VAR = 1
    first_sample = np.random.normal(MU, VAR, 1000)
    first_estimator = UnivariateGaussian()
    first_estimator.fit(first_sample)
    print(first_estimator.mu_, first_estimator.var_)

    # Question 2 - Empirically showing sample mean is consistent
    estimates = np.ndarray(100, dtype='object')
    for count, m in enumerate(np.arange(10, 1001, 10)):
        print(m)
        m_sample = np.random.normal(MU, VAR, m)
        estimator = UnivariateGaussian()
        estimator.fit(m_sample)
        estimates[count] = (m_sample, estimator.mu_, estimator.var_)

    mu_error = np.asarray([np.abs(mu - MU) for sample, mu, var in estimates])
    mu_error_normalized = (mu_error-min(mu_error))/(max(mu_error)-min(mu_error))
    samples = [sample for sample, mu, var in estimates]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Mean error vs Sample size', 'PDF of fitted model - sample size 1000'))

    fig.append_trace(go.Scatter(x=list(map(len, samples)), y=mu_error_normalized, mode='lines',
                                line={'color': 'black'}, name='sample mean error'), row=1, col=1)
    fig.update_xaxes(title_text="sample size", row=1, col=1)
    fig.update_yaxes(title_text="expectation error", row=1, col=1)

    # Question 3 - Plotting Empirical PDF of fitted model
    fig.append_trace(go.Scatter(y=first_estimator.pdf(first_sample), x=first_sample,
                                mode='markers', name='fitted PDF value', marker={'color': 'black'}), row=1, col=2)
    normal_pdf = lambda val, mu, var: (1 / np.sqrt(var * 2 * np.pi)) * \
                             pow(np.e, -0.5 * pow((val - mu) / np.sqrt(var), 2))
    fig.append_trace(go.Scatter(y=[normal_pdf(val, MU, VAR) for val in first_sample], x=first_sample,
                                mode='markers', name='original PDF value', marker={'color': 'green'}), row=1, col=2)
    fig.update_xaxes(title_text="sample value", row=1, col=2)
    fig.update_yaxes(title_text="PDF value - density", row=1, col=2)
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
