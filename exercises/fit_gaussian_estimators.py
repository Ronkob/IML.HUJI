from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "simple_white"

fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Mean error vs Sample size', 'PDF of fitted model - sample size 1000'))

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
        m_sample = np.random.normal(MU, VAR, m)
        estimator = UnivariateGaussian()
        estimator.fit(m_sample)
        estimates[count] = (m_sample, estimator.mu_, estimator.var_)

    mu_error = np.asarray([np.abs(mu - MU) for sample, mu, var in estimates])
    mu_error_normalized = (mu_error - min(mu_error)) / (max(mu_error) - min(mu_error))
    samples = [sample for sample, mu, var in estimates]

    fig.append_trace(go.Scatter(x=list(map(len, samples)), y=mu_error, mode='lines+markers',
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
    MU_MULTY = np.asarray([0, 0, 4, 0])

    SIGMA = np.asarray([[1, 0.2, 0, 0.5],
                        [0.2, 2, 0, 0],
                        [0, 0, 1, 0],
                        [0.5, 0, 0, 1]])

    second_sample = np.random.multivariate_normal(MU_MULTY, SIGMA, 1000)
    second_estimator = MultivariateGaussian()
    second_estimator.fit(second_sample)
    print(second_estimator.mu_, '\n', second_estimator.cov_)

    # Question 5 - Likelihood evaluation
    DIM = 200
    likelihood_matrix = np.zeros((DIM, DIM))
    f_matrix = np.asarray([[(f1, f3) for f3 in np.linspace(-10, 10, DIM)] for f1 in np.linspace(-10, 10, DIM)])

    for i, f1 in enumerate(np.linspace(-10, 10, DIM)):
        for j, f3 in enumerate(np.linspace(-10, 10, DIM)):
            likelihood_matrix[i, j] = \
                second_estimator.log_likelihood(np.array([f1, 0, f3, 0]), SIGMA, second_sample)

    fig2 = go.Figure(data=go.Heatmap(y=list(np.linspace(-10, 10, DIM)), x=list(np.linspace(-10, 10, DIM)),
                                     z=likelihood_matrix, colorscale='inferno'))
    fig2.update_layout(
        title='Log-Likelihood as a function of f1,f3',

    )

    fig2.update_xaxes(
        title_text='f1'
    )
    fig2.update_yaxes(
        title_text='f3'
    )
    fig2.show()


    # Question 6 - Maximum likelihood
    f_cords = (np.unravel_index(np.argmax(likelihood_matrix), likelihood_matrix.shape))
    print(np.round(f_matrix[f_cords], decimals=3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
