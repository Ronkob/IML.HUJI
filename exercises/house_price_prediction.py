from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
import sys


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv("../datasets/house_prices.csv").dropna().drop_duplicates()

    """
    preprocessing part
    
    handling dates:
     - need to figure out (date is the home sale date)
    
    data corrections:
    - Waterfront, View, Condition, Grade: change to nominal values
    - zipcode: make as arbitrary numbers
    
    new features:
    - house_age
    - renovation_flag
    
    missing data filling:
    - luckily, there are no missing data 
    
    anomalies:
    - exclude wierd houses with no bathrooms
    
    """
    # new features
    house_age = 2022 - full_data["yr_built"]
    renovation_flag = (full_data["yr_renovated"] != 0).astype(int)
    full_data['house_age'] = house_age
    full_data['renovation_flag'] = renovation_flag
    full_data['renovation_age'] = 2022 - full_data["yr_renovated"]

    # changing the neighbors relation
    # full_data['sqft_living15'] = np.abs(full_data['sqft_living15']-full_data['sqft_living'])
    # full_data['sqft_lot15'] = np.abs(full_data['sqft_lot15'] - full_data['sqft_lot'])

    # remove outlaw rows
    full_data = full_data[full_data.waterfront.isin([0, 1]) &
            full_data.view.isin(range(5)) &
            full_data.condition.isin(range(1, 6)) &
            full_data.grade.isin(range(1, 15))]

    # anomalies handling
    full_data = full_data[full_data.bathrooms > 0]
    full_data = full_data[full_data.sqft_lot < 1300000]
    full_data = full_data[full_data.sqft_lot15 < 50000]

    features_used = full_data[['bedrooms', 'bathrooms', 'sqft_living', 'zipcode',
                               'sqft_lot', 'sqft_living15', 'sqft_lot15',
                               'floors', 'waterfront', 'view', 'condition', 'grade',
                               'sqft_above', 'sqft_basement', 'house_age', 'renovation_flag', 'renovation_age']]

    # categorical variables handling
    features_used = pd.get_dummies(features_used, prefix='zipcode_', columns=['zipcode'])

    labels = full_data["price"]

    return features_used, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    def pearson_corr(feature1, feature2):
        fm1 = feature1.mean()
        fm2 = feature2.mean()

        pearson = ((feature1 - fm1) * (feature2 - fm2)).sum() / \
                  (np.sqrt(((feature1 - fm1) ** 2).sum()) * np.sqrt(((feature2 - fm2) ** 2).sum()))
        return pearson

    feature_list = ['bedrooms', 'bathrooms', 'sqft_living',
                    'sqft_lot', 'sqft_living15', 'sqft_lot15',
                    'floors', 'waterfront', 'view', 'condition', 'grade', 'lat', 'long',
                    'sqft_above', 'sqft_basement', 'house_age', 'renovation_flag', 'renovation_age']

    features_pearson = [pearson_corr(X[feature], y) for feature in feature_list]
    fig = px.scatter(x=feature_list, y=features_pearson)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, response = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(features, response)

    # Question 3 - Split samples into training- and testing sets.
    # TODO: implement my own split test train
    from sklearn.model_selection import train_test_split

    train_X, test_X, train_y, test_y = train_test_split(features, response)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    loss = np.zeros(shape=(91, 100))
    for p in range(10, 101):
        for i in range(100):
            model = LinearRegression()
            train_X_sample = np.asarray(train_X.sample(frac=(p / 100), random_state=i*p))
            train_y_sample = np.asarray(train_y.sample(frac=(p / 100), random_state=i*p))
            test_X_sample = np.asarray(test_X.sample(frac=(p / 100), random_state=i*p*2))
            test_y_sample = np.asarray(test_y.sample(frac=(p / 100), random_state=i*p*2))
            model.fit(train_X_sample, train_y_sample)
            run_loss = model._loss(test_X_sample, test_y_sample)
            loss[p - 10, i] = run_loss

    loss_mean = loss.mean(axis=1)
    loss_var = loss.std(axis=1)
    print('this is shape loss min ', loss_mean.shape, ' this is shpe y_true', loss_var.shape)
    x_axis_values = [p for p in range(10, 101)]

    fig_data = (go.Scatter(x=x_axis_values, y=loss_mean, mode="markers+lines", name="Mean Prediction",
                           line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
                go.Scatter(x=x_axis_values, y=loss_mean-2 * loss_var, fill=None, mode="lines",
                           line=dict(color="lightgrey"), showlegend=False),
                go.Scatter(x=x_axis_values, y=loss_mean+2 * loss_var, fill='tonexty', mode="lines",
                           line=dict(color="lightgrey"), showlegend=False),)

    fig = go.Figure()
    for f in fig_data:
        fig.add_trace(f)
    fig.show()
