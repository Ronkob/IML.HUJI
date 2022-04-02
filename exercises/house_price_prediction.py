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
    full_data = pd.read_csv("../datasets/house_prices.csv")

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
    full_data['yr_renovated'][full_data.yr_renovated == 0] = full_data['yr_built'][full_data.yr_renovated == 0]
    full_data['renovation_age'] = 2022 - full_data["yr_renovated"]

    # anomalies handling
    full_data = full_data[full_data.bathrooms > 0]

    # categorical variables handling
    """
    letting go of the zipcode because is correlated with the longitude/latitude
    """

    features = full_data[['bedrooms', 'bathrooms', 'sqft_living',
                          'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
                          'sqft_above', 'sqft_basement', 'house_age', 'renovation_flag', 'renovation_age']]
    labels = full_data["price"]
    return features, labels


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

        pearson = ((feature1-fm1)*(feature2-fm2)).sum() / \
            (np.sqrt(((feature1-fm1)**2).sum()) * np.sqrt(((feature2-fm2)**2).sum()))
        return pearson

    feature_list = ['bedrooms', 'bathrooms', 'sqft_living',
                    'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
                    'sqft_above', 'sqft_basement', 'house_age', 'renovation_flag', 'renovation_age']

    y = [pearson_corr(X[feature], y) for feature in feature_list]
    fig = px.scatter(x=feature_list, y=y)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, response = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(features, response)

    # Question 3 - Split samples into training- and testing sets.

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

