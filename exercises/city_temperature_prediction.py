import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
from typing import NoReturn

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv("../datasets/City_Temperature.csv", parse_dates=['Date']).dropna().drop_duplicates()

    # anomalies handling
    full_data = full_data[full_data.Temp > -20]  # there are no real measurements below -20

    # new features
    full_data['DayOfYear'] = full_data['Date'].dt.dayofyear

    return full_data


def q2(full_data) -> NoReturn:
    israel_data = full_data[full_data.Country == 'Israel']
    fig1 = px.scatter(israel_data, x='DayOfYear', y='Temp', color='Year',
                      labels={'DayOfYear': 'Day of Year', 'Temp': 'Temperature'},
                      title='Daily Average Temperature by Year')
    fig1.write_image("DayOfYearTempQ2.1.png")

    temp_month_std = israel_data.groupby('Month').Temp.agg(lambda col: np.std(col))
    fig2 = px.bar(temp_month_std, y='Temp', labels=({'Temp': 'Standard Deviation'}),
                  title='Standard Deviation of Average Daily Temperature, by Month')
    fig2.write_image("MonthTempStdQ2.2.png")
    return None


def q3(full_data) -> NoReturn:
    monthly = full_data.groupby(['Country', 'Month']).Temp.agg([np.average, np.std])
    x = monthly.index.get_level_values(1)
    color = monthly.index.get_level_values(0)
    fig3 = px.line(monthly, x=x, y='average', color=color, error_y='std',
                   labels={'x': 'Month Number', 'average': 'Avg Temperature'},
                   title='Monthly Temperature Average, Standard Deviation By Country')
    fig3.write_image("MonthAvgTempStdByCountryQ3.1.png")
    return None


def q4(full_data) -> int:
    israel_data = full_data[full_data.Country == 'Israel']
    features = israel_data['DayOfYear']
    response = israel_data['Temp']
    train_X, train_y, test_X, test_y = split_train_test(features, response)
    train_X, train_y, test_X, test_y = np.asarray(train_X).flatten(), np.asarray(train_y).flatten(), \
                                       np.asarray(test_X).flatten(), np.asarray(test_y).flatten()

    loss = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_X, train_y)
        loss.append(np.round(model.loss(test_X, test_y), decimals=2))
    results = list(enumerate(loss, start=1))
    print(results)
    fig4 = px.bar(x=np.asarray(results)[:, 0], y=np.asarray(results)[:, 1],
                  labels={'x': 'k Values', 'y': 'Loss Rate'}, title='Model Loss Rate by k-Value')
    fig4.write_image("BarLossRateByKValueQ4.1.png")
    best_k = np.argmin(np.asarray(results)[:, 1]) + 1
    return best_k


def q5(full_data, k) -> NoReturn:
    israel_data = full_data[full_data.Country == 'Israel']
    features = israel_data['DayOfYear']
    response = israel_data['Temp']
    train_X, train_y, _, _ = split_train_test(features, response)
    train_X, train_y = np.asarray(train_X).flatten(), np.asarray(train_y).flatten()
    model = PolynomialFitting(k)
    model.fit(train_X, train_y)

    loss = []
    for country in full_data.Country.unique():
        country_data = full_data[full_data.Country == country]
        loss.append([country, model.loss(country_data['DayOfYear'], country_data['Temp'])])
    loss = np.asarray(loss)
    fig5 = px.bar(x=loss[:, 0], y=loss[:, 1].astype(float), labels={'x': 'Country', 'y': 'Loss Rate'},
                  title='Loss Rate by Country - Model Fitted for Israel')
    fig5.write_image("LossByCountryQ5.1.png")
    return None


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    full_data = load_data('.../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    q2(full_data)

    # Question 3 - Exploring differences between countries
    q3(full_data)

    # Question 4 - Fitting model for different values of `k`
    best_k = q4(full_data)

    # Question 5 - Evaluating fitted model on different countries
    q5(full_data, best_k)
