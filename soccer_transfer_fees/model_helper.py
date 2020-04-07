import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, lars_path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm


def model_compare(X, y, ridge_a, lasso_a):
    """Fits linear regression, ridge, and lasso models to the input data and prints out the
    R^2, RMSE, MAE, and coefficients for all of them.

    Uses sklearn and statsmodels.

    inputs:
        X: Feature data

        y: Target data

        ridge_a: alpha coefficient for ridge model

        lasso_a: alpha coefficient for lasso model

    returns:
        None
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=12)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_tr_sc = scaler.transform(X_train)
    X_te_sc = scaler.transform(X_test)

    lm_ridge = Ridge(alpha=ridge_a)
    lm_lasso = Lasso(alpha=lasso_a)
    lm = LinearRegression()

    lm.fit(X_train, y_train)
    lm_ridge.fit(X_tr_sc, y_train)
    lm_lasso.fit(X_tr_sc, y_train)

    lm_pred = lm.predict(X_test)
    ridge_pred = lm_ridge.predict(X_te_sc)
    lasso_pred = lm_lasso.predict(X_te_sc)

    print('\n----- Linear Regression -----')
    print(f'R2: {r2_score(y_test, lm_pred):.3f}; RMSE: {np.sqrt(mean_squared_error(y_test, lm_pred)):.2f}; MAE: {mean_absolute_error(y_test, lm_pred):.2f}\n')
    print(pd.Series(dict(zip(X_test.columns, lm.coef_))))

    print('\n----- Ridge Regression -----')
    print(f'R2: {r2_score(y_test, ridge_pred):.3f}; RMSE: {np.sqrt(mean_squared_error(y_test, ridge_pred)):.2f}; MAE: {mean_absolute_error(y_test, ridge_pred):.2f}\n')
    print(pd.Series(dict(zip(X_test.columns, lm_ridge.coef_))))

    print('\n----- Lasso Regression -----')
    print(f'R2: {r2_score(y_test, lasso_pred):.3f}; RMSE: {np.sqrt(mean_squared_error(y_test, lasso_pred)):.2f}; MAE: {mean_absolute_error(y_test, lasso_pred):.2f}\n')
    print(pd.Series(dict(zip(X_test.columns, lm_lasso.coef_))))

    lsm = sm.OLS(y_train, sm.add_constant(X_train))
    fit = lsm.fit()
    print(fit.summary())


def lars_plot(X, y):
    """Computes and plots the LARS path for the given data

    inputs:
        X: Feature data

        y: target_data

    returns:
        None
    """

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=12)

    # Scale the variables
    std = StandardScaler()
    std.fit(X_train.values)

    X_tr = std.transform(X_train.values)

    # Note: lars_path takes numpy matrices, not pandas dataframes

    print("Computing regularization path using the LARS ...")
    alphas, _, coefs = lars_path(X_tr, y_train.values, method='lasso')

    # plotting the LARS path

    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]

    plt.figure(figsize=(15, 15))
    plt.plot(xx, coefs.T)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed')
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title('LASSO Path')
    plt.axis('tight')
    plt.legend(X_train.columns)
    plt.show()


def eval_plots(actual, pred):
    """Produces Q-Q, actual vs. predicted, residual vs. predicted, and residual histogram plots.

    inputs:
        actual: target data

        pred: predicted data

    returns:
        None
    """

    res = actual - pred

    sns.set(font_scale=1.5)

    # Q-Q Plot
    plt.figure(figsize=(12, 12))
    stats.probplot(res, dist='norm', plot=plt)

    # Actual vs. Predicted
    plt.figure(figsize=(12, 12))
    plt.scatter(pred, actual, alpha=0.8)
    plt.plot(np.linspace(0, max([max(actual), max(pred)]), 100), np.linspace(
        0, max([max(actual), max(pred)]), 100), 'r')
    plt.xlabel('Predicted (million £)')
    plt.ylabel('Actual (million £)')
    plt.title('Data vs. Model Transfer Fee')
#     plt.savefig('data_vs_model_plot.svg');

    # Residual vs. Predicted
    plt.figure(figsize=(12, 12))
    plt.scatter(pred, res)
    plt.xlabel('Prediction (million £)', fontsize=20)
    plt.ylabel('Residual', fontsize=20)
    plt.title('Residual vs. Predicted Transfer Fee')
#     plt.savefig('residual_vs_pred.svg');

    # Residual Histogram
    plt.figure(figsize=(12, 8))
    plt.hist(res)
    plt.xlabel('Residual')
    plt.ylabel('Count')
