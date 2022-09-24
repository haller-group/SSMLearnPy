import logging


import numpy as np
from numpy.lib.arraysetops import isin
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

logger = logging.getLogger("ridge_regression")


def get_fit_ridge(
    X, 
    y,
    poly_degree: int=2,
    fit_intercept: bool=False,
    alpha: list=0,
    cv: int=2
):
    if(isinstance(X, list)):
        logger.info("Transforming data")
        X = get_matrix(X)
        y = get_matrix(y)
    
    logger.debug(f"X shape: {X.shape}, y shape: {y.shape}")

    if cv>=2 and isinstance(alpha, list):
        logger.info(f"CV={cv} on ridge regression")
        regressor = MultiOutputRegressor(
            estimator=RidgeCV(
                fit_intercept=False,
                alphas=alpha,
                cv=cv)
        )

    else:
        logger.info("Skipping CV on ridge regression")
        if isinstance(alpha, list):
            raise RuntimeError("Found alpha to be a list and cv to be <2.")
        regressor = MultiOutputRegressor(
            estimator=Ridge(
                fit_intercept=False,
                alpha=alpha)
        )

    mdl = Pipeline(
        [
            ('poly_transf', PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ('scaler', StandardScaler(with_mean=False)),
            ('ridge_regressor', regressor)
        ]
    )

    logger.info("Fitting regression model")
    mdl.fit(X.T, y.T)
    return mdl


def get_matrix(l:list):
    return np.concatenate(l, axis=1)