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
    constraints: list = None,
    poly_degree: int=2,
    fit_intercept: bool=False,
    alpha: list=0,
    cv: int=2
):
    """Fit a ridge regression model to the data. 
        X: (n_features, n_samples) or list 
        y: (n_outputs, n_samples) or list 
        constraints: list of lists: [LHS, RHS] such that model.predict(LHS[i]) == RHS[i]. model.predict(LHS[i]) and RHS[i] should have the same shape
    """    
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
    # explicitly set sample weights to 1 in case we have constraints
    sample_weight = np.ones(X.shape[1])
        
    logger.info("Fitting regression model")
    if constraints is not None:
        # TODO: should do a cleaner implementation
        # if we have constraints, add them to X and y with a large weight
        constLHS, constRHS = constraints
        for i in range(len(constLHS)):
            lhs = np.array(constLHS[i]).reshape(-1,1)
            rhs = np.array(constRHS[i]).reshape(-1,1)
            X = np.append(X, lhs, axis = 1)
            y = np.append(y, rhs, axis = 1)
        sample_weight = np.ones(X.shape[1])
        sample_weight[-len(constLHS):] = 1e10
        

    mdl.fit(X.T, y.T, ridge_regressor__sample_weight = sample_weight)

    mdl.map_info = {}
    scaler_coefs = mdl.named_steps.scaler.scale_
    estimators = mdl.named_steps.ridge_regressor.estimators_
    map_coefs = np.zeros((len(estimators), len(scaler_coefs)))
    for iRow in range(len(estimators)):
        map_coefs[iRow,:] = estimators[iRow].coef_ / scaler_coefs
    mdl.map_info['coefficients'] = map_coefs
    mdl.map_info['exponents'] = mdl.named_steps.poly_transf.powers_

    return mdl


def get_matrix(l:list):
    return np.concatenate(l, axis=1)