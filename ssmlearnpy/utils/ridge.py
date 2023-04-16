import logging


import numpy as np
from numpy.lib.arraysetops import isin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from ssmlearnpy.utils.preprocessing import get_matrix
from ssmlearnpy.geometry.dimensionality_reduction import LinearChart
from ssmlearnpy.utils.preprocessing import complex_polynomial_features, generate_exponents, compute_polynomial_map
from typing import NamedTuple
from scipy.optimize import minimize, least_squares

logger = logging.getLogger("ridge_regression")


def get_fit_ridge(
    X, 
    y,
    constraints: list = None,
    do_scaling: bool = True,
    poly_degree: int=2,
    fit_intercept: bool=False,
    alpha: list=0,
    cv: int=2
):
    """Fit a ridge regression model to the data. 
    Parameters:
        X: (n_features, n_samples) or list 
        y: (n_outputs, n_samples) or list 
        constraints: list of lists: [LHS, RHS] such that model.predict(LHS[i]) == RHS[i].
                 model.predict(LHS[i]) and RHS[i] should have the same shape
        do_scaling: bool, whether to apply a StandardScaler to the data before fitting
        poly_degree: int, degree of the polynomial to fit
        fit_intercept: bool, whether to include the constant term in the regression
                        if False, this means that the model will be forced to pass through the origin
        alpha: float or list of floats, regularization parameter
        cv: int, number of folds for cross validation. If cv>=2, alpha must be a list
    Returns:
        mdl: sklearn Pipeline object containing a PolynomialFeatures,
                 an optional StandardScaler, and Ridge regression 
    """    
    if(isinstance(X, list)):
        logger.info("Transforming data")
        X = get_matrix(X)
        y = get_matrix(y)
    
    logger.debug(f"X shape: {X.shape}, y shape: {y.shape}")

    if cv>=2 and isinstance(alpha, list):
        logger.info(f"CV={cv} on ridge regression")
        regressor = RidgeCV(
                fit_intercept=fit_intercept,
                alphas=alpha,
                cv=cv
                )

    else:
        logger.info("Skipping CV on ridge regression")
        if isinstance(alpha, list):
            raise RuntimeError("Found alpha to be a list and cv to be <2.")
        regressor = Ridge(
                fit_intercept=fit_intercept,
                alpha=alpha
                )
    if do_scaling: # default is to include a standard scaler
        mdl = Pipeline(
            [
                ('poly_transf', PolynomialFeatures(degree=poly_degree, include_bias=False)),
                ('scaler', StandardScaler(with_mean=False)),
                ('ridge_regressor', regressor)
            ]
        )
    else:
        mdl = Pipeline(
            [
                ('poly_transf', PolynomialFeatures(degree=poly_degree, include_bias=False)),
                ('ridge_regressor', regressor)
            ]
        )
    # explicitly set sample weights to 1 in case we have constraints
    sample_weight = np.ones(X.shape[1])
        
    logger.info("Fitting regression model")
    if constraints is not None:
        # if we have constraints, add them to X and y with a large weight
        constLHS, constRHS = constraints
        for l, r in zip(constLHS, constRHS):
            lhs = np.array(l).reshape(-1,1)
            rhs = np.array(r).reshape(-1,1)
            X = np.append(X, lhs, axis = 1)
            y = np.append(y, rhs, axis = 1)
        sample_weight = np.ones(X.shape[1])
        sample_weight[-len(constLHS):] = 1e10

    mdl.fit(X.T, y.T, ridge_regressor__sample_weight = sample_weight)
    mdl.map_info = {}
    map_coefs = np.zeros(mdl.named_steps.ridge_regressor.coef_.shape)

    if do_scaling:
        scaler_coefs = mdl.named_steps.scaler.scale_
    else:
        scaler_coefs = np.ones(map_coefs.shape[1])
    
    map_coefs = mdl.named_steps.ridge_regressor.coef_ / scaler_coefs
    mdl.map_info['coefficients'] = map_coefs
    mdl.map_info['exponents'] = mdl.named_steps.poly_transf.powers_
    return mdl




def fit_reduced_coords_and_parametrization(
    X, 
    n_dim: int,
    poly_degree: int=2,
    fit_intercept: bool=False,
    alpha: list=0,
    cv: int=2,
    initial_guess = None,
    penalty_linear_cons =1e-5,
    penalty_nonlinear_cons =1e-5,
    **optimize_kwargs
):
    """
        X: (n_features, n_samples) or list
    """
    if isinstance(X, list):
        logger.info("Transforming data")
        X = get_matrix(X)    
    logger.debug(f"X shape: {X.shape}")
    n_targets = X.shape[0]
    n_features = n_dim
    n_linear_coefs = n_targets * n_features # n_samples * n_features

    if initial_guess is None:
        # compute initial guess: projection matrix from svd
        # and nonlinear coefficients from ridge regression
        initial_guess = generate_initial_guess(X, n_dim, poly_degree, fit_intercept, alpha, cv)
        logger.debug(f"Computing inintial guess: {initial_guess.shape}")
    # initial guess should be a vector
    def compute_error(z):
        # rearrange z into coefficient arrays:
        linear_coefs, nonlinear_coefs = unpack_linear_nonlinear_coefficients(z, n_linear_coefs, n_features, n_targets)
        X_reduced = np.matmul(linear_coefs.T, X) # projection
        Error_linear = X - np.matmul(linear_coefs, X_reduced) # linear prediction

        nonlinear_features = complex_polynomial_features(X_reduced, degree = poly_degree, skip_linear = True) # have to exclude the linear features
        Error = Error_linear - np.matmul(nonlinear_coefs, nonlinear_features)
        #Error = np.linalg.norm(Error)**2
        Error = Error.ravel()
        if alpha is not None:  # ridge regularization
            Error = np.concatenate((Error, np.sqrt(alpha) *z))
        # add constraints as penalty terms:
        constraint_linear = (linear_coefs.T @ linear_coefs - np.eye(n_features)).ravel()
        constraint_nonlinear = (linear_coefs.T @ nonlinear_coefs).ravel()
        Error = np.concatenate((Error, np.sqrt(penalty_linear_cons) * constraint_linear, np.sqrt(penalty_nonlinear_cons) * constraint_nonlinear))
        return  Error / X.shape[1]# + multiplier1 * constraint1 + multiplier2 * constraint2
    # These are implemented as soft-constraints. 
    # TODO: implement as hard constraints (e.g. with SLSQP)
    # TODO: Much slower than matlab. Possibly precalculating the Jacobian will help. 
    if optimize_kwargs is None:
        optimize_kwargs = {'method': 'lm', 'maxfev': 1e5, 'ftol': 1e-6, 'gtol':1e-6, 'verbose': 0}
    res = least_squares(compute_error, initial_guess, **optimize_kwargs) # we can use the special sturcutre of the loss and use scipy.optimize.least_suqares()
    logger.debug(f"Optimization terminated. Success: {res.success}")
    optimal_linear_coef, optimal_nonlinear_coef = unpack_linear_nonlinear_coefficients(res.x, n_linear_coefs, n_features, n_targets)
    encoder = LinearChart(
        n_dim,
        matrix_representation = optimal_linear_coef
        )
    joint_coefs = np.concatenate((optimal_linear_coef, optimal_nonlinear_coef), axis=1) 
    powers = generate_exponents(n_features, poly_degree)

    decoder = Decoder(
        compute_polynomial_map(joint_coefs, poly_degree), # callable function
        {'coefficients': joint_coefs, 'exponents': powers}
        )
    return encoder, decoder


def generate_initial_guess(
        X,
        n_dim,
        poly_degree,
        fit_intercept,
        alpha,
        cv
):

    Lc = LinearChart(n_dim)
    Lc.fit(X)
    X_reduced = Lc.predict(X)
    initial_model = get_fit_ridge(X_reduced, X, do_scaling=False, poly_degree=poly_degree, fit_intercept=fit_intercept, alpha=alpha, cv=cv)
    # discard the linear part of the ridge model:
    ridge_nonlinear_coeffs = initial_model.map_info['coefficients'][:, n_dim:]
    initial_guess = np.concatenate([ Lc.matrix_representation.ravel(), ridge_nonlinear_coeffs.ravel()]) 
    return initial_guess

def unpack_linear_nonlinear_coefficients(
    z,
    n_linear_coefs,
    n_features,
    n_targets
):
    linear_coefs = z[:n_linear_coefs].reshape(n_targets, n_features)
    n_nonlinear_coefs = z.shape[0] - n_linear_coefs
    n_nonlinear_features = int(n_nonlinear_coefs/n_targets)
    nonlinear_coefs = z[n_linear_coefs:].reshape(n_targets, n_nonlinear_features)
    return linear_coefs, nonlinear_coefs

class Decoder(NamedTuple): # to behave in the same way as the pipeline object generated by get_fit_ridge
    predict : callable
    map_info : dict
    fit = None
