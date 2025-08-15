import logging


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.gaussian_process.kernels import Kernel

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from ssmlearnpy.utils.preprocessing import get_matrix, generate_exponents, PolynomialFeaturesWithPattern
from ssmlearnpy.geometry.dimensionality_reduction import LinearChart
from ssmlearnpy.utils.preprocessing import complex_polynomial_features, generate_exponents, compute_polynomial_map, unpack_coefficient_matrices_from_vector
from typing import NamedTuple, Union
from scipy.optimize import minimize, least_squares, lsq_linear


logger = logging.getLogger("ridge_regression")



def get_fit_ridge(
    X, 
    y,
    constraints: list = None,
    do_scaling: bool = True,
    poly_degree: int=2,
    rbf_kernel: Kernel = None,
    kernel_grid_size: int = 20,
    fit_intercept: bool=False,
    alpha: list=0,
    cv: int=2
):
    """Fit a ridge regression model to the data. 
    Parameters:
    ----------
        X: (n_features, n_samples) or list 
        y: (n_outputs, n_samples) or list 
        constraints: list of lists: [LHS, RHS] such that model.predict(LHS[i]) == RHS[i].
                 model.predict(LHS[i]) and RHS[i] should have the same shape
        do_scaling: bool, whether to apply a StandardScaler to the data before fitting
        poly_degree: int, degree of the polynomial to fit
        rbf_kernel: sklearn kernel object, optional. If not None, the polynomial features will be combined with the kernel features
        kernel_grid_size: int, number of points to use for the kernel grid per dimension
        fit_intercept: bool, whether to include the constant term in the regression
                        if False, this means that the model will be forced to pass through the origin
        alpha: float or list of floats, regularization parameter
        cv: int, number of folds for cross validation. If cv>=2, alpha must be a list
    Returns:
    -------
        mdl: sklearn Pipeline object containing a PolynomialFeatures,
                 an optional StandardScaler, and Ridge regression 
    """    
    if isinstance(X, list):
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
        
    if rbf_kernel is not None:
        logger.info(f"Using kernel {str(rbf_kernel)} and polynomial features degree {poly_degree}")

        # Create an equidistant grid of points close to the original features X
        grid_range = 0.1 * (X.max(axis=1) - X.min(axis=1))  # Define a range around the original features
        grid_points = [np.linspace(X[i].min() - grid_range[i], X[i].max() + grid_range[i], num=kernel_grid_size) for i in range(X.shape[0])]
        grid = np.array(np.meshgrid(*grid_points)).reshape(X.shape[0], -1)

        Y = grid.T  # Use the grid points as Y
        feature_transf = FeatureUnion([
            ('poly_transf', PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ('kernel_transf', FunctionTransformer(rbf_kernel, kw_args={'Y': Y}) )
        ])  # Combine features from polynomial and rbf kernel
    else:
        logger.info(f"Using polynomial features degree {poly_degree}")
        feature_transf = FeatureUnion([
            ('poly_transf', PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ])

    if do_scaling:# default is to include a standard scaler
        mdl = Pipeline(
            [
                ('feature_transf', feature_transf),
                ('scaler', StandardScaler(with_mean=False)),
                ('ridge_regressor', regressor)
            ]
        )
    else:
        mdl = Pipeline(
            [
                ('feature_transf', feature_transf),
                ('ridge_regressor', regressor)
            ]
        )
    # explicitly set sample weights to 1 in case we have constraints
    sample_weight = np.ones(X.shape[1])
    
    logger.info("Fitting regression model")
    if constraints is not None:
        # if we have constraints, add them to X and y with a large weight
        logger.info("Adding constraints to regression model")
        X, y, sample_weight = add_constraints(X, y, constraints, sample_weight, weight=1e10)

    # mdl.predict() expects the matrix in the form (n_samples, n_features)
    mdl.fit(X.T, y.T, ridge_regressor__sample_weight = sample_weight)
    
    mdl.map_info = {}
    map_coefs = np.zeros(mdl.named_steps.ridge_regressor.coef_.shape)

    if do_scaling:
        scaler_coefs = mdl.named_steps.scaler.scale_
    else:
        scaler_coefs = np.ones(map_coefs.shape[1])
    
    map_coefs = mdl.named_steps.ridge_regressor.coef_ / scaler_coefs
    mdl.map_info['coefficients'] = map_coefs
    mdl.map_info['exponents'] = mdl.named_steps.feature_transf['poly_transf'].powers_
    return mdl


def get_fit_ridge_parametric(
    X, 
    y,
    parameters,
    origin_remains_fixed: bool = True,
    constraints: list = None,
    do_scaling: bool = True,
    poly_degree: int=2,
    poly_degree_parameter: int=2,
    fit_intercept: bool=False,
    alpha: list=0,
    cv: int=2
):
    """Fit a ridge regression model to the data, with parameter-dependent coefficients
    Here X and y must be lists of trajectories.
    Parameters:
    ----------
        X: (n_features, n_samples) or list 
        y: (n_outputs, n_samples) or list 
        parameters: list of parameters for each trajectory. 
                    Each element should look like (n_parameters, n_samples) or (n_parameters,1). 
        origin_remains_fixed: bool. If True, then the regression will not contain terms that depend only on the parameter. 
                                otherwise the parameter is simply treated as an additional feature. 
        constraints: list of lists: [LHS, RHS] such that model.predict(LHS[i]) == RHS[i].
                 model.predict(LHS[i]) and RHS[i] should have the same shape. 
                 As a result, the last entries in LHS[i] should refer to the parameters.
        do_scaling: bool, whether to apply a StandardScaler to the data before fitting
        poly_degree: int, degree (with respect to the state) of the polynomial to fit
        poly_degree_parameter: int, degree of the polynomial to fit for the parameters. 
                            In general, poly_degree_parameter != poly_degree, but it should be at most poly_degree.
        fit_intercept: bool, whether to include the constant term in the regression
                        if False, this means that the model will be forced to pass through the origin
        alpha: float or list of floats, regularization parameter
        cv: int, number of folds for cross validation. If cv>=2, alpha must be a list
    Returns:
    -------
        mdl: sklearn Pipeline object containing a PolynomialFeaturesWithPattern,
                 an optional StandardScaler, and Ridge regression. 
    """   
    if isinstance(X, list):
        logger.info("Transforming data")
        if parameters[0].shape[1] == X[1].shape[1]: # we have a parameter for each sample
            X = get_matrix(X)
            y = get_matrix(y)
            parameters = get_matrix(parameters)
        else:
            parameters = [np.tile(p, X[i].shape[1]) for i,p in enumerate(parameters)]
            parameters = get_matrix(parameters)
            X = get_matrix(X)
            y = get_matrix(y)
        X_and_params = np.append(X, parameters, axis = 0)
    else: 
        raise NotImplementedError("X and y must be lists of trajectories")
    logger.debug(f"X shape: {X.shape}, y shape: {y.shape}, parameters shape: {parameters.shape}")
    n_features = X.shape[0]
    n_params = parameters.shape[0]

    if cv>=2 and isinstance(alpha, list):
        raise NotImplementedError("CV not implemented for parametric regression")
    regressor = Ridge(
            fit_intercept=fit_intercept,
            alpha=alpha
            )
    
    # decide which features to include in the regression: 
    exponents = generate_exponents(n_features+n_params, degree = poly_degree, include_bias=False).T # transpose back to the PolyFeatures format
    structure = True * exponents[:,0]

    if origin_remains_fixed:
        # the parameters were inserted at the end of the vector 
        if n_features == 1:
            contains_state_dependent = exponents[:,0]
        else:
            contains_state_dependent = np.logical_or(*[exponents[:,i] for i in range(n_features)]) # if any of the state features has a nonzero exponent ~ logical_or

        degree_of_parameter = np.array([exponents[:,i] for i in range(n_features, n_features+n_params)])[0,:]
        if n_params > 1:
            degree_of_parameter = np.sum(at_most_degree_param, axis = 1)
        at_most_degree_param = degree_of_parameter <= poly_degree_parameter # if the sum of the exponents of the parameters is at most poly_degree_parameter
        structure = np.logical_and(contains_state_dependent, at_most_degree_param)
        logger.debug(f"Number of features to include: {np.sum(structure)}")
    if do_scaling: # default is to include a standard scaler
        mdl = Pipeline(
            [
                ('poly_transf', PolynomialFeaturesWithPattern(degree=poly_degree, include_bias=False, structure=structure)),
                ('scaler', StandardScaler(with_mean=False)),
                ('ridge_regressor', regressor)
            ]
        )
    else:
        mdl = Pipeline(
            [
                ('poly_transf', PolynomialFeaturesWithPattern(degree=poly_degree, include_bias=False, structure=structure)),
                ('ridge_regressor', regressor)
            ]
        )

    sample_weight = np.ones(X_and_params.shape[1])

    if constraints is not None:
        # if we have constraints, add them to X and y with a large weight
        logger.info("Adding constraints to regression model")
        X_and_params, y, sample_weight = add_constraints(X_and_params, y, constraints, sample_weight, weight=1e10)

    mdl.fit(X_and_params.T, y.T, ridge_regressor__sample_weight = sample_weight)

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
    penalty_linear_cons = 1e-5,
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

    if poly_degree == 1:
        # then simply do an svd
        encoder = LinearChart(n_dim)
        encoder.fit(X)
        decoder = Decoder(
            lambda x : np.matmul(encoder.matrix_representation, x.T).T, # callable function
            {'coefficients': encoder.matrix_representation.T, 'exponents': np.eye(n_dim)}
            )
        return encoder, decoder

    if initial_guess is None:
        # compute initial guess: projection matrix from svd
        # and nonlinear coefficients from ridge regression
        initial_guess = generate_initial_guess(X, n_dim, poly_degree, fit_intercept, alpha, cv)
        logger.debug(f"Computing inintial guess: {initial_guess.shape}")
    # initial guess should be a vector
    def compute_error(z):
        # rearrange z into coefficient arrays:
        linear_coefs, nonlinear_coefs = unpack_coefficient_matrices_from_vector(z, n_linear_coefs, n_features, n_targets)
        X_reduced = np.matmul(linear_coefs.T, X) # projection
        Error_linear = X - np.matmul(linear_coefs, X_reduced) # linear prediction
        nonlinear_features = complex_polynomial_features(X_reduced.T, degree = poly_degree, skip_linear = True) # have to exclude the linear features
        Error = Error_linear - np.matmul(nonlinear_coefs, nonlinear_features.T)
        Error = Error.ravel()
        if alpha is not None:  # ridge regularization
            Error = np.concatenate((Error, np.sqrt(alpha) *z))
        # add constraints as penalty terms:
        constraint_linear = (np.matmul(linear_coefs.T, linear_coefs) - np.eye(n_features)).ravel()
        constraint_nonlinear = np.matmul(linear_coefs.T, nonlinear_coefs).ravel()
        Error = np.concatenate((Error, np.sqrt(penalty_linear_cons) * constraint_linear, np.sqrt(penalty_nonlinear_cons) * constraint_nonlinear))
        return  Error / X.shape[1]
    # These are implemented as soft-constraints. 
    # TODO: implement as hard constraints (e.g. with SLSQP)
    # TODO: Slower than matlab. Possibly precalculating the Jacobian will help. 
    if optimize_kwargs is None:
        optimize_kwargs = {'method': 'lm', 'maxfev': 1e5, 'ftol': 1e-6, 'gtol':1e-6, 'verbose': 0}
    res = least_squares(compute_error, initial_guess, **optimize_kwargs) # we can use the special sturcutre of the loss and use scipy.optimize.least_suqares()
    logger.debug(f"Optimization terminated. Success: {res.success}")
    optimal_linear_coef, optimal_nonlinear_coef = unpack_coefficient_matrices_from_vector(res.x, n_linear_coefs, n_features, n_targets)
    encoder = LinearChart(
        n_dim,
        matrix_representation = optimal_linear_coef
        )
    joint_coefs = np.concatenate((optimal_linear_coef, optimal_nonlinear_coef), axis=1) 
    powers = generate_exponents(n_features, poly_degree)
    decoder = Decoder(
        lambda x : compute_polynomial_map(joint_coefs,
                                           poly_degree)(x.T).T, # have to transpose once more to be consistent with the ridge regression object
                                                            # that expects x to be (n_samples, n_features). 
                                                            # Compute_polynomial_map expects (n_features, n_samples)
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

class Decoder(NamedTuple): # to behave in the same way as the pipeline object generated by get_fit_ridge
    predict : callable
    map_info : dict
    fit = None


def add_constraints(
        X,
        y,
        constraints,
        sample_weight,
        weight=1e10
):
    const_lhs, const_rhs = constraints
    for l, r in zip(const_lhs, const_rhs):
        
        lhs = np.array(l).reshape(-1,1)
        rhs = np.array(r).reshape(-1,1)
        X = np.append(X, lhs, axis = 1)
        y = np.append(y, rhs, axis = 1)
    sample_weight = np.ones(X.shape[1])
    sample_weight[-len(const_lhs):] = weight
    return X, y, sample_weight



def fit_inverse(
        function,
        trajectories_source,
        degree, 
        trajectories_target = None,
        near_identity = True,
        ):
    """
    Using linear regression, we identify the inverse of the given function,
    i. e., on the given trajectory data
    inverse(function (trajectories_source) ) = trajectories_target.
    trajectories_source and trajectories_target are lists of (n_features, n_samples) arrays 
    and can be different. E. g. they are related by a linear transformation. M @ trajectories_source = trajectories_target
    Parameters:
    -----------
    function: callable
        The function to be inverted. Should take a (n_features, n_samples) array
            as input and return a (n_targets, n_samples) array.
    trajectories_source: list of (n_features, n_samples) arrays
    trajectories_target: list of (n_targets, n_samples) arrays. Defaults to None. 
                        in this case trajectories_source is used for both source and target
    degree: int, maximal degree of the polynomial approximation
    near_identity: bool, if True, the linear part of the inverse is fixed to be the identity 

    Returns:
    --------
    coefficients: (n_features, n_targets) array
    inverse: callable
        The inverse of the given function. Takes a (n_targets, n_samples) array
        as input and returns a (n_features, n_samples) array.
    """
    if trajectories_target is None:
        trajectories_target = trajectories_source
    image = [function(t) for t in trajectories_source]
    if ~np.imag(image[0]).any(): # if this is real valued
        mdl = get_fit_ridge(image, trajectories_target, do_scaling = False, poly_degree = degree, alpha = 0)
        return mdl.map_info['coefficients'], mdl.predict
    else:
        ndofs = int(trajectories_target[0].shape[0] / 2) # we assume that the second half of the features are complex conjugates of the first
        nonlinear_features = [complex_polynomial_features(y.T, degree).T for y in image]
        # CX= Y where C is the coefficient matrix,  X is the nonlinear features of image, Y is the target
        X = get_matrix(nonlinear_features)
        Y = get_matrix([t[:ndofs, :]  for t in trajectories_target])
        X = np.conjugate(X).T
        Y = np.conjugate(Y).T
        coeffs = []
        # solves Ax = b  -> x^T A^T = b^T
        XX = np.matmul(np.conjugate(X).T, X)
        XY = np.matmul(np.conjugate(X).T, Y)
        XXinv = np.linalg.inv(XX)
        result = np.matmul(XXinv, XY) # unconstrained solution
        if near_identity: # constraining the linear part
            # https://en.wikipedia.org/wiki/Ordinary_least_squares#Constrained_estimation
            # Q ~ constraints_lhs.T
            constraint_lhs = np.zeros((2*ndofs, XX.shape[1]))
            constraint_rhs = np.zeros((2*ndofs, XY.shape[1]))
            for j in range(2*ndofs):
                constraint_lhs[j, j] = 1.
            for i in range(ndofs):
                constraint_rhs[i*ndofs] = 1.
            matrix_middle = np.conjugate(constraint_lhs).T @ np.linalg.inv(constraint_lhs @ XXinv @ np.conjugate(constraint_lhs).T)
            difference = constraint_lhs @ result - constraint_rhs # zero if the constraint is satisfied
            result = result - XXinv @ matrix_middle @ difference
            #difference = constraint_lhs @ result - constraint_rhs
        callable = compute_polynomial_map(np.conjugate(result).T, degree)
    return np.conjugate(result).T, callable




