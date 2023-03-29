
from sklearn.preprocessing import PolynomialFeatures
from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate
import numpy as np
from ssmlearnpy.utils.preprocessing import complex_polynomial_features



class NonlinearCoordinateTransform():
    """
    General class to transform to perform a nonlinear change of coordinates
    If the original coordinates are denoted z, we realize a change of coordinates to y = T(z)
    and z = T^{-1}(y). 
    The transformation T and its inverse are polynomial maps of a specified degree.
    No constant term is included, but the linear part is peresent.

    The tranformation is given as a PolyinomialFeatures object multiplied by a vector of coefficients
    If the data is given as a matrix Z of shape (n_features, n_samples)
    PolynomialFeatures expects the data to be of shape (n_samples, n_features) so we transpose one time
    The polynomial features are collected in a matrix X of shape (n_samples, n_poly_features)
    The coefficients of the transformation have to then be organized in a matrix of shape (n_poly_features, n_features)

    """
    def __init__(self, dimension, degree, transform_coefficients = None, inverse_transform_coefficients = None):
        # can initialize with the coefficients of the transformation or as an empty object. in that case use the set_transform_coefficients method
        self.transform_coefficients = transform_coefficients
        self.inverse_transform_coefficients = inverse_transform_coefficients
        self.dimension = dimension
        self.degree = degree
        self.transform_map = None
        self.inverse_transform_map = None
        if self.transform_coefficients is not None:
            self.transform_map = self._compute_transform_map(self.transform_coefficients)
        if self.inverse_transform_coefficients is not None:
            self.inverse_transform_map = self._compute_transform_map(self.inverse_transform_coefficients)
    
    def _compute_transform_map(self, transform_coefficients):
        """
        Compute the polynomial map corresponding to the transformation coefficients
        """
        #features = PolynomialFeatures(degree=self.degree, include_bias=False)
        #complex_polynomial_features
        # return a function to be fitted and used in the transform method
        #print(transform_coefficients)
 #       cc = 0*transform_coefficients
#        cc[:2,:2]=transform_coefficients[:2,:2]
        return lambda x : transform_coefficients@complex_polynomial_features(x,degree=self.degree) # note the transpose because PolynomialFeatures expects the data to be of shape (n_samples, n_features)
    

    def set_transform_coefficients(self, transform_coefficients):
        self.transform_coefficients = transform_coefficients
        self.transform_map = self._compute_transform_map(self.transform_coefficients)
    def set_inverse_transform_coefficients(self, inverse_transform_coefficients):
        self.inverse_transform_coefficients = inverse_transform_coefficients
        self.inverse_transform_map = self._compute_transform_map(self.inverse_transform_coefficients)




    def transform(self, z):
        """
        Transform the coordinates z to y
        """
        if self.transform_coefficients is None:
            raise RuntimeError("The transformation coefficients have not been set")
        return self.transform_map(z) # transpose again to get the output in the right shape


    def inverse_transform(self, z):
        """
        Perform the inverse transformation from the coordinates y to z
        """
        if self.inverse_transform_coefficients is None:
            raise RuntimeError("The inverse transformation coefficients have not been set")
        return self.inverse_transform_map(z).T
    
        

class NormalForm:
    """
    General class to transform to normal form

    Parameters:
        - Linear part (matrix): linear part of the dynamics 
        - tolerance: tolerance for the resonance condition
        
    """
    def __init__(self, LinearPart, tolerance = None):
        self.LinearPart = LinearPart
        self.tolerance = tolerance # A good candidate for the tolerance is 10*min(Re(\lambda))
        if self.tolerance is None:
            eigenvalues, _ = np.linalg.eig(self.LinearPart) # eigenvalues are sorted
            self.tolerance = 10*np.min(np.abs(np.real(eigenvalues)))
        self.resonances = None
        self.diagonalized = False
        self.dynamics_structure = None
        self.transformation_structure = None
        
    def _nonlinear_coeffs(self, degree = 3):
        """
        Computes the coefficients of the nonlinear part of the vector field. For more info see M. Cenedese (2021) 
        """
        dimensions = self.LinearPart.shape[0] # number of dimensions of the manifold ~ number of eigenvalues
        # generate a dummy set of polynomial features to read off the coefficient matrix:
        # TODO: not too elegant
        poly = PolynomialFeatures(degree=degree, include_bias=False).fit(np.ones( (1, dimensions) ))
        exponents_of_nonlinear_map = poly.powers_.T # gives a matrix of shape (dimensions, number of monomials of degree <= degree)
        exponents_of_nonlinear_map = exponents_of_nonlinear_map[: , dimensions:]
        return exponents_of_nonlinear_map
    


    def _eigenvalue_lin_combinations(self, degree = 3):
        """
        Construct the matrix made up of linear combinations of eigenvalues. For more info see e. g. M. Cenedese (2021)
        """
        eigenvalues, _ = np.linalg.eig(self.LinearPart) # eigenvalues are sorted
        eigenvalues_diag = np.diag(eigenvalues)
        coeffs_for_exponents = self._nonlinear_coeffs(degree)
        return np.repeat(np.diag(eigenvalues_diag).reshape(-1,1), coeffs_for_exponents.shape[1], axis = 1) - np.matmul(np.diag(eigenvalues_diag), coeffs_for_exponents)
    
    def _resonance_condition(self, degree = 3):
        """
        Check if the eigenvalue linear combinations satisfy the resonance condition
        """
        lincombs = self._eigenvalue_lin_combinations(degree)
        return np.abs(lincombs) < self.tolerance # return a boolean array

    def set_dynamics_and_transformation_structure(self, type='flow', degree = 3):
        """
        For transforming to normal form we need to minimize
        ||d\dt (y + f_normalform(y)) - \Lambda*(y+f_normalform(y)) - N_normalform(y+f_normalform)||^2
        where \Lambda is the linear part of the dynamics and N_normalform is the nonlinear part of the normal form dynamics
        f_normalform is the nonlinear part of the normal form transformation z = T^{-1}(y)
        based on the linear part we enforce the structure of the nonlinearities in N_normalform and f_normalform.
        Parameters:
            times: list of times
            trajectories: list of trajectories (each of shape (n_features, n_samples))
            type: 'flow' or 'map'
            degree: max. degree of the polynomial transformation

        Returns:
            dydt - Lambda*y: (list of (n_features, n_samples))
        """
        
        if type != 'flow':
            raise NotImplementedError("Only flow is implemented")

        #if self.diagonalized:
        size = self.LinearPart.shape[0] 
        ndofs = int(size / 2) # only consider the first half of the coordinates: conjugates are discarded 
        N_structure = self._resonance_condition(degree)[:ndofs, :] # sparse structure for the nonlinearities
        self.dynamics_structure = np.sum(N_structure, axis = 0).astype(dtype=bool) # if there is a 1 in the structure matrix, we keep the corresponding column ~ logical or
        self.transformation_structure = np.logical_not(self.dynamics_structure) # complement
        return 



def prepare_normalform_transform_optimization(times, trajectories, LinearPart, type = 'flow', degree = 3):
    """
    Precompute terms for the optimization problem
    Parameters:
        times: list of times
        trajectories: list of trajectories (each of shape (n_features, n_samples))
        LinearPart: linear part of the dynamics
        type: 'flow' or 'map'
        degree: max. degree of the polynomial transformation
    Returns:
        linear error
        Dynamics_normalform_polynomial_features: list of non-zero polynomial features of the trajectories included in the normal form dynamics
        Transform_normalform_polynomial_features: list of non-zero polynomial features of the trajectories included in the normal form transformation
        Transform_normalform_polynomial_features_timederivative: list of non-zero polynomial features of the trajectories included in the normal form transformation's time derivative
    """

    size = LinearPart.shape[0] 
    ndofs = int(size / 2) # only consider the first half of the coordinates: conjugates are discarded 
    if type != 'flow':
        raise NotImplementedError("Only flow is implemented")

    normalform = NormalForm(LinearPart)
    normalform.set_dynamics_and_transformation_structure( type, degree)
    linear_error = []
    _, dy_dt = shift_or_differentiate(trajectories, times, type = 'flow') # only flow is implemented
    for i, y in enumerate(trajectories):
        # compute the error in the linear part
        linear_error.append(dy_dt[i] - LinearPart.dot(y)) 

    # precomute nonlinear terms in the transformation map
    Transformation_normalform_polynomial_features = []
    for y in trajectories:
        Transformation_normalform_polynomial_features.append(complex_polynomial_features(y, degree = degree, skip_linear = True, structure = normalform.transformation_structure))
    # compute the derivative of the transformed coordinates
    _, Transformation_normalform_polynomial_features_timederivative = shift_or_differentiate(Transformation_normalform_polynomial_features, times, type = 'flow')
    # also remove the unnecessary coordinates from the linear error 
    return normalform, [l[:ndofs, :] for l in linear_error], Transformation_normalform_polynomial_features, Transformation_normalform_polynomial_features_timederivative



def create_normalform_transform_objective(times, trajectories, Linearpart, type = 'flow', degree = 3):
    """
    Minimize
    ||d\dt (y + f_normalform(y)) - \Lambda*(y+f_normalform(y)) - N_normalform(y+f_normalform)||^2
    where \Lambda is the linear part of the dynamics and N_normalform is the nonlinear part of the normal form dynamics
    f_normalform is the nonlinear part of the normal form transformation z = T^{-1}(y)
    based on the linear part we enforce the structure of the nonlinearities in N_normalform and f_normalform.
    Parameters:
        times: list of times
        trajectories: list of trajectories (each of shape (n_features, n_samples))
        LinearPart: linear part of the dynamics
        type: 'flow' or 'map'
        degree: max. degree of the polynomial transformation
    Returns:
        normalform transformation
        normalform dynamics
    """
    
    normalform, linear_error, Transformation_normalform_polynomial_features, Transformation_normalform_polynomial_features_timederivative = prepare_normalform_transform_optimization(times, trajectories, Linearpart, type, degree)
    # set up optimization
    size = Linearpart.shape[0] 
    ndofs = int(size / 2) # only consider the first half of the coordinates: conjugates are discarded 
    # set up optimization
    n_unknowns_dynamics = np.sum(normalform.dynamics_structure)
    n_unknowns_transformation = Transformation_normalform_polynomial_features[0].shape[0]
    def objective(x): # x is purely real
        # convert to complex form. First half is real, second half is imaginary
        n_unknowns = int( len(x) / 2)
        
        #print(n_unknowns)
        x_complex = x[:n_unknowns] + 1j*x[n_unknowns:]
        # separate these into matrices of coefficients: 
        coeff_dynamics = x_complex[:n_unknowns_dynamics].reshape(ndofs, int(n_unknowns_dynamics/ndofs)).reshape((ndofs, int(n_unknowns_dynamics/ndofs)))
        coeff_transformation = x_complex[n_unknowns_dynamics:n_unknowns_dynamics + n_unknowns_transformation].reshape((ndofs, int(n_unknowns_transformation/ndofs)))
        nonlinear_error1 = []
        nonlinear_error2 = []
        derivative_error = []
        for i, y in enumerate(trajectories):
            transformed = y[:ndofs] + coeff_transformation@Transformation_normalform_polynomial_features[i]
            transformed_conj = np.repeat(transformed, 2, axis = 0)  # need to add the complex conjugate to evaluate the transformed coords
            transformed_conj[ndofs:, :] = np.conj(transformed[:ndofs, :])
            #  compute N_normalform(y+f_normalform)
            temp = complex_polynomial_features(transformed_conj, degree = degree, skip_linear = True, structure = normalform.dynamics_structure)
            nonlinear_error1.append(coeff_dynamics@temp)
            # compute Lambda*(f_normalform)
            nonlinear_error2.append( normalform.LinearPart[:ndofs, :ndofs]@coeff_transformation@Transformation_normalform_polynomial_features[i])
            # compute d/dt (f_normalform)
            derivative_error.append(coeff_transformation@Transformation_normalform_polynomial_features_timederivative[i])
        # the total error is: linear_error + d/dt (f_normalforn) - Lambda*(f_normalform) - N_normalform(y + f_normalform)
        error_along_trajs = [np.linalg.norm((l + d - n1 - n2))**2 for l, d, n1, n2 in zip(linear_error, derivative_error, nonlinear_error1, nonlinear_error2)]
        return np.sum(error_along_trajs) # sum over all trajectories. Could normalize with the number of trajectories
    return normalform, n_unknowns_dynamics, n_unknowns_transformation, objective

def unpack_optimized_coeffs(optimal_x, ndofs, normalform, n_unknowns_dynamics, n_unknowns_transformation):
    """
    Unpack the optimized coefficients into a dictionary
    Parameters:
        optimal_x: optimized coefficients
        ndofs: number of degrees of freedom
        normalform: normalform object to get the structure of the nonzero coefficients
        n_unknowns_dynamics: number of unknowns in the dynamics
        n_unknowns_transformation: number of unknowns in the transformation
    Returns:
        dictionary with optimized coefficients
    """
    n_unknowns = int( len(optimal_x) / 2)
    x_complex = optimal_x[:n_unknowns] + 1j*optimal_x[n_unknowns:]
    # separate these into matrices of coefficients: 
    coeff_dynamics = x_complex[:n_unknowns_dynamics].reshape(ndofs, int(n_unknowns_dynamics/ndofs)).reshape((ndofs, int(n_unknowns_dynamics/ndofs)))
    coeff_transformation = x_complex[n_unknowns_dynamics:n_unknowns_dynamics + n_unknowns_transformation].reshape((ndofs, int(n_unknowns_transformation/ndofs)))

    # insert the zeros into the coefficient matrices, which is dictated by normalform.dynamics_structure and normalform.transformation_structure
    coeff_dynamics = insert_zeros(coeff_dynamics, normalform.dynamics_structure)
    coeff_transformation = insert_zeros(coeff_transformation, normalform.transformation_structure)

    return {'coeff_dynamics': coeff_dynamics, 'coeff_transformation': coeff_transformation}


# def create_normalform_transofmration(times, trajectories, Linearpart, type = 'flow', degree = 3):
#     """
#     Minimize
#     ||d\dt (y + f_normalform(y)) - \Lambda*(y+f_normalform(y)) - N_normalform(y+f_normalform)||^2
#     where \Lambda is the linear part of the dynamics and N_normalform is the nonlinear part of the normal form dynamics
#     f_normalform is the nonlinear part of the normal form transformation z = T^{-1}(y)
#     based on the linear part we enforce the structure of the nonlinearities in N_normalform and f_normalform.
#     Parameters:
#         times: list of times
#         trajectories: list of trajectories (each of shape (n_features, n_samples))
#         LinearPart: linear part of the dynamics
#         type: 'flow' or 'map'
#         degree: max. degree of the polynomial transformation
#     Returns:
#         normalform transformation
#         normalform dynamics
#     """
#     n_unknowns_dynamics, n_unknowns_transformation, objective = create_normalform_transform_objective(times, trajectories, Linearpart, type, degree)
#     ndofs = int(Linearpart.shape[0] / 2) # only consider the first half of the coordinates: conjugates are discarded 
#     # set up optimization
#     x0 = np.random.rand(n_unknowns_dynamics + n_unknowns_transformation) # initial guess
#     # optimize
#     optimal_x = minimize(objective, x0, method = 'L-BFGS-B', options = {'disp': True, 'maxiter': 10000}).x
#     # unpack the optimized coefficients
#     return unpack_optimized_coeffs(optimal_x, ndofs, n_unknowns_dynamics, n_unknowns_transformation)


def wrap_optimized_coefficients(ndofs, LinearPart, degree, optimized_coefficients, find_inverse = False):
    """
    Wrap the optimized coefficients into a NonlinearCoordinateTransform object
    Parameters:
        optimized_coefficients: dictionary with optimized coefficients
        find_inverse: if True, the inverse transformation is combuted with regression
    Returns:
        NonlinearCoordinateTransform object
        normal form dynamics
    """
    coeff_dynamics = optimized_coefficients['coeff_dynamics']
    coeffs_transformation = optimized_coefficients['coeff_transformation']
    #print(coeffs_transformation.shape)
    coeffs_transformation = np.concatenate((np.eye(2*ndofs)[:ndofs, :], coeffs_transformation), axis = 1)
    coeffs_trajsformation_conj = np.repeat(coeffs_transformation, 2, axis = 0)  # need to add the complex conjugate to evaluate the transformed coords
    coeffs_trajsformation_conj[ndofs:, :] = np.conj(coeffs_transformation[:ndofs, :])
    # need to add the linear part of the transformation:
    linear_transformation = np.eye(2*ndofs)
    transform_coefficients = np.concatenate((linear_transformation, coeffs_trajsformation_conj), axis = 1)
    #print(transform_coefficients)

    return coeffs_trajsformation_conj, NonlinearCoordinateTransform(2*ndofs, degree, inverse_transform_coefficients=coeffs_trajsformation_conj)



def insert_zeros(coef, structure):
    # the coefficient matrix can be complex too:
    if np.dtype(coef[0,0]) == np.dtype('complex128'):
        full_coeff_matrix = np.zeros(structure.shape, dtype = 'complex128').reshape(-1,1)
    else:
        full_coeff_matrix = np.zeros(structure.shape).reshape(-1,1)
    full_coeff_matrix[structure] = coef.reshape(-1,1)
    return full_coeff_matrix.reshape(1, -1)
    

