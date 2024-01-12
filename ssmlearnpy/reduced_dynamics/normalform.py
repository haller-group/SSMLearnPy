
from sklearn.preprocessing import PolynomialFeatures
from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate
import numpy as np
from ssmlearnpy.utils.ridge import get_fit_ridge
from ssmlearnpy.utils.preprocessing import get_matrix, complex_polynomial_features, generate_exponents, compute_polynomial_map, insert_complex_conjugate, unpack_coefficient_matrices_from_vector
from ssmlearnpy.utils import ridge
from typing import NamedTuple



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

    def __init__(
        self,
        dimension,
        degree,
        transform_coefficients=None,
        inverse_transform_coefficients=None,
        linear_transform = None,
    ):
        # can initialize with the coefficients of the transformation or as an empty object. in that case use the set_transform_coefficients method
        # coefficients should be given as a matrix of shape (n_outputs, n_poly_features)
        self.transform_coefficients = transform_coefficients
        self.inverse_transform_coefficients = inverse_transform_coefficients
        self.dimension = dimension
        self.degree = degree
        self.transform_map = None
        self.inverse_transform_map = None
        self.linear_transform = None
        if linear_transform is not None:
            self.linear_transform = linear_transform # should be the inverse of the eigenvector basis
        # to be consistent with the ssm.decode() method,
        # self.transform_map and self.inverse_transform_map expect a matrix of shape (n_samples, n_features)
        if self.transform_coefficients is not None:
            self.transform_map = compute_polynomial_map(
                self.transform_coefficients, self.degree, linear_transform = self.linear_transform)
        if self.inverse_transform_coefficients is not None:
            self.inverse_transform_map = compute_polynomial_map(
                self.inverse_transform_coefficients, self.degree, linear_transform = self.linear_transform)

    def set_transform_coefficients(
            self,
            transform_coefficients
    ):
        self.transform_coefficients = transform_coefficients
        self.transform_map = compute_polynomial_map(
            self.transform_coefficients, self.degree, linear_transform = np.eye(self.dimension))

    def set_inverse_transform_coefficients(
            self,
            inverse_transform_coefficients
    ):
        self.inverse_transform_coefficients = inverse_transform_coefficients
        self.inverse_transform_map = compute_polynomial_map(
            self.inverse_transform_coefficients, self.degree, linear_transform = self.linear_transform)

    def transform(self, z):
        """
        Transform the coordinates z to y.
        Parameters: 
            z:  matrix of shape (n_features, n_samples), or list of matrices
        returns:
            y: matrix of shape (n_features, n_samples), or list of matrices
        """
        if self.transform_coefficients is None:
            raise RuntimeError(
                "The transformation coefficients have not been set")
        # transpose again to get the output in the right shape
        InvMatrix = np.linalg.inv(self.linear_transform)
        if isinstance(z, list):
            return [InvMatrix@self.transform_map(z_i).T for z_i in z]
        else:
            return InvMatrix@self.transform_map(z).T

    def inverse_transform(self, z):
        """
        Perform the inverse transformation from the coordinates y to z
        Parameters: 
            y:  matrix of shape (n_features, n_samples), or list of matrices
        returns:
            z: matrix of shape (n_features, n_samples), or list of matrices
        """
        if self.inverse_transform_coefficients is None:
            raise RuntimeError(
                "The inverse transformation coefficients have not been set")
        if isinstance(z, list):
            return [self.inverse_transform_map(z_i).T for z_i in z]
        else:
            return self.inverse_transform_map(z).T
        

class NormalForm:
    """
    General class to transform to normal form

    Parameters:
        - Linear part (matrix): linear part of the dynamics 
        - tolerance: tolerance for the resonance condition

    """

    def __init__(
        self,
        LinearPart,
        tolerance=None
    ):
        # A good candidate for the tolerance is 10*min(Re(\lambda))
        self.tolerance = tolerance
        if self.tolerance is None:
            eigenvalues, _ = np.linalg.eig(
                LinearPart)  # eigenvalues are sorted
            self.tolerance = 10*np.min(np.abs(np.real(eigenvalues)))
        self.resonances = None
        # we precompute the transformation that diagnolizes the linear part
        self.LinearPart, self.diagonalizing_matrix, self.diagonalized = diagonalize_linear_part(
            LinearPart)
        # self.LinearPart is always assumed to be diagonal and can be written as
        # self.LinearPart = (self.diagonalizing_matrix)^{-1} @ original_matrix @ self.diagonalizing_matrix
        self.dynamics_structure = None
        self.transformation_structure = None

    def _nonlinear_coeffs(
        self,
        degree=3
    ):
        """
        Computes the coefficients of the nonlinear part of the vector field. For more info see M. Cenedese (2021) 
        """
        dimension = self.LinearPart.shape[0]  # number of dimensions of the manifold ~ number of eigenvalues
        # gives a matrix of shape (dimension, number of monomials of degree <= degree)
        exponents_of_nonlinear_map = generate_exponents(dimension, degree)
        exponents_of_nonlinear_map = exponents_of_nonlinear_map[:, dimension:]
        return exponents_of_nonlinear_map

    def _eigenvalue_lin_combinations(
        self,
        degree=3,
        use_center_manifold_style = False
    ):
        """
        Construct the matrix made up of linear combinations of eigenvalues. For more info see e. g. M. Cenedese (2021)
        """
        if use_center_manifold_style:
            LinearPart = np.imag(self.LinearPart)
        else:
            LinearPart = self.LinearPart
        coeffs_for_exponents = self._nonlinear_coeffs(degree)
        return np.repeat(np.diag(LinearPart).reshape(-1, 1),
                         coeffs_for_exponents.shape[1], axis=1) - \
            np.matmul(np.diag(LinearPart), coeffs_for_exponents)

    def _resonance_condition(
        self,
        degree=3,
        use_center_manifold_style = False
    ):
        """
        Check if the eigenvalue linear combinations satisfy the resonance condition
        """
        lincombs = self._eigenvalue_lin_combinations(degree, use_center_manifold_style = use_center_manifold_style)
        return np.abs(lincombs) < self.tolerance  # return a boolean array
    
    def scale_diagonalizing_matrix(self, scaling_matrix):
        self.diagonalizing_matrix = np.matmul(scaling_matrix, self.diagonalizing_matrix)

    def set_dynamics_and_transformation_structure(
        self,
        type='flow',
        degree=3,
        use_center_manifold_style = False
    ):
        
        if type != 'flow':
            raise NotImplementedError("Only flow is implemented")

        size = self.LinearPart.shape[0]
        # only consider the first half of the coordinates: conjugates are discarded
        ndofs = int(size / 2)
        # assumes that the first half of the coordinates are the real part and the second half the imaginary part.
        # sparse structure for the nonlinearities
        N_structure = self._resonance_condition(degree, use_center_manifold_style = use_center_manifold_style)[:ndofs, :]
        # if there is a 1 in the structure matrix, we keep the corresponding column ~ logical or
        self.dynamics_structure = np.sum(
            N_structure, axis=0).astype(dtype=bool)
        self.transformation_structure = np.logical_not(
            self.dynamics_structure)  # complement
        return


def prepare_normalform_transform_optimization(
    times,
    trajectories,
    LinearPart,
    type='flow',
    degree = 3,
    do_scaling = True,
    tolerance = None,
    use_center_manifold_style = False
):
    """
    Precompute terms for the optimization problem
    Parameters:
        times: list of times
        trajectories: list of trajectories (each of shape (n_features, n_samples))
        LinearPart: linear part of the dynamics
        type: 'flow' or 'map'
        degree: max. degree of the polynomial transformation
        do_scaling: if True, the transformation is scaled to yield modal coordinates of amplitude 1/2
        tolerance : tolerance for the resonance condition
        use_center_manifold_style: if True, all resonances are eliminated that would be present in a center manifold calculation. 
    Returns:
        linear error
        Dynamics_normalform_polynomial_features: list of non-zero polynomial features of the trajectories included in the normal form dynamics
        Transform_normalform_polynomial_features: list of non-zero polynomial features of the trajectories included in the normal form transformation
        Transform_normalform_polynomial_features_timederivative: list of non-zero polynomial features of the trajectories included in the normal form transformation's time derivative
    """

    size = LinearPart.shape[0]
    # only consider the first half of the coordinates: conjugates are discarded
    ndofs = int(size / 2)
    if type != 'flow':
        raise NotImplementedError("Only flow is implemented")
    normalform = NormalForm(LinearPart, tolerance=tolerance)
    if do_scaling:
        scaling_matrix = rescale_linear_part(normalform.diagonalizing_matrix, trajectories)
        normalform.scale_diagonalizing_matrix(scaling_matrix)

    normalform.set_dynamics_and_transformation_structure(type, degree, use_center_manifold_style=use_center_manifold_style)
    # check if the normal form is applicable. If there are real eigenvalues, raise an error
    if np.allclose(np.imag(np.diag(normalform.LinearPart)), 0):
        raise ValueError(
            "The normal form is not applicable. All eigenvalues need to be complex")
    if normalform.diagonalized is False:
        trajectories = [np.matmul(normalform.diagonalizing_matrix, y)
                        for y in trajectories]
    linear_error = []
    _, dy_dt = shift_or_differentiate(
        trajectories, times, type='flow')  # only flow is implemented
    for i, y in enumerate(trajectories):
        # compute the error in the linear part
        linear_error.append(dy_dt[i] - normalform.LinearPart.dot(y))
    # precomute nonlinear terms in the transformation map
    transformation_normalform_polynomial_features = []
    for y in trajectories:
        transformation_normalform_polynomial_features.append(
            complex_polynomial_features(y.T,
                                        degree=degree,
                                        skip_linear=True,
                                        structure=normalform.transformation_structure
                                        ).T
        )
    # compute the derivative of the nonlinear features
    _, transformation_normalform_polynomial_features_timederivative = shift_or_differentiate(
        transformation_normalform_polynomial_features, times, type='flow')
    # also remove the unnecessary coordinates from the linear error
    return trajectories, \
        normalform,\
        [l[:ndofs, :] for l in linear_error],\
        transformation_normalform_polynomial_features, \
        transformation_normalform_polynomial_features_timederivative

def create_normalform_initial_guess(initial_reduced_model,
                                        normalform,
                                        type='flow',
                                        degree=3
                                        ):
    
    """We do a ridge regression on the diagonalized coordinates (modal coordinates). 
    The initial condition for the optimization is taken as the modal dynamics ridge regression problem. 
    """
    # compute the modal coordinates
    initial_nonlinear_coeffs = initial_reduced_model.map_info['coefficients']
    initial_nonlinear_coeffs_in_modalcoords = np.matmul(np.linalg.inv(normalform.diagonalizing_matrix), initial_nonlinear_coeffs)
    # Set the initial guess for the normal form dynamics coeffs:
    size = normalform.LinearPart.shape[0]
    ndofs = int(size / 2)
    initial_nonlinear_coeffs_in_modalcoords = initial_nonlinear_coeffs_in_modalcoords[:ndofs, :] # 
    initial_guess_dynamics = initial_nonlinear_coeffs_in_modalcoords[:, size:][:, normalform.dynamics_structure]
    # Set the initial guess for the normal form transformation coeffs as -1* the modal dynamics coeffs.
    initial_guess_transformation =  -1*initial_nonlinear_coeffs_in_modalcoords[:, size:][:, normalform.transformation_structure]
    iniitial_guess_joint_complex  = np.concatenate((initial_guess_dynamics.ravel(), initial_guess_transformation.ravel()))
    return np.concatenate((np.real(iniitial_guess_joint_complex), np.imag(iniitial_guess_joint_complex)))

def create_normalform_transform_objective(
    times,
    trajectories,
    Linearpart,
    type='flow',
    degree=3,
    do_scaling = True,
    tolerance=None,
    use_center_manifold_style = False
):
    """
    Minimize
    ||d\dt (y + f_normalform(y)) - \Lambda*(y+f_normalform(y)) - N_normalform(y+f_normalform)||^2
    where \Lambda is the linear part of the dynamics and N_normalform is the nonlinear part of the normal form dynamics
    f_normalform is the nonlinear part of the normal form transformation z = T^{-1}(y) = y + f_normalform(y).
    Based on the linear part we enforce the structure of the nonlinearities in N_normalform and f_normalform.
    Parameters:
        times: list of times
        trajectories: list of trajectories (each of shape (n_features, n_samples))
        LinearPart: linear part of the dynamics
        type: 'flow' or 'map'
        degree: max. degree of the polynomial transformation
        do_scaling: if True, the transformation is scaled to yield modal coordinates of amplitude 1/2
        tolerance: tolerance for the resonance condition
    Returns:
        normalform Object
        n_unknowns_dynamics: number of unknowns to optimize for in the dynamics
        n_unknowns_transformation: number of unknowns to optimize for in the transformation
        objective: objective function to be minimized. 
                    Takes a single argument x, which is a real vector of size 2 * (n_unknowns_dynamics + n_unknowns_transformation)
    """
    # from this point, trajectories are assumed to be in the diagonalized coordinates
    trajectories, normalform, linear_error, \
        transformation_normalform_polynomial_features, \
        transformation_normalform_polynomial_features_timederivative = prepare_normalform_transform_optimization(times,
                                                                                                                 trajectories,
                                                                                                                 Linearpart,
                                                                                                                 type,
                                                                                                                 degree,
                                                                                                                 do_scaling=do_scaling,
                                                                                                                   tolerance=tolerance,
                                                                                                                   use_center_manifold_style = use_center_manifold_style)
    # set up optimization
    size = Linearpart.shape[0]
    # only consider the first half of the coordinates: conjugates are discarded
    ndofs = int(size / 2)
    # set up optimization
    n_unknowns_dynamics = np.sum(normalform.dynamics_structure) #* nodfs
    n_unknowns_transformation = transformation_normalform_polynomial_features[0].shape[0] #* ndofs  # need to create a matrix of shape (ndofs, transformation_normalform_polynomial_features[0])
    def objective(x):  # x is purely real
        # convert to complex form. First half is real, second half is imaginary
        n_unknowns = int(len(x) / 2)
        x_complex = x[:n_unknowns] + 1j*x[n_unknowns:]
        # separate these into matrices of coefficients:
        n_rows = int(n_unknowns_dynamics/ndofs)
        coeff_dynamics, coeff_transformation = unpack_coefficient_matrices_from_vector(
            x_complex, n_unknowns_dynamics, n_rows, ndofs)
        nonlinear_error1 = []
        nonlinear_error2 = []
        derivative_error = []
        for i, y in enumerate(trajectories):
            transformed = y[:ndofs, :] + np.matmul(
                coeff_transformation, transformation_normalform_polynomial_features[i])
            transformed_conj = insert_complex_conjugate(transformed)
            # here transformed_conj has shape (2*ndofs, n_samples)
            # compute N_normalform(y+f_normalform)
            temp = complex_polynomial_features(transformed_conj.T,
                                               degree=degree,
                                               skip_linear=True,
                                               structure=normalform.dynamics_structure
                                               ).T
            nonlinear_error1.append(np.matmul(coeff_dynamics, temp))
            # compute Lambda*(f_normalform)
            nonlinear_error2.append(
                np.matmul(normalform.LinearPart[:ndofs, :ndofs],
                          np.matmul(coeff_transformation,
                          transformation_normalform_polynomial_features[i])
                          )
            )
            # compute d/dt (f_normalform)
            derivative_error.append(
                np.matmul(coeff_transformation, 
                          transformation_normalform_polynomial_features_timederivative[i])
                )
        # the total error is: linear_error + d/dt (f_normalforn) - Lambda*(f_normalform) - N_normalform(y + f_normalform)
        error_along_trajs = [l + d - n1 - n2 for l, d, n1, n2 in zip(
            linear_error, derivative_error, nonlinear_error1, nonlinear_error2)]
        # sum over all trajectories. Could normalize with the number of trajectories

        sum_error = np.hstack([[np.real(e.ravel()), np.imag(e.ravel())] for e in error_along_trajs]) # this is total error over all trajectories (complex)
        return np.array(sum_error).ravel()

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
    n_unknowns = int(len(optimal_x) / 2)
    x_complex = optimal_x[:n_unknowns] + 1j*optimal_x[n_unknowns:]
    # separate these into matrices of coefficients:
    coeff_dynamics = x_complex[:n_unknowns_dynamics].reshape(ndofs, int(
        n_unknowns_dynamics/ndofs)).reshape((ndofs, int(n_unknowns_dynamics/ndofs)))
    coeff_transformation = x_complex[n_unknowns_dynamics:n_unknowns_dynamics +
                                     n_unknowns_transformation].reshape((ndofs, int(n_unknowns_transformation/ndofs)))

    # insert the zeros into the coefficient matrices, which is dictated by normalform.dynamics_structure and normalform.transformation_structure
    coeff_dynamics = insert_zeros(
        coeff_dynamics, normalform.dynamics_structure)
    coeff_transformation = insert_zeros(
        coeff_transformation, normalform.transformation_structure)

    return {'coeff_dynamics': coeff_dynamics, 'coeff_transformation': coeff_transformation}


def wrap_optimized_coefficients(
        ndofs,
        normalform,
        degree,
        optimized_coefficients,
        find_inverse = False,
        trajectories = None,
        **regression_kwargs
        ):
    """
    Wrap the optimized coefficients into a NonlinearCoordinateTransform object
    Parameters:
        ndofs: number of degrees of freedom
        normalform: normalform object used in the fit
        degree: max. degree used in the fit
        optimized_coefficients: dictionary with the optimal coefficients for the dynamics and for the transformation
        find_inverse: if True, the inverse transformation is combuted with ridge regression.
            if this is True, trajectories must be provided
    Returns:
        NonlinearCoordinateTransform object. Only its inverse_transform() method is fitted by default. 
                            If find_inverse is True, the transform() method is also fitted. 
        dynamics: dictionary collecting the exponents, coefficients and a callable vectorfield
    """
    coeff_dynamics = optimized_coefficients['coeff_dynamics']
    coeff_dynamics = np.concatenate(
        (normalform.LinearPart[:ndofs, :], coeff_dynamics), axis=1)
    coeffs_transformation = optimized_coefficients['coeff_transformation']
    coeffs_transformation = np.concatenate(
        (np.eye(2*ndofs)[:ndofs, :], coeffs_transformation), axis=1)
    # collect the reduced dynamics in a dictionary:


    class Dynamics(NamedTuple): # to behave in the same way as the pipeline object generated by get_fit_ridge
        predict : callable
        map_info : dict
        fit = None

    dynamics = Dynamics(lambda x : compute_polynomial_map(coeff_dynamics, degree)(x).T,
                        {})

    dynamics.map_info['coefficients'] = coeff_dynamics
    dynamics.map_info['exponents'] = generate_exponents(2 * ndofs, degree)

    def vectorfield(t, x):
        xeval = x.reshape(-1,1)
        evaluation = compute_polynomial_map(coeff_dynamics, degree)(xeval).T
        # need to reshape this for the complex_polynomial_features function
        return insert_complex_conjugate(evaluation)[:,0]
    dynamics.map_info['vectorfield'] = vectorfield
    transformation = NonlinearCoordinateTransform(
                                        2*ndofs,
                                        degree, 
                                        inverse_transform_coefficients = coeffs_transformation,
                                        linear_transform = normalform.diagonalizing_matrix
                                    )
    if find_inverse:
        if trajectories is None:
            raise ValueError('If find_inverse is True, trajectories must be provided')
        trajectories_in_modal_coords =[np.matmul(normalform.diagonalizing_matrix, t)  for t in trajectories]
        # trajectories_in_normal_coords contains complex data, so can't use the ordinary ridge regression
        # the inverse also needs to be near identity: y = z + f_inverse(z), f_inverse \in O(z^2)
        coeffs, _ = ridge.fit_inverse(transformation.inverse_transform,
                                      trajectories_source = trajectories,degree = degree,
                                      trajectories_target = trajectories_in_modal_coords,
                                      near_identity=True)

        transformation.set_transform_coefficients(np.array(coeffs))
    return transformation, dynamics


def diagonalize_linear_part(LinearPart):
    # do a linear transformation such that the lin. part
    # is diagonal and the eigenvalues are sorted according to their real parts
    eigenvalues, eigenvectors = np.linalg.eig(LinearPart)
    index = eigenvalues.argsort()[::-1]  # largest eigenvalues first
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]
    # np.linalg.inv(eigen) @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors) = i
    # only true if it was sorted correctly too
    is_diagonal = np.allclose(np.diag(eigenvalues), LinearPart)
    return np.diag(eigenvalues), np.linalg.inv(eigenvectors), is_diagonal

def rescale_linear_part(diagonalizing_matrix, trajectories):
    # rescale the linear part such that the max amplitude of the trajectories is 1/2
    # this is done by rescaling the diagonalizing matrix
    trajectories_matrix = get_matrix(trajectories)
    modal_trajectories = np.matmul(diagonalizing_matrix, trajectories_matrix)
    max_modal_amplitude = np.max(np.abs(modal_trajectories), axis = 1)

    rescalematrix = np.linalg.inv(2* max_modal_amplitude *np.eye(2))
    return rescalematrix # np.matmul(rescalematrix, diagonalizing_matrix)



def insert_zeros(coef, structure):
    # the coefficient matrix can be complex:
    if np.dtype(coef[0, 0]) == np.dtype('complex128'):
        full_coeff_matrix = np.zeros(
            structure.shape, dtype='complex128').reshape(-1, 1)
    else:
        full_coeff_matrix = np.zeros(structure.shape).reshape(-1, 1)
    full_coeff_matrix[structure] = coef.reshape(-1, 1)
    return full_coeff_matrix.reshape(1, -1)