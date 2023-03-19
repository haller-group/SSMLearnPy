from sklearn.preprocessing import PolynomialFeatures
import numpy as np




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
        features = PolynomialFeatures(degree=self.degree, include_bias=False)
        # return a function to be fitted and used in the transform method
        return lambda x : features.fit_transform(x.T).dot(transform_coefficients.T) # note the transpose because PolynomialFeatures expects the data to be of shape (n_samples, n_features)
    

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
        return self.transform_map(z).T # transpose again to get the output in the right shape


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
        linear_combinations = np.zeros(coeffs_for_exponents.shape) # output is the same shape 
        return np.repeat(np.diag(eigenvalues_diag).reshape(-1,1), coeffs_for_exponents.shape[1], axis = 1) - np.matmul(np.diag(eigenvalues_diag), coeffs_for_exponents)
    
    def _resonance_condition(self, degree = 3):
        """
        Check if the eigenvalue linear combinations satisfy the resonance condition
        """
        lincombs = self._eigenvalue_lin_combinations(degree)
        return np.abs(lincombs) < self.tolerance # return a boolean array
