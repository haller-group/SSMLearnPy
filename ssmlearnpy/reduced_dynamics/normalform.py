from sklearn.preprocessing import PolynomialFeatures
import numpy as np

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
        Construct the matrix made up of linear combinations of eigenvalues
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
