from scipy.integrate import solve_ivp
from ssmlearnpy.utils import ridge
import numpy as np
from ssmlearnpy.utils.preprocessing import complex_polynomial_features, compute_polynomial_map
from sklearn.preprocessing import PolynomialFeatures


def test_complex_polynomial_features():
    np.random.seed(0)
    x = np.random.rand(20,3) + 1j*0.#np.random.rand(1,3) 
    complex_features = complex_polynomial_features(x, degree=2)
    poly_features= PolynomialFeatures(degree=2, include_bias=False).fit_transform(np.real(x))
    assert(np.allclose(np.real(complex_features), poly_features))



def test_constrained_regression():
    # Test constrained ridge regression 
    x = np.random.rand(20,3)
    yy = (x[:,0]**2 + x[:,1]**2)*(x[:,0]-1 + x[:,1]-1 + x[:,2]-1)
    
    x = np.concatenate((x, np.ones((1,3))), axis = 0)
    yy = np.concatenate((yy, np.zeros((1))), axis = 0)
    
    yy = yy.reshape(-1,1)
    mdl = ridge.get_fit_ridge(x.T, yy.T, poly_degree=5, constraints=[[[1., 1., 1. ]], [[0.]]])
    assert(np.allclose(mdl.predict(x), yy, atol=1e-6))

def test_compute_polynomial_map():
    # Test compute_polynomial_map
    x = np.random.rand(20, 3)
    yy = np.random.rand(20, 2)
    x = np.concatenate((x, np.ones((1,3))), axis = 0)
    yy = np.concatenate((yy, np.zeros((1, 2))), axis = 0)
    mdl = ridge.get_fit_ridge(x.T, yy.T, poly_degree=3, constraints=[[[1., 1., 1. ]], [[0., 0.]]])
    coeffs_ = mdl.map_info['coefficients']
    complex_features = complex_polynomial_features(x, degree=3)
    mapp = compute_polynomial_map(coeffs_, degree=3, include_bias=False)
    assert np.allclose(mapp(x.T).T,mdl.predict(x)) 
    
if __name__ == '__main__':
    test_complex_polynomial_features()
    test_constrained_regression()
    test_compute_polynomial_map()