from scipy.integrate import solve_ivp
from ssmlearnpy.utils import ridge
from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate

import numpy as np

def test_differentiation():
    # test the finite difference function
    x = np.linspace(0, 10, 1000)
    y = np.sin(x).reshape(1,-1)
    X, y_diff = shift_or_differentiate([y], [x], 'flow')
    assert np.allclose(y_diff, np.cos(x))
    assert np.allclose(X, y)
     

def vectorfield(t,x):
    # needed to test regression
    # damped duffing oscillator with fixed points at x=0 and x=+/-1.
    # x[0] is position, x[1] is velocity
    return np.array([x[1], x[0]-x[0]**3 - 0.1*x[1]])

def test_ridge():
    # Test constrained ridge regression 

    # Generate data to fit the model (actual trajcetories)
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    ic_0 = np.array([0.1, 0])
    ic_1 = np.array([-0.1, 0])
    sol_0 = solve_ivp(vectorfield, [t[0], t[-1]], ic_0, t_eval=t)
    sol_1 = solve_ivp(vectorfield, [t[0], t[-1]], ic_1, t_eval=t)
    trajectories = [sol_0.y, sol_1.y]
    X, y  = shift_or_differentiate(trajectories, [t,t], 'flow')

    # Fit reduced model
    mdl = ridge.get_fit_ridge(X, y, poly_degree = 5)
    assert mdl.map_info['coefficients'].shape == (2, 20)
    # generate data to evaluate vector field
    xx = np.linspace(-1, 1, 10)
    yy = np.linspace(-0.5, 0.5, 10)
    Xgrid, Ygrid = np.meshgrid(xx, yy)
    toEvaluate = np.array([Xgrid.ravel(), Ygrid.ravel()]).T
    # evaluate vector field
    y_pred = mdl.predict(toEvaluate)
    # check that the vector field is correct
    true_y = np.array([vectorfield(0, point) for point in toEvaluate])
    assert np.allclose(y_pred, true_y, atol=1e-2) # the vector fields match to within 1e-2

def test_ridge_constrained():
    assert np.allclose(y_pred, true_y, atol=1e-2) # the vector fields match to within 1e-2

if __name__ == '__main__':
    test_differentiation()
    test_ridge()
    test_ridge_constrained()

    
    