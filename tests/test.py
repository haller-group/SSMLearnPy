from scipy.integrate import solve_ivp
from ssmlearnpy.utils import ridge
from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate
from ssmlearnpy.reduced_dynamics.normalform import NormalForm


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
    #print(np.max(np.abs(y_pred-true_y)))
    constLHS = [[0,0],
                [-1, 0],
                [1, 0]
    ]

    constRHS = [[0,0],
                [0, 0],
                [0, 0]
    ]
    cons = [constLHS, constRHS]
    assert np.allclose(y_pred, true_y, atol=1e-2) # the vector fields match to within 1e-2

def test_ridge_constrained():
    # Generate data to fit the model (actual trajcetories)
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    ic_0 = np.array([0.1, 0])
    ic_1 = np.array([-0.1, 0])
    sol_0 = solve_ivp(vectorfield, [t[0], t[-1]], ic_0, t_eval=t)
    sol_1 = solve_ivp(vectorfield, [t[0], t[-1]], ic_1, t_eval=t)
    trajectories = [sol_0.y, sol_1.y]
    X, y  = shift_or_differentiate(trajectories, [t,t], 'flow')
    # add the constraints to the fixed points explicitly
    constLHS = [[0,0],
                [-1, 0],
                [1, 0]
    ]

    constRHS = [[0,0],
                [0, 0],
                [0, 0]
    ]
    cons = [constLHS, constRHS]
    # Fit reduced model
    mdl = ridge.get_fit_ridge(X, y, poly_degree = 5, constraints = cons)
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
    # check that the constraints are satisfied
    assert np.allclose(mdl.predict(np.array(constLHS)), np.array(constRHS))



# test normal form transforms:
def test_normalform_nonlinear_coeffs():
    linearpart = np.ones((2,2))
    nf = NormalForm(linearpart)
    nonlinear_coeffs = nf._nonlinear_coeffs(degree = 3)

    truecoeffs = np.array([[2, 1, 0, 3, 2, 1, 0], [0, 1, 2, 0, 1, 2, 3]])
    assert np.all(nonlinear_coeffs == truecoeffs)

def test_normalform_lincombinations():
    # use the linear part of the above vectorfield at -1, 0:
    linearpart = np.array([[-0.05 - 1j*1.41332940251026, 0], [0, -0.05 + 1j*1.41332940251026]])
    nf = NormalForm(linearpart)
    lincombs = nf._eigenvalue_lin_combinations(degree = 3)
    lamb = -0.05 - 1j*1.41332940251026
    lambconj = -0.05 + 1j*1.41332940251026
    true_lincombs = np.array([[-lamb, -lambconj, lamb - 2*lambconj, -2*lamb, -2*np.real(lamb), -2*lambconj, lamb-3*lambconj],
                              [lambconj-2*lamb, -lamb, -lambconj, lambconj-3*lamb, -2*lamb, -2*np.real(lamb), -2*lambconj]])
    assert np.allclose(lincombs, true_lincombs)
    
def test_normalform_resonance():
     # use the linear part of the above vectorfield at -1, 0:
    linearpart = np.array([[-0.05 - 1j*1.41332940251026, 0], [0, -0.05 + 1j*1.41332940251026]])
    nf = NormalForm(linearpart)
    resonances = nf._resonance_condition(degree = 3)
    whereistrue = np.where(resonances)
    assert np.allclose(whereistrue, [[0,1],[4,5]])
    
if __name__ == '__main__':
    test_differentiation()
    test_ridge()
    test_ridge_constrained()
    test_normalform_nonlinear_coeffs()
    test_normalform_lincombinations()
    test_normalform_resonance()
    
    