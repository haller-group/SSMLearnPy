from scipy.integrate import solve_ivp
from ssmlearnpy.utils import ridge
from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate
from ssmlearnpy.reduced_dynamics.normalform import NormalForm, NonlinearCoordinateTransform
from sklearn.preprocessing import PolynomialFeatures

from ssmlearnpy.reduced_dynamics.normalform import NonlinearCoordinateTransform, NormalForm, create_normalform_transform_objective
import numpy as np


from ssmlearnpy.utils.preprocessing import complex_polynomial_features

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
    #print(X[0].shape)
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



def test_nonlinear_change_of_coords():
    xx = np.linspace(-1, 1, 10)
    yy = np.linspace(-0.5, 0.5, 10)
    transformedx = xx+yy+xx**2+yy**2
    vect = np.array([xx, yy])
    #print(vect.shape)
    transformedy = xx-yy+xx**3-yy*xx
    poly_features= PolynomialFeatures(degree=3, include_bias=False).fit(vect.T)
    vect_transform = poly_features.transform(vect.T)
    #print(vect_transform.shape)
    exponents = poly_features.powers_.T
    #print(exponents)
    # (2, 10)
    # (10, 9)
    # [[1 0 2 1 0 3 2 1 0]
    # [0 1 0 1 2 0 1 2 3]]
    transformationCoeffs = np.array([[1,1, 1, 0, 1,0, 0,0,0], # x + y +x^2 +y^2
                                     [1, -1, 0, -1, 0, 1, 0, 0, 0]]) # x - y + x^3 - yx
    TF = NonlinearCoordinateTransform(2, 3, transform_coefficients=transformationCoeffs)
    z = TF.transform(vect)
    assert np.allclose(z[0,:], transformedx)

    assert np.allclose(z[1,:], transformedy)
    

def test_complex_polynomial_features():
    Y = np.random.rand(2,10)
    poly = PolynomialFeatures(degree=3, include_bias=False).fit_transform(Y.T)
    complexpoly = complex_polynomial_features(Y, degree=3)
    assert np.allclose(poly.T, complexpoly)

def test_complex_polynomial_features():
    Y = np.random.rand(2,10)
    poly = PolynomialFeatures(degree=3, include_bias=False).fit_transform(Y.T)
    complexpoly = complex_polynomial_features(Y, degree=3)
    assert np.allclose(poly.T, complexpoly)



def test_normalform_transform():
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    ic_0 = np.array([0.1, 0])
    ic_1 = np.array([0.112, 0])
    sol_0 = solve_ivp(vectorfield, [t[0], t[-1]], ic_0, t_eval=t)
    sol_1 = solve_ivp(vectorfield, [t[0], t[-1]], ic_1, t_eval=t)
    trajectories = [sol_0.y - np.array([1,0]).reshape(-1,1), sol_1.y - np.array([1,0]).reshape(-1,1)]
    times = [t, t]
    X, y  = shift_or_differentiate(trajectories, times, 'flow') # get an estimate for the linear part
    #print(X[0].shape)
    # add the constraints to the fixed points explicitly
    constLHS = [[-1, 0]]
    constRHS = [[0, 0]]
    cons = [constLHS, constRHS]
    # Fit reduced model
    mdl = ridge.get_fit_ridge(X, y, poly_degree = 3, constraints = cons)
    linearPart = mdl.map_info['coefficients'][:,:2]
    w, v = np.linalg.eig(linearPart)
    #print(np.linalg.inv(v)@linearPart@v)
    linpartDiag = np.linalg.inv(v)@linearPart@v
    # v^{-1}@A@v is diagonal. transform the coordinates with v^{-1}.
    newcoord = [np.linalg.inv(v)@x for x in trajectories]
    
    #print(newcoord[0][0,:]-np.conj(newcoord[0])[1,:]) is all zeros.

    objectiv = create_normalform_transform_objective(times, newcoord, linpartDiag, degree = 3)
    print(objectiv(np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])).shape)

if __name__ == '__main__':
    test_differentiation()
    test_ridge()
    test_ridge_constrained()
    test_normalform_nonlinear_coeffs()
    test_normalform_lincombinations()
    test_normalform_resonance()
    test_nonlinear_change_of_coords()
    test_normalform_transform()
    #test_complex_polynomial_features()
    