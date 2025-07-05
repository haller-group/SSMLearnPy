from scipy.integrate import solve_ivp
from ssmlearnpy.utils import ridge

from ssmlearnpy import SSMLearn

from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate
from ssmlearnpy.reduced_dynamics.normalform import NormalForm, NonlinearCoordinateTransform
from sklearn.preprocessing import PolynomialFeatures
from ssmlearnpy.utils.preprocessing import get_matrix , PolynomialFeaturesWithPattern, sort_complex_eigenpairs
from ssmlearnpy.reduced_dynamics.normalform import NonlinearCoordinateTransform, NormalForm, create_normalform_transform_objective, prepare_normalform_transform_optimization, unpack_optimized_coeffs
from ssmlearnpy.geometry.coordinates_embedding import coordinates_embedding
from scipy.optimize import minimize
import numpy as np
from scipy.io import savemat, loadmat

from ssmlearnpy.utils.preprocessing import complex_polynomial_features

def test_differentiation():
    # test the finite difference function
    x = np.linspace(0, 10, 1000)
    y = np.sin(x).reshape(1,-1)
    X, y_diff = shift_or_differentiate([y], [x], 'flow')
    assert np.allclose(y_diff, np.cos(x))
    assert np.allclose(X, y)
     

def vectorfield(t,x, r = 1):
    # needed to test regression
    # damped duffing oscillator with fixed points at x=0 and x=+/-1.
    # x[0] is position, x[1] is velocity
    return np.array([x[1], x[0]-r*x[0]**3 - 0.1*x[1]])

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


def test_ridge_with_or_without_scaling():
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    ic_0 = np.array([0.1, 0])
    ic_1 = np.array([-0.1, 0])
    sol_0 = solve_ivp(vectorfield, [t[0], t[-1]], ic_0, t_eval=t)
    sol_1 = solve_ivp(vectorfield, [t[0], t[-1]], ic_1, t_eval=t)
    trajectories = [sol_0.y, sol_1.y]
    X, y  = shift_or_differentiate(trajectories, [t,t], 'flow')

    # Fit reduced model
    mdl = ridge.get_fit_ridge(X, y, poly_degree = 5) # with scaling
    mdl_noscale = ridge.get_fit_ridge(X, y, do_scaling = False, poly_degree = 5) # with scaling
    # generate data to evaluate vector field
    xx = np.linspace(-1, 1, 10)
    yy = np.linspace(-0.5, 0.5, 10)
    Xgrid, Ygrid = np.meshgrid(xx, yy)
    toEvaluate = np.array([Xgrid.ravel(), Ygrid.ravel()]).T
    # evaluate vector field
    y_pred_scale = mdl.predict(toEvaluate)
    y_pred_noscale = mdl_noscale.predict(toEvaluate)
    assert np.allclose(y_pred_scale, y_pred_noscale)

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

# def test_delay_embedding():
#     #data = loadmat('../examples/brakereussbeam/data.mat')['data_BRB']

#     TimeDIC = data[0,0].item()[0]
#     DisplacementDIC = data[0,0].item()[1]
#     TimeACC = data[0,0].item()[2]
#     AccelerationACC = data[0,0].item()[3]
#     Xmesh = data[0,0].item()[4]
#     Units = data[0,0].item()[5]
#     LocationACC = data[0,0].item()[6]
#     PFFResultsACC = data[0,0].item()[7]
#     ssm = SSMLearn(
#         t = [TimeDIC.ravel()], 
#         x = [DisplacementDIC], 
#         ssm_dim=2, 
#         dynamics_type = 'flow'
#     )
#     #referenceData = loadmat('test_BRB_from_ssmlearn.mat')['yData']
#     t_y, y, opts_embedding = coordinates_embedding(ssm.emb_data['time'], ssm.emb_data['observables'],
#                                                imdim = ssm.ssm_dim, over_embedding = 5)
#     assert np.allclose(t_y, referenceData[0,0])
#     assert np.allclose(y, referenceData[0,1])

# def test_dimensionality_reduction():
#     #reference_yData = loadmat('test_BRB_from_ssmlearn.mat')['yData']
#     #reference_etaData = loadmat('test_BRB_from_ssmlearn.mat')['etaData']
    
#     ssm = SSMLearn(
#         t = [reference_yData[0,0]], 
#         x = [reference_yData[0,1]], 
#         ssm_dim=2, 
#         derive_embdedding = False,
#         dynamics_type = 'flow'# use the embedding from the reference data
#     )
#     ssm.get_reduced_coordinates('linearchart')
#     assert np.allclose(ssm.emb_data['reduced_coordinates'][0], reference_etaData[0,1])


def test_fit_reduced_coords_and_parametrization():
    t = np.linspace(0, 10, 1000)
    ic_0 = np.array([0.1, 0])
    ic_1 = np.array([-0.1, 0])
    sol_0 = solve_ivp(vectorfield, [t[0], t[-1]], ic_0, t_eval=t)
    sol_1 = solve_ivp(vectorfield, [t[0], t[-1]], ic_1, t_eval=t)
    trajectories = [sol_0.y, sol_1.y]
    t_emb, y_emb, _ = coordinates_embedding([t,t], trajectories, imdim = 2, over_embedding = 5)
    enc, dec = ridge.fit_reduced_coords_and_parametrization(y_emb, n_dim = 2, poly_degree=3)
    y_rec = dec.predict(enc.predict(y_emb[0]).T) 
    assert np.allclose(y_rec, y_emb[0].T, atol = 1e-3)
    y_rec = dec.predict(enc.predict(y_emb[1]).T)
    assert np.allclose(y_rec, y_emb[1].T, atol = 1e-3)
    # check the contraints:
    linear_coeff = enc.matrix_representation
    nonlinear_coeff = dec.map_info['coefficients'][:, 2:]
    assert np.allclose(np.matmul(linear_coeff.T, linear_coeff), np.eye(2))
    assert np.allclose(np.matmul(linear_coeff.T, nonlinear_coeff), np.zeros((2, nonlinear_coeff.shape[1])))


def test_polynomial_features_pattern():
    ## original_polyFeatures:
    #print(pf.powers_[pattern,:])
    X = np.random.rand(10,2)
    pf = PolynomialFeatures(degree=3, include_bias=False).fit(X)
    pattern = np.logical_or(np.logical_and(pf.powers_[:,0]==2, pf.powers_[:,1]==1), np.logical_and(pf.powers_[:,0]==1, pf.powers_[:,1]==2))
    # only include x**2 y or x * y**2 terms
    pf2 = PolynomialFeaturesWithPattern(degree =3, include_bias = False, structure = pattern)
    transformed = pf2.fit_transform(X)
    control = np.vstack((X[:,0]**2 * X[:,1], X[:,0] * X[:,1]**2)).T
    assert np.allclose(control, transformed)

#import matplotlib.pyplot as plt
def test_get_fit_ridge_parametric():
    ## z = x**2 + 2x*r + r**2 * y
    Rs = [np.array([0]).reshape(-1,1), np.array([1]).reshape(-1,1), np.array([2]).reshape(-1,1)]
    Zs = []
    Xs = []
    for r in Rs:
        xs = np.random.rand(100)
        ys = np.random.rand(100)
        X, Y = np.meshgrid(xs, ys)
        points = np.vstack((X.ravel(), Y.ravel()))
        z = points[0,:]**2 + 2*points[0,:]*r + r**2 * points[1,:]
        Zs.append(z)
        Xs.append(points)
    mdl = ridge.get_fit_ridge_parametric(Xs, Zs, Rs, poly_degree = 4, origin_remains_fixed=True, poly_degree_parameter=2, do_scaling=False)
    for i,r in enumerate(Rs):
        XX = np.vstack((Xs[i], r*np.ones((1, Xs[0].shape[1]))))
        yy = mdl.predict(XX.T)
        #print(np.max(np.abs(yy - Zs[i].T)))
        assert np.allclose(yy, Zs[i].T)
    #     plt.figure()
    #     plt.plot(Zs[i].T, Zs[i].T, '-')
    #     plt.plot(Zs[i].T, yy, '.')
    # plt.show()

def test_complex_polynomial_features():
    Y = np.random.rand(2,10)
    poly = PolynomialFeatures(degree=3, include_bias=False).fit_transform(Y.T)
    #print(PolynomialFeatures(degree=3, include_bias=False).fit(Y.T).powers_.shape)
    complexpoly = complex_polynomial_features(Y.T, degree=3)
    assert np.allclose(poly.T, complexpoly.T)



def test_sort_complex_eigenpairs_2d():
    A = np.array(
        [[-0.6, -1],
         [1,  -0.6]]
    )
    d, v = np.linalg.eig(A)

    d_sorted, v_sorted = sort_complex_eigenpairs(d, v)

    expected_sorted_real_parts = np.array([-0.6,-0.6])
    expected_sorted_imag_parts = np.array([1, -1])
    # check sorting
    assert np.allclose(np.real(d_sorted), expected_sorted_real_parts, atol=1e-10)
    assert np.allclose(np.imag(d_sorted), expected_sorted_imag_parts, atol=1e-10)
    # check eigenvectors
    for i in range(len(d_sorted)):
        Av = A @ v_sorted[:, i]
        lv = d_sorted[i] * v_sorted[:, i]
        assert np.allclose(Av, lv, atol=1e-10)


def test_sorted_complex_eigenpairs_2d():
    A = np.diag(
        [-0.1 +1.j,    -0.1 -1.j]
        )
    d, v = np.linalg.eig(A)

    d_sorted, v_sorted = sort_complex_eigenpairs(d, v)

    expected_sorted_real_parts = np.array([-0.1, -0.1])
    expected_sorted_imag_parts = np.array([1, -1])
    # check sorting
    assert np.allclose(np.real(d_sorted), expected_sorted_real_parts, atol=1e-10)
    assert np.allclose(np.imag(d_sorted), expected_sorted_imag_parts, atol=1e-10)
    # check eigenvectors
    for i in range(len(d_sorted)):
        Av = A @ v_sorted[:, i]
        lv = d_sorted[i] * v_sorted[:, i]
        assert np.allclose(Av, lv, atol=1e-10)

def test_unsorted_complex_eigenpairs_2d():
    A = np.diag(
        [-0.1 -1.j,   -0.1 +1.j]
        )
    d, v = np.linalg.eig(A)

    d_sorted, v_sorted = sort_complex_eigenpairs(d, v)

    expected_sorted_real_parts = np.array([-0.1,-0.1])
    expected_sorted_imag_parts = np.array([1, -1])
    # check sorting
    assert np.allclose(np.real(d_sorted), expected_sorted_real_parts, atol=1e-10)
    assert np.allclose(np.imag(d_sorted), expected_sorted_imag_parts, atol=1e-10)
    # check eigenvectors
    for i in range(len(d_sorted)):
        Av = A @ v_sorted[:, i]
        lv = d_sorted[i] * v_sorted[:, i]
        assert np.allclose(Av, lv, atol=1e-10)

def test_sorted_complex_eigenpairs_4d():
    A = np.diag(
        [-0.1 +1.j,      -0.33+6.8802j,  -0.1 -1.j,      -0.33-6.8802j]
        )
    d, v = np.linalg.eig(A)

    d_sorted, v_sorted = sort_complex_eigenpairs(d, v)

    expected_sorted_real_parts = np.array([-0.1,-0.33, -0.1, -0.33])
    expected_sorted_imag_parts = np.array([1, 6.8802, -1, -6.8802])
    # check sorting
    assert np.allclose(np.real(d_sorted), expected_sorted_real_parts, atol=1e-10)
    assert np.allclose(np.imag(d_sorted), expected_sorted_imag_parts, atol=1e-10)
    # check eigenvectors
    for i in range(len(d_sorted)):
        Av = A @ v_sorted[:, i]
        lv = d_sorted[i] * v_sorted[:, i]
        assert np.allclose(Av, lv, atol=1e-10)

def test_sort_complex_eigenpairs_4d():
    A = np.array(
        [[-0.6, -1, 0, 0],
         [1,  -0.6, 0, 0],
        [0, 0, -.2, -3],
         [0, 0, 3,  -.2]]
    )
    d, v = np.linalg.eig(A)

    d_sorted, v_sorted = sort_complex_eigenpairs(d, v)

    expected_sorted_real_parts = np.array([-0.2, -0.6, -0.2, -0.6])
    expected_sorted_imag_parts = np.array([3, 1, -3, -1])
    # check sorting
    assert np.allclose(np.real(d_sorted), expected_sorted_real_parts, atol=1e-10)
    assert np.allclose(np.imag(d_sorted), expected_sorted_imag_parts, atol=1e-10)
    # check eigenvectors
    for i in range(len(d_sorted)):
        Av = A @ v_sorted[:, i]
        lv = d_sorted[i] * v_sorted[:, i]
        assert np.allclose(Av, lv, atol=1e-10)


if __name__ == '__main__':
    test_differentiation()
    test_ridge()
    test_ridge_constrained()
    test_ridge_with_or_without_scaling()
    #test_delay_embedding()
    #test_dimensionality_reduction()
    test_complex_polynomial_features()
    test_polynomial_features_pattern()
    test_fit_reduced_coords_and_parametrization()
    test_get_fit_ridge_parametric()