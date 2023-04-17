from scipy.integrate import solve_ivp
from ssmlearnpy.utils import ridge

from ssmlearnpy import SSMLearn

from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate
from ssmlearnpy.reduced_dynamics.normalform import NormalForm, NonlinearCoordinateTransform
from sklearn.preprocessing import PolynomialFeatures
from ssmlearnpy.utils.preprocessing import get_matrix , PolynomialFeaturesWithPattern
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

def test_delay_embedding():
    data = loadmat('../examples/brakereussbeam/data.mat')['data_BRB']

    TimeDIC = data[0,0].item()[0]
    DisplacementDIC = data[0,0].item()[1]
    TimeACC = data[0,0].item()[2]
    AccelerationACC = data[0,0].item()[3]
    Xmesh = data[0,0].item()[4]
    Units = data[0,0].item()[5]
    LocationACC = data[0,0].item()[6]
    PFFResultsACC = data[0,0].item()[7]
    ssm = SSMLearn(
        t = [TimeDIC.ravel()], 
        x = [DisplacementDIC], 
        ssm_dim=2, 
        dynamics_type = 'flow'
    )
    referenceData = loadmat('test_BRB_from_ssmlearn.mat')['yData']
    t_y, y, opts_embedding = coordinates_embedding(ssm.emb_data['time'], ssm.emb_data['observables'],
                                               imdim = ssm.ssm_dim, over_embedding = 5)
    assert np.allclose(t_y, referenceData[0,0])
    assert np.allclose(y, referenceData[0,1])

def test_dimensionality_reduction():
    reference_yData = loadmat('test_BRB_from_ssmlearn.mat')['yData']
    reference_etaData = loadmat('test_BRB_from_ssmlearn.mat')['etaData']
    
    ssm = SSMLearn(
        t = [reference_yData[0,0]], 
        x = [reference_yData[0,1]], 
        ssm_dim=2, 
        derive_embdedding = False,
        dynamics_type = 'flow'# use the embedding from the reference data
    )
    ssm.get_reduced_coordinates('linearchart')
    assert np.allclose(ssm.emb_data['reduced_coordinates'][0], reference_etaData[0,1])


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
    #print(PolynomialFeatures(degree=3, include_bias=False).fit(Y.T).powers_.shape)
    complexpoly = complex_polynomial_features(Y.T, degree=3)
    assert np.allclose(poly.T, complexpoly.T)

def test_prepare_normalform_transform_optimization():
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    ic_0 = np.array([0.1, 0])
    ic_1 = np.array([0.112, 0])
    sol_0 = solve_ivp(vectorfield, [t[0], t[-1]], ic_0, t_eval=t)
    sol_1 = solve_ivp(vectorfield, [t[0], t[-1]], ic_1, t_eval=t)
    trajectories = [sol_0.y - np.array([1,0]).reshape(-1,1), sol_1.y - np.array([1,0]).reshape(-1,1)]
    times = [t, t]
    X, y  = shift_or_differentiate(trajectories, times, 'flow') # get an estimate for the linear part
    # add the constraints to the fixed points explicitly
    constLHS = [[-1, 0]]
    constRHS = [[0, 0]]
    cons = [constLHS, constRHS]
    # Fit reduced model
    mdl = ridge.get_fit_ridge(X, y, poly_degree = 3, constraints = cons)
    linearPart = mdl.map_info['coefficients'][:,:2]
    _, v = np.linalg.eig(linearPart)
    linpartDiag = np.linalg.inv(v)@linearPart@v
    # v^{-1}@A@v is diagonal. transform the coordinates with v^{-1}.
    newcoord = [np.linalg.inv(v)@x for x in trajectories]
    nf, linerror, Transformation_normalform_polynomial_features, DTransformation_normalform_polynomial_features = prepare_normalform_transform_optimization( times, newcoord, linpartDiag)

    assert(DTransformation_normalform_polynomial_features[0].shape == (6, 1000))



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
    mdl = ridge.get_fit_ridge(X, y, poly_degree = 5, constraints = cons)
    linearPart = mdl.map_info['coefficients'][:,:2]
    w, v = np.linalg.eig(linearPart)
    #print(np.linalg.inv(v)@linearPart@v)
    linpartDiag = np.linalg.inv(v)@linearPart@v
    # v^{-1}@A@v is diagonal. transform the coordinates with v^{-1}.
    newcoord = [np.linalg.inv(v)@x for x in trajectories]
    dictTosave = {}
    dictTosave['trajectories'] = trajectories
    dictTosave['times'] = times
    dictTosave['newcoord'] = newcoord
    savemat('test_SSMLearn.mat', dictTosave)
    #print(newcoord[0][0,:]-np.conj(newcoord[0])[1,:]) is all zeros.
    n_unknowns_dynamics, n_unknowns_transformation, objectiv = create_normalform_transform_objective(times, newcoord, linpartDiag, degree = 3)
    np.random.seed(0)

    icfrommatlab = np.array([
  -0.046665229741644,
   0.088871461583745,
  -0.026512243121989,
  -0.081083271970684,
   0.323878856507470,
  -0.039156803945355,
  -0.097782606679839,
  -0.671912310868208,
  -1.346404263906166,
   0.222731265038940,
   0.014122867866736,
  -0.003328268772583,
  -0.009526614976568,
   0.902348990674412])
    ic_ = np.array([1, 0,1, 1,1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    res = minimize(objectiv, icfrommatlab, method='BFGS', options={'disp': True})
    d = unpack_optimized_coeffs(res.x, 1, n_unknowns_dynamics, n_unknowns_transformation)
    print(d['coeff_dynamics'])
    print(d['coeff_transformation'].shape)
    real_imag = np.array([np.real(d['coeff_dynamics']), np.imag(d['coeff_dynamics'])])
    assert(np.allclose(real_imag, np.array([-0.52697, -5.1146])))
        



if __name__ == '__main__':
    test_differentiation()
    test_ridge()
    test_ridge_constrained()
    test_ridge_with_or_without_scaling()
    #test_delay_embedding()
    #test_dimensionality_reduction()
    #test_complex_polynomial_features()
    test_polynomial_features_pattern()
    test_fit_reduced_coords_and_parametrization()
    test_get_fit_ridge_parametric()
    test_normalform_nonlinear_coeffs()
    test_normalform_lincombinations()
    test_normalform_resonance()
    #test_nonlinear_change_of_coords()
    #test_prepare_normalform_transform_optimization()
    #test_normalform_transform()