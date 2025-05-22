from scipy.integrate import solve_ivp
from ssmlearnpy.utils import ridge

from ssmlearnpy import SSMLearn

from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate
from ssmlearnpy.reduced_dynamics.normalform import (
    NormalForm,
    NonlinearCoordinateTransform,
    create_normalform_initial_guess,
    wrap_optimized_coefficients,
)
from sklearn.preprocessing import PolynomialFeatures
from ssmlearnpy.utils.preprocessing import (
    get_matrix,
    PolynomialFeaturesWithPattern,
    insert_complex_conjugate,
)
from ssmlearnpy.reduced_dynamics.normalform import (
    NonlinearCoordinateTransform,
    NormalForm,
    create_normalform_transform_objective_optimized,
    prepare_normalform_transform_optimization,
    unpack_optimized_coeffs,
)
from ssmlearnpy.geometry.coordinates_embedding import coordinates_embedding
from scipy.optimize import minimize, least_squares
import numpy as np
from ssmlearnpy.utils import ridge
from scipy.io import savemat, loadmat

from ssmlearnpy.utils.preprocessing import complex_polynomial_features


def vectorfield(t, x, r=1):
    # needed to test regression
    # damped duffing oscillator with fixed points at x=0 and x=+/-1.
    # x[0] is position, x[1] is velocity
    return np.array([x[1], x[0] - r * x[0] ** 3 - 0.1 * x[1]])


# test normal form transforms:
def test_normalform_nonlinear_coeffs():
    A = np.array(
        [[-0.6, -1],
         [1,  -0.6]]
    )
    nf = NormalForm(A)
    nonlinear_coeffs = nf._nonlinear_coeffs(degree=3)
    truecoeffs = np.array([[2, 1, 0, 3, 2, 1, 0], [0, 1, 2, 0, 1, 2, 3]])
    assert np.all(nonlinear_coeffs == truecoeffs)


def test_normalform_lincombinations_2d():
    # use the linear part of the above vectorfield at -1, 0:
    linearpart = np.array(
        [[-0.05 + 1j * 1.41332940251026, 0], [0, -0.05 - 1j * 1.41332940251026]]
    )
    nf = NormalForm(linearpart)
    lincombs = nf._eigenvalue_lin_combinations(degree=3)
    lamb = -0.05 + 1j * 1.41332940251026
    lambconj = -0.05 - 1j * 1.41332940251026
    true_lincombs = np.array(
        [
            [
                -lamb,
                -lambconj,
                lamb - 2 * lambconj,
                -2 * lamb,
                -2 * np.real(lamb),
                -2 * lambconj,
                lamb - 3 * lambconj,
            ],
            [
                lambconj - 2 * lamb,
                -lamb,
                -lambconj,
                lambconj - 3 * lamb,
                -2 * lamb,
                -2 * np.real(lamb),
                -2 * lambconj,
            ],
        ]
    )
    assert np.allclose(lincombs, true_lincombs)

def test_normalform_lincombinations_4d():
    # get the true values from matlab
    # lambda_1 = -0.1 + 1j 
    # lambda_2 = -0.33 + 6.8802j
    # MM = repmat(d_nf,1,size(Expmat_n,1))-repmat(transpose(Expmat_n*d_nf),k,1);
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(script_dir, 'test_lincombs.mat')

    lincombs_matlab_data = loadmat(test_file_path)
    lincombs_matlab_matrix = lincombs_matlab_data['MM']
    exponents_matlab = lincombs_matlab_data['Expmat_n']
    linearpart = np.diag(
        lincombs_matlab_data['d_nf'][:,0]
    )

    nf = NormalForm(linearpart)
    lincombs = nf._eigenvalue_lin_combinations(degree=3)
    assert np.allclose(nf._nonlinear_coeffs(degree = 3), exponents_matlab.T) # exponents match 
    assert np.allclose(lincombs, lincombs_matlab_matrix) # lin.combinations match



def test_normalform_resonance_2d():
    # use the linear part of the above vectorfield at -1, 0:
    linearpart = np.array(
        [[-0.05 - 1j * 1.41332940251026, 0], [0, -0.05 + 1j * 1.41332940251026]]
    )
    nf = NormalForm(linearpart)
    resonances = nf._resonance_condition(degree=3)
    whereistrue = np.where(resonances)
    assert np.allclose(whereistrue, [[0, 1], [4, 5]])




def test_normalform_resonance_4d():
    # lambda_1 = -0.1 + 1j 
    # lambda_2 = -0.33 + 6.8802j
    A = np.diag(
    [-0.1 +1.j,      -0.33+6.8802j,  -0.1 -1.j,      -0.33-6.8802j]
    )
    nf = NormalForm(A)
    resonances = nf._resonance_condition(degree=3, use_center_manifold_style=True)
    lincombs = nf._eigenvalue_lin_combinations(degree=3, use_center_manifold_style=True)
    exponents = nf._nonlinear_coeffs(degree = 3)
    assert np.sum(resonances) == 8 
    # lambda_1 resonance: 
    # lambda_1 + conj(lambda_1) + lambda_1
    # lambda_2 + conj(lambda_2) + lambda_1
    assert np.allclose(exponents[:, resonances[0,:]].T, [[2, 0, 1, 0], [1, 1, 0, 1]])
    # conj(lambda_1) resonance:
    # lambda_1 + conj(lambda_1) + conj(lambda_1)
    # lambda_2 + conj(lambda_2) + conj(lambda_1)
    assert np.allclose(exponents[:, resonances[2,:]].T, [[1, 0, 2, 0], [0, 1, 1, 1]])
    # lambda_2 resonance: 
    # lambda_2 + conj(lambda_2) + lambda_2
    # lambda_1 + conj(lambda_1) + lambda_2
    assert np.allclose(exponents[:, resonances[1,:]].T, [[1, 1, 1, 0], [0, 2, 0, 1]])
    # conj(lambda_2) resonance: 
    # lambda_2 + conj(lambda_2) + conj(lambda_2)
    # lambda_1 + conj(lambda_1) + conj(lambda_2)
    assert np.allclose(exponents[:, resonances[3,:]].T, [[1, 0, 1, 1], [0, 1, 0, 2]])
    
def test_nonlinear_change_of_coords():
    xx = np.linspace(-1, 1, 10)
    yy = np.linspace(-0.5, 0.5, 10)
    transformedx = xx + yy + xx**2 + yy**2
    vect = np.array([xx, yy])
    transformedy = xx - yy + xx**3 - yy * xx
    poly_features = PolynomialFeatures(degree=3, include_bias=False).fit(vect.T)
    vect_transform = poly_features.transform(vect.T)
    exponents = poly_features.powers_.T
    # print(exponents)
    # (2, 10)
    # (10, 9)
    # [[1 0 2 1 0 3 2 1 0]
    # [0 1 0 1 2 0 1 2 3]]
    transformationCoeffs = np.array(
        [[1, 1, 1, 0, 1, 0, 0, 0, 0], [1, -1, 0, -1, 0, 1, 0, 0, 0]]  # x + y +x^2 +y^2
    )  # x - y + x^3 - yx
    TF = NonlinearCoordinateTransform(
        2,
        3,
        transform_coefficients=transformationCoeffs,
        inverse_transform_coefficients=transformationCoeffs,
    )
    z = TF.inverse_transform(vect).T
    assert np.allclose(z[0, :], transformedx)
    assert np.allclose(z[1, :], transformedy)


def test_set_dynamics_and_transformation_structure():
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    ic_0 = np.array([0.1, 0])
    ic_1 = np.array([0.112, 0])
    sol_0 = solve_ivp(vectorfield, [t[0], t[-1]], ic_0, t_eval=t)
    sol_1 = solve_ivp(vectorfield, [t[0], t[-1]], ic_1, t_eval=t)
    trajectories = [
        sol_0.y - np.array([1, 0]).reshape(-1, 1),
        sol_1.y - np.array([1, 0]).reshape(-1, 1),
    ]
    times = [t, t]
    X, y = shift_or_differentiate(
        trajectories, times, "flow"
    )  # get an estimate for the linear part
    # add the constraints to the fixed points explicitly
    constLHS = [[-1, 0]]
    constRHS = [[0, 0]]
    cons = [constLHS, constRHS]
    # Fit reduced model
    mdl = ridge.get_fit_ridge(X, y, poly_degree=3, constraints=cons)
    linearPart = mdl.map_info["coefficients"][:, :2]
    nf = NormalForm(linearPart)
    nf.set_dynamics_and_transformation_structure("flow", 3)
    truestructure = [False, False, False, False, True, False, False]
    for s, t in zip(nf.dynamics_structure[0,:], truestructure):
        assert s == t


def test_prepare_normalform_transform_optimization():
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    ic_0 = np.array([0.1, 0])
    ic_1 = np.array([0.112, 0])
    sol_0 = solve_ivp(vectorfield, [t[0], t[-1]], ic_0, t_eval=t)
    sol_1 = solve_ivp(vectorfield, [t[0], t[-1]], ic_1, t_eval=t)
    trajectories = [
        sol_0.y - np.array([1, 0]).reshape(-1, 1),
        sol_1.y - np.array([1, 0]).reshape(-1, 1),
    ]
    times = [t, t]
    X, y = shift_or_differentiate(
        trajectories, times, "flow"
    )  # get an estimate for the linear part
    # add the constraints to the fixed points explicitly
    constLHS = [[-1, 0]]
    constRHS = [[0, 0]]
    cons = [constLHS, constRHS]
    # Fit reduced model
    mdl = ridge.get_fit_ridge(X, y, poly_degree=3, constraints=cons)
    linearPart = mdl.map_info["coefficients"][:, :2]
    (
        _,
        nf,
        linerror,
        Transformation_normalform_polynomial_features,
        DTransformation_normalform_polynomial_features,
    ) = prepare_normalform_transform_optimization(times, trajectories, linearPart)


def test_create_normalform_initial_guess():
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    ic_0 = np.array([0.1, 0])
    ic_1 = np.array([0.112, 0])
    sol_0 = solve_ivp(vectorfield, [t[0], t[-1]], ic_0, t_eval=t)
    sol_1 = solve_ivp(vectorfield, [t[0], t[-1]], ic_1, t_eval=t)
    trajectories = [
        sol_0.y - np.array([1, 0]).reshape(-1, 1),
        sol_1.y - np.array([1, 0]).reshape(-1, 1),
    ]
    times = [t, t]
    X, y = shift_or_differentiate(
        trajectories, times, "flow"
    )  # get an estimate for the linear part
    # add the constraints to the fixed points explicitly
    constLHS = [[-1, 0]]
    constRHS = [[0, 0]]
    cons = [constLHS, constRHS]
    # Fit reduced model
    mdl = ridge.get_fit_ridge(X, y, poly_degree=3, constraints=cons)
    linearPart = mdl.map_info["coefficients"][:, :2]
    (
        _,
        nf,
        linerror,
        Transformation_normalform_polynomial_features,
        DTransformation_normalform_polynomial_features,
    ) = prepare_normalform_transform_optimization(times, trajectories, linearPart)
    initial_guess = create_normalform_initial_guess(mdl, nf)
    assert len(initial_guess) == 14


def test_fit_inverse_2d(show_plot = False):
    # (eta_1, eta_2) = T(z, conj(z)) computes the reduced coordinates from the nf. coordinates
    np.random.seed(1)
    x = np.random.rand(1, 400) + 1j * np.random.rand(1, 400)
    transformCoeffs = np.random.rand(1, 20) + 1j * np.random.rand(1, 20)
    transformCoeffs = transformCoeffs * 0.01
    transformCoeffs[0,0] = 1
    transformCoeffs[0,1] = 0 # ensure identity jacobian

    xx = insert_complex_conjugate(x)
    TF = NonlinearCoordinateTransform(
        2, 5, inverse_transform_coefficients=transformCoeffs, linear_transform=np.eye(2)
    )
    
    trajectory = [xx]
    coeffs, inverse = ridge.fit_inverse(
        TF.inverse_transform, trajectory, 5, near_identity=True
    )
    image = TF.inverse_transform(trajectory)
    TF.set_transform_coefficients(coeffs)
    inversetransformed = [inverse(t) for t in image]
    if show_plot:
        plt.figure()

        plt.plot(xx[0, :].real, image[0][0,:],'.')
        plt.plot(xx[0, :].real, xx[0, :], '-',  c='black')
        plt.xlabel('Real z_1')
        plt.ylabel('eta_1')
        plt.figure()
        plt.plot(xx[0, :].imag, image[0][1,:],'.')
        plt.plot(xx[0, :].imag, xx[0, :].imag, '-', c='black')
        plt.xlabel('Im z_1')
        plt.ylabel('eta_2')

        plt.figure()
        plt.title('$T^{-1}(T(z,conj(z)))==(z, conj(z))$')
        plt.plot(xx[0, :].real, inversetransformed[0][0,:].real,'.')
        plt.plot(xx[0, :].real, xx[0, :].real,'-', c='black')

        plt.figure()
        plt.title('$T^{-1}(T(z,conj(z)))==(z, conj(z))$')
        plt.plot(xx[0, :].imag, inversetransformed[0][0,:].imag,'.')
        plt.plot(xx[0, :].imag, xx[0, :].imag,'-', c='black')

        plt.show()
    assert np.allclose(xx[0,:], inversetransformed[0][0,:], atol = 1e-2)
import matplotlib.pyplot as plt


def test_fit_inverse_4d(show_plot = False):
    np.random.seed(1)
    x = np.random.rand(2, 400) + 1j * np.random.rand(2, 400)

    transformCoeffs = np.random.rand(2, 34) + 1j * np.random.rand(2, 34)
    transformCoeffs = transformCoeffs * 0.01
    transformCoeffs[0,0] = 1.

    transformCoeffs[0,1] = 0

    transformCoeffs[1,0] = 0
    transformCoeffs[1,1] = 1

    xx = insert_complex_conjugate(x)
    TF = NonlinearCoordinateTransform(
        4, 3, inverse_transform_coefficients=transformCoeffs, linear_transform=np.eye(4)
    )


    trajectory = [xx]
    coeffs_, inverse = ridge.fit_inverse(
        TF.inverse_transform, trajectory, 3, near_identity=True
    )
    image = TF.inverse_transform(trajectory)
    TF.set_transform_coefficients(coeffs_)
    inversetransformed = [inverse(t) for t in image]
    if show_plot:
        plt.figure()

        plt.plot(xx[0, :].real, image[0][0,:],'.')
        plt.plot(xx[0, :].real, xx[0, :], '-',  c='black')
        plt.xlabel('Real z_1')
        plt.ylabel('eta_1')
        plt.figure()
        plt.plot(xx[0, :].imag, image[0][1,:],'.')
        plt.plot(xx[0, :].imag, xx[0, :].imag, '-', c='black')
        plt.xlabel('Im z_1')
        plt.ylabel('eta_2')

        plt.figure()
        plt.title('$T^{-1}(T(z,conj(z)))==(z, conj(z))$')
        plt.plot(xx[0, :].real, inversetransformed[0][0,:].real,'.')
        plt.plot(xx[0, :].real, xx[0, :].real,'-', c='black')

        plt.figure()
        plt.title('$T^{-1}(T(z,conj(z)))==(z, conj(z))$')
        plt.plot(xx[0, :].imag, inversetransformed[0][0,:].imag,'.')
        plt.plot(xx[0, :].imag, xx[0, :].imag,'-', c='black')

        plt.show()
    assert np.allclose(xx[0,:], inversetransformed[0][0,:], atol = 1e-2)
    assert np.allclose(xx[1,:], inversetransformed[0][1,:], atol = 1e-2)




def test_normalform_transform():
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    ic_0 = np.array([0.1, 0])
    ic_1 = np.array([0.112, 0])
    sol_0 = solve_ivp(vectorfield, [t[0], t[-1]], ic_0, t_eval=t)
    sol_1 = solve_ivp(vectorfield, [t[0], t[-1]], ic_1, t_eval=t)
    trajectories = [
        sol_0.y - np.array([1, 0]).reshape(-1, 1),
        sol_1.y - np.array([1, 0]).reshape(-1, 1),
    ]
    times = [t, t]
    X, y = shift_or_differentiate(
        trajectories, times, "flow"
    )  # get an estimate for the linear part
    # print(X[0].shape)
    # add the constraints to the fixed points explicitly
    constLHS = [[-1, 0]]
    constRHS = [[0, 0]]
    cons = [constLHS, constRHS]
    # Fit reduced model
    mdl = ridge.get_fit_ridge(X, y, poly_degree=5, constraints=cons)
    linearPart = mdl.map_info["coefficients"][:, :2]

    nf, n_unknowns_dynamics, n_unknowns_transformation, objectiv = (
        create_normalform_transform_objective_optimized(times, trajectories, linearPart, degree=5)
    )
    initial_guess = create_normalform_initial_guess(mdl, nf)
    # we can use the special sturcutre of the loss and use scipy.optimize.least_suqares()
    res = least_squares(objectiv, initial_guess)
    d = unpack_optimized_coeffs(
        res.x, 1, nf, n_unknowns_dynamics, n_unknowns_transformation
    )
    trf, dyn = wrap_optimized_coefficients(
        1, nf, 5, d, find_inverse=True, trajectories=trajectories, raw_coeffs=res.x, near_identity=True
    )
    image = trf.inverse_transform(trajectories)
    inversetransformed = [trf.transform(t) for t in image]
    print([inversetransformed])


def test_misc_conjugates():
    np.random.seed(0)
    x = np.random.rand(1, 3) + 1j * np.random.rand(1, 3)
    y = insert_complex_conjugate(x)
    coeffs = np.random.rand(1, 20) + 1j * np.random.rand(1, 20)
    nonlin_features = complex_polynomial_features(y.T, degree=5)
    conj_second = insert_complex_conjugate(coeffs @ nonlin_features.T)
    coeffs_with_conj = insert_complex_conjugate(coeffs)
    conj_first = coeffs_with_conj @ nonlin_features.T

    # print(np.allclose(conj_first, np.conj(conj_second)))
    x = np.random.rand(3) + 1j * np.random.rand(3)

    XX = np.array([x, np.conj(x), x**2, x * np.conj(x), np.conj(x) ** 2])
    coeffs = np.random.rand(1, 5) + 1j * np.random.rand(1, 5)
    conj_second = insert_complex_conjugate(coeffs @ XX)
    coeffs_with_conj = insert_complex_conjugate(coeffs)
    # conj_first = coeffs_with_conj@XX

    conj_first = np.concatenate((coeffs @ XX, np.conj(coeffs) @ np.conj(XX)), axis=0)

    assert np.allclose(conj_first, conj_second)


def test_4d_manifold_fit_convergence():
    from ssmlearnpy.main.main import SSMLearn

    t = np.linspace(0, 100, 10000)
    signal = np.exp(-0.1*t)*np.sin(t) +  np.exp(-0.37*t)*np.sin(3.2 * t)
    dim = 4
    xData = [[t, np.array([signal])]]
    t_y, y, _ = coordinates_embedding(
        [xData[0][0]],
        [xData[0][1]],
        imdim=dim,
    )
    t = t_y[0]
    ssm = SSMLearn(
        t=t_y,
        x=y,
        derive_embdedding=False,
        ssm_dim=dim,
        dynamics_type="flow",
        dynamics_structure="normalform",
    )
    ssm.get_parametrization()
    ssm.get_reduced_dynamics(
        normalform_args={
            "degree": 3,
            "do_scaling": False,
            "tolerance": None,
            "ic_style": "zero",
            "method": "trf",
            "jac": "3-point",
            "max_iter": 1000,
            "use_center_manifold_style": True,
        }
    )
    assert ssm.reduced_dynamics.map_info['normalform_transformation'] is not None



def test_2d_manifold_fit_normal_form_dynamics():
    from ssmlearnpy.main.main import SSMLearn

    t = np.linspace(0, 100, 10000)
    signal = np.exp(-0.1*t)*np.sin(t)
    dim = 2
    xData = [[t, np.array([signal])]]
    t_y, y, _ = coordinates_embedding(
        [xData[0][0]],
        [xData[0][1]],
        imdim=dim,
    )
    t = t_y[0]
    ssm = SSMLearn(
        t=t_y,
        x=y,
        derive_embdedding=False,
        ssm_dim=dim,
        dynamics_type="flow",
        dynamics_structure="normalform",
    )
    ssm.get_parametrization()
    ssm.get_reduced_dynamics(
        normalform_args={
            "degree": 3,
            "do_scaling": True,
            "tolerance": None,
            "ic_style": "zero",
            "method": "trf",
            "jac": "3-point",
            "max_iter": 1000,
            "use_center_manifold_style": True,
        }
    )
    reduced_traj = ssm.emb_data["reduced_coordinates"][0]
    nf_gt = ssm.normalform_transformation.inverse_transform(reduced_traj)
    assert np.allclose(nf_gt[0,:], np.conj(nf_gt[1,:])) # check structure of the coordinates
    pred_nf = solve_ivp(
        ssm.reduced_dynamics.map_info["vectorfield"],
        [t[0], t[-1]],
        nf_gt[:, 0],
        t_eval=t,
        method="RK45",
    ).y
    # prediction errors
    assert np.allclose(pred_nf[0,:], nf_gt[0,:], atol = 1e-3) # comparable to matlab
    
    


def test_4d_manifold_fit_normal_form_dynamics():
    from ssmlearnpy.main.main import SSMLearn

    t = np.linspace(0, 100, 10000)
    signal = np.exp(-0.1*t)*np.sin(t) +  np.exp(-0.37*t)*np.sin(3.2 * t)
    dim = 4
    xData = [[t, np.array([signal])]]
    t_y, y, _ = coordinates_embedding(
        [xData[0][0]],
        [xData[0][1]],
        imdim=dim,
    )
    t = t_y[0]
    ssm = SSMLearn(
        t=t_y,
        x=y,
        derive_embdedding=False,
        ssm_dim=dim,
        dynamics_type="flow",
        dynamics_structure="normalform",
    )
    ssm.get_parametrization()
    ssm.get_reduced_dynamics(
        normalform_args={
            "degree": 3,
            "do_scaling": True,
            "tolerance": None,
            "ic_style": "zero",
            "method": "trf",
            "jac": "3-point",
            "max_iter": 1000,
            "use_center_manifold_style": True,
        }
    )
    reduced_traj = ssm.emb_data["reduced_coordinates"][0]
    nf_gt = ssm.normalform_transformation.inverse_transform(reduced_traj)
    assert np.allclose(nf_gt[0,:], np.conj(nf_gt[2,:])) # check structure of the coordinates
    pred_nf = solve_ivp(
        ssm.reduced_dynamics.map_info["vectorfield"],
        [t[0], t[-1]],
        nf_gt[:, 0],
        t_eval=t,
        method="RK45",
    ).y
    # prediction errors
    # slow component is very accurate
    assert np.allclose(pred_nf[0,:], nf_gt[0,:], atol = 1e-4) # comparable to matlab
    assert np.allclose(pred_nf[1,:], nf_gt[1,:], atol = 3e-3) # comparable to matlab
    

def test_4d_manifold_fit_normal_form_complete_pred(show_plot = False):
    from ssmlearnpy.main.main import SSMLearn
    t = np.linspace(0, 100, 10000)
    signal = np.exp(-0.1*t)*np.sin(t) +  np.exp(-0.37*t)*np.sin(3.2 * t)
    dim = 4
    xData = [[t, np.array([signal])]]
    t_y, y, _ = coordinates_embedding(
        [xData[0][0]],
        [xData[0][1]],
        imdim=dim,
    )
    t = t_y[0]
    ssm = SSMLearn(
        t=t_y,
        x=y,
        derive_embdedding=False,
        ssm_dim=dim,
        dynamics_type="flow",
        dynamics_structure="normalform",
    )
    ssm.get_parametrization()
    ssm.get_reduced_dynamics(
        normalform_args={
            "degree": 3,
            "do_scaling": True,
            "tolerance": None,
            "ic_style": "zero",
            "method": "trf",
            "jac": "3-point",
            "max_iter": 1000,
            "use_center_manifold_style": True,
        }
    )
    observable_traj = ssm.emb_data["observables"][0]

    reduced_traj = ssm.emb_data["reduced_coordinates"][0]
    nf_gt = ssm.normalform_transformation.inverse_transform(reduced_traj)
    pred_nf = solve_ivp(
        ssm.reduced_dynamics.map_info["vectorfield"],
        [t[0], t[-1]],
        nf_gt[:, 0],
        t_eval=t,
        method="RK45",
    ).y
    pred_reduced = ssm.normalform_transformation.transform(pred_nf).real
    pred_obs = ssm.decoder.predict(pred_reduced.T).T

    if show_plot: 
        plt.figure()
        plt.title('Normal form coords: z1')
        plt.plot(nf_gt[0,:].real, nf_gt[0,:].imag, '-', label = 'True')
        plt.plot(pred_nf[0,:].real, pred_nf[0,:].imag, '--', label = 'Pred')
        plt.legend()
        
        plt.figure()
        plt.title('Normal form coords: z2')
        plt.plot(nf_gt[1,:].real, nf_gt[1,:].imag, '-', label = 'True')
        plt.plot(pred_nf[1,:].real, pred_nf[1,:].imag, '--', label = 'Pred')
        plt.legend()
        
        plt.figure()
        plt.title('Reduced coords')
        plt.plot(reduced_traj[0,:], reduced_traj[1,:], '-', label = 'True')
        plt.plot(pred_reduced[0,:], pred_reduced[1,:], '--', label = 'Pred')
        plt.legend()

        plt.figure()
        plt.title('Observable')
        plt.plot(t, observable_traj[0], '-', label="True")
        plt.plot(t, pred_obs[0], '--', label="Pred")
        plt.legend()

        plt.show()
    # prediction errors
    
    assert np.allclose(reduced_traj[0,:], pred_reduced[0,:], atol = 1e-2)
    assert np.allclose(reduced_traj[1,:], pred_reduced[1,:], atol = 1e-2)
    assert np.allclose(observable_traj[0], pred_obs[0], atol = 1e-2)






#if __name__ == "__main__":
    # test_misc_conjugates()
    # test_normalform_nonlinear_coeffs()
    # test_normalform_lincombinations()
    # test_normalform_resonance()
    # test_nonlinear_change_of_coords()
    # test_set_dynamics_and_transformation_structure()
    # test_prepare_normalform_transform_optimization()
    # test_create_normalform_initial_guess()

    # test_fit_inverse()
    # test_normalform_transform()
    #test_fit_inverse_2d(show_plot=False)
    #test_fit_inverse_4d(show_plot=False)
    #test_4d_manifold_fit_normal_form_complete_pred(show_plot=True)
    