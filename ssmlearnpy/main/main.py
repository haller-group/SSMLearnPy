import logging

import numpy as np

from ssmlearnpy.geometry.coordinates_embedding import coordinates_embedding
from ssmlearnpy.geometry.dimensionality_reduction import reduce_dimensions
from ssmlearnpy.geometry.encode_decode import decode_geometry
from ssmlearnpy.geometry.encode_decode import encode_geometry

from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate
from ssmlearnpy.reduced_dynamics.advector import advect

from ssmlearnpy.utils.compute_errors import compute_errors
from ssmlearnpy.utils.ridge import (
    get_fit_ridge,
    fit_reduced_coords_and_parametrization,
    get_fit_ridge_parametric,
)
from ssmlearnpy.utils.ridge import get_matrix
from ssmlearnpy.utils.file_handler import get_vectors
from ssmlearnpy.utils.plots import compute_surface
import ssmlearnpy.reduced_dynamics.normalform as normalform
from scipy.optimize import minimize, least_squares
from copy import deepcopy
from typing import Literal
import ipdb


logger = logging.getLogger("SSMLearn")


class SSMLearn:
    """
    Main class to perform SSM-based model reduction of dynamical systems.
    Contains the trajectory data, the reduced coordinates and the reduced dynamics,
    as well as methods to map the reduced coordinates to the original ones (encode-decode).

    The class should be initialized with training data that comes from a dynamical system,
    either given as an ODE or an iterated mapping.

    The class can also be initialized with a path to a file containing the training data, saved as csv files.

    Helper functions to do full predictions are also implemented.
        predict_geometry(): given the reduced coordinates, predict the full system coordinates
        predict_reduced_dynamics(): advect the reduced coordiantes in time, either by numerical solution of an ODE or iterated mapping
        predict(): given the reduced coordinates, advect them in time and then predict the full system coordinates

    Parameters:
        - t: list of time of the different trajectories, shape=(n_trajectories,)
        - x: list of trajectories, shape=(n_trajectories, )
        - params: list of parameters of the different trajectories, shape=(n_trajectories, n_params)
        - reduced_coordinates (optional): list of reduced coordinates of the different trajectories, shape=(n_trajectories, n_reduced_coordinates)
        - derive_embdedding (optional): if True, delay embedding is used to derive the full system coordinates
        - ssm_dim: Dimension of the SSM (spectral submanifold)
        - coordinates_embeddings_args: dictionary of arguments to pass to the coordinates embedding function, such as over_embedding
        - dynamics_type: type of dynamics to use for the reduced dynamics. Can be 'flow' or 'map'
        - dynamics_structure: structure of the reduced dynamics: can be either 'generic' or 'normalform. If normal form is selected, then the reduced dynamics is computed as an extended normal form with a sparse structure. Otherwise, the reduced dynamics is computed as a polynomial map with all coefficients free to be fitted.
        - error_metric: metric to use to compute the error between the full and reduced system

    Attributes:
        - input_data: dictionary containing the raw input data
        - emb_data: dictionary containing the delay-embedded input data, if derive_embdedding is True.
                    Otherwise, emb_data['observables'] = input_data['observables']
                    emb_data['time'] contains the times at which the trajectories are recorder
                    emb_data['params'] contains the parameters of the trajectories
                    emb_data['reduced_coordinates'] contains the reduced coordinates of the trajectories.
                            Can be called at initialization, but can be computed from emb_data['observables']
        - decoder: mapping from the reduced coordinates to the full system coordinates (from emb_data['reduced_coordinates'] to emb_data['observables'])
        - encoder: mapping from the full system coordinates to the reduced coordinates (from emb_data['observables'] to emb_data['reduced_coordinates'])
        - reduced_dynamics: reduced dynamics of the system, either a flow or a map
        - geometry_predictions: dictionary containing the predictions of the full system coordinates
        - reduced_dynamics_predictions: dictionary containing the predictions of the reduced coordinates
        - predictions: dictionary containing the joint predictions: predictions of the reduced dynamics followed by prediction of the geometry
        - normalform_transformation: NonlinearCoordinateTransform object containing the transformation from the original coordinates to the normal form coordinates. It is fitted when get_reduced_dynamics() is called and dynamics_structure = 'normalform'.
    """

    def __init__(
        self,
        path_to_trajectories: str = None,
        t: list = None,
        x: list = None,
        offset: np.ndarray = None,
        params: list = None,
        reduced_coordinates: list = None,
        derive_embdedding=True,
        ssm_dim: int = None,
        coordinates_embeddings_args: dict = {},
        dynamics_type="flow",
        dynamics_structure: str = Literal["generic", "normalform"],
        error_metric="NTE",
    ) -> None:
        self.input_data = {}

        if path_to_trajectories:
            self.input_data["time"], self.input_data["observables"] = self.import_data(
                path_to_trajectories
            )

        elif t and x:
            self.input_data["time"] = t
            self.input_data["observables"] = x
            self.input_data["offset"] = offset

        else:
            raise RuntimeError(
                (
                    f"Not enought parameters specified. Found: path_to_trajectories={path_to_trajectories}"
                    + f"t={t}, x={x}. Please either set the path to the trajectories file or pass them to the class"
                )
            )

        self.check_inputs()
        self.emb_data = {}

        if derive_embdedding and not reduced_coordinates:
            logger.info("Getting coordinates embeddings")
            self.emb_data = {}
            self.emb_data["time"], self.emb_data["observables"], embedding_info = (
                coordinates_embedding(
                    self.input_data["time"],
                    self.input_data["observables"],
                    ssm_dim,
                    offset=self.input_data["offset"],
                    **coordinates_embeddings_args,
                )
            )
            self.emb_data["offset"] = embedding_info["embedded_offset"]
        else:
            self.emb_data["time"] = self.input_data["time"]
            self.emb_data["observables"] = self.input_data["observables"]
            self.emb_data["offset"] = self.input_data["offset"]

        self.emb_data["params"] = params
        self.emb_data["reduced_coordinates"] = reduced_coordinates

        self.encoder = None
        self.decoder = None
        self.reduced_dynamics = None
        self.dynamics_type = dynamics_type
        self.dynamics_structure = dynamics_structure
        self.error_metric = error_metric
        self.geometry_predictions = {}
        self.reduced_dynamics_predictions = {}
        self.predictions = {}
        self.ssm_dim = ssm_dim
        self.normalform_transformation = None

    @staticmethod
    def import_data(path):
        x, t = get_vectors(path)
        return x, t

    def check_inputs(self):
        n_traj_time = len(self.input_data["time"])
        assert n_traj_time == len(self.input_data["observables"])

    def get_reduced_coordinates(self, method="linearchart", **keyargs) -> None:
        """
        Compute the reduced coordinates of the trajectories using the given method.
        method: can be 'basic', 'linearchart' or 'fastssm'.
            basic: use the first ssm_dim coordinates of the delay-embedded trajectories
            linearchart: perform an SVD and keep the first ssm_dim coordinates
            fastssm: same as linearchart. We keep the name fastssm to be consistent with the matlab implementation
        If the reduced coordinates have already been computed, skip.
        """
        self.encoder = reduce_dimensions(method=method, n_dim=self.ssm_dim, **keyargs)
        if self.emb_data["reduced_coordinates"] is None:
            self.encoder.fit(self.emb_data["observables"], self.emb_data["offset"])
            self.emb_data["reduced_coordinates"] = self.encoder.predict(
                self.emb_data["observables"]
            )
        else:
            logger.info("Reduced coordinated already passed to SSMLearn, skipping.")

    def get_parametrization(self, **regression_args) -> None:
        if (
            self.emb_data["reduced_coordinates"] is not None
        ):  # reduced coordinates have been precomputed
            if self.emb_data["params"] is not None:
                self.decoder = get_fit_ridge_parametric(
                    self.emb_data["reduced_coordinates"],
                    self.emb_data["observables"],
                    self.emb_data["params"],
                    **regression_args,
                )
            else:
                self.decoder = get_fit_ridge(
                    self.emb_data["reduced_coordinates"],
                    self.emb_data["observables"],
                    self.emb_data["offset"],
                    **regression_args,
                )
        else:
            self.encoder, self.decoder = fit_reduced_coords_and_parametrization(
                self.emb_data["observables"], self.ssm_dim, **regression_args
            )  # get both decoder and encoder
            self.emb_data["reduced_coordinates"] = [
                self.encoder.predict(trajectory)
                for trajectory in self.emb_data["observables"]
            ]

        return

    def encode(self, x):
        """wrapper for encoder.predict. Expects a trajectory of shape (n_features, n_samples)
        returns the reduced coordinates of shape (n_dim, n_samples)
        """
        return self.encoder.predict(x)

    def decode(self, y):
        """wrapper for decoder.predict. Expects a reduced trajectory of shape (n_dim, n_samples)
        returns the full trajectory of shape (n_features, n_samples)
        """
        out = self.decoder.predict(y.T).T
        if self.emb_data["offset"] is not None:
            out += self.emb_data["offset"].reshape(-1, 1)
        return out

    def get_surface(
        self,
        idx_reduced_coordinates=[1, 2],
        idx_observables=1,
        surf_margin=10,
        mesh_step=100,
    ) -> None:

        x_data = get_matrix(self.emb_data["reduced_coordinates"])
        if self.ssm_dim == 2:
            U, _, _ = np.linalg.svd(x_data, full_matrices=True)
            max_vals = (1 + surf_margin / 100) * np.amax(np.matmul(U.T, x_data), axis=1)
            transf_mesh_generation = np.matmul(U, np.diag(max_vals))
        else:
            raise NotImplementedError((f"Not implemented."))

        surface_dict = compute_surface(
            surface_function=self.decode,
            idx_reduced_coordinates=idx_reduced_coordinates,
            transf_mesh_generation=transf_mesh_generation,
            idx_observables=idx_observables,
            mesh_step=mesh_step,
        )

        return surface_dict

    def get_reduced_dynamics(
        self,
        normalform_args={
            "degree": 3,
            "do_scaling": True,
            "tolerance": None,
            "ic_style": "random",
            "max_iter": 1000,
            "method": "lm",
            "jac": "2-point",
        },
        **regression_args,
    ) -> None:
        """Compute the reduced dynamics from the data supplied to the class.

        Paramters:
            normalform_args (dict, optional): Contains all normal form related arguments. Defaults to {}.
                - normalform_args['degree']
                - normalform_args['do_scaling']
                - normalform_args['tolerance']
                - normalform_args['ic_style']: Can be random, informed or zero. If informed, then an initial guess is computed from the initial regression.
                - normalform_args['max_iter']: Maximum number of iterations for the optimization
                - normalform_args['method']: method to be passed to the least_squares function
                - normalform_args['jac']: jacobian to be passed to the least_squares function
                - normalform_args['use_center_manifold_style']: if True, then the center manifold style is used to compute the normal form transformation.
        """
        X, y = shift_or_differentiate(
            self.emb_data["reduced_coordinates"],
            self.emb_data["time"],
            self.dynamics_type,
        )
        if self.emb_data["params"] is not None:
            self.reduced_dynamics = get_fit_ridge_parametric(
                X, y, self.emb_data["params"], **regression_args
            )
        else:
            self.reduced_dynamics = get_fit_ridge(X, y, **regression_args)
        linear_part = self.reduced_dynamics.map_info["coefficients"][:, : X[0].shape[0]]
        d, v = np.linalg.eig(linear_part)
        self.reduced_dynamics.map_info["eigenvalues_linear_part"] = d
        self.reduced_dynamics.map_info["eigenvectors_linear_part"] = v
        self.eigenvalues = d
        self.eigenvectors = v
        self.reduced_coords_dynamics = deepcopy(self.reduced_dynamics)

        if (
            self.dynamics_structure == "normalform" and d.dtype == complex
        ):  # compute the normal form transformation after an initial guess has been computed
            ndofs = int(linear_part.shape[0] / 2)
            if self.ssm_dim % 2 != 0:
                raise NotImplementedError(
                    (f"Normal form transformation not implemented for odd dimensions.")
                )
            (
                nf_object,
                # objective_dict,
                n_unknowns_dynamics,
                n_unknowns_transformation,
                objective,
            ) = normalform.create_normalform_transform_objective_optimized(
                self.emb_data["time"],
                self.emb_data["reduced_coordinates"],
                linear_part,
                degree=normalform_args["degree"],
                do_scaling=normalform_args["do_scaling"],
                tolerance=normalform_args["tolerance"],
                use_center_manifold_style=normalform_args["use_center_manifold_style"],
            )

            # create 3 kinds of initial guesses:
            if normalform_args["ic_style"] == "random":
                initial_guess = np.random.rand(
                    (n_unknowns_dynamics + n_unknowns_transformation) * 2
                )  # both real and imaginary parts
            elif normalform_args["ic_style"] == "informed":
                initial_guess = normalform.create_normalform_initial_guess(
                    self.reduced_dynamics, nf_object
                )
            elif normalform_args["ic_style"] == "zero":
                initial_guess = np.zeros(
                    (n_unknowns_dynamics + n_unknowns_transformation) * 2
                )
            # TODO remove
            # initial_guess = (
            #     np.ones((n_unknowns_dynamics + n_unknowns_transformation) * 2) * 0.1
            # )

            res = least_squares(
                objective,
                initial_guess,
                method=normalform_args["method"],
                jac=normalform_args["jac"],
                max_nfev=normalform_args["max_iter"],
                # max_nfev=1,
            )
            if not res.success:
                print(f"Optimization did not converge. Message: {res.message}")
                logger.error((f"Optimization did not converge. Message: {res.message}"))
            else:
                # ipdb.set_trace()
                logger.info((f"Optimization converged. Message: {res.message}"))
                print(f"Optimization converged. Message: {res.message}")
                print(f"Number of iterations: {res.nfev}")
                logger.info(f"Number of iterations: {res.nfev}")
            unpacked_coeffs = normalform.unpack_optimized_coeffs(
                res.x, ndofs, nf_object, n_unknowns_dynamics, n_unknowns_transformation
            )
            transformation, dynamics = normalform.wrap_optimized_coefficients(
                ndofs,
                nf_object,
                normalform_args["degree"],
                unpacked_coeffs,
                find_inverse=True,
                trajectories=self.emb_data["reduced_coordinates"],
            )
            # self.normalform_reduced_dynamics = dynamics
            # self.normalform_reduced_dynamics.map_info["normalform_transformation"] = (
            #     transformation
            # )
            self.normalform_transformation = transformation
            self.reduced_dynamics = dynamics
            self.reduced_dynamics.map_info["normalform_transformation"] = transformation
        return

    def predict_geometry(self, idx_trajectories=0, t=[], x=[], x_reduced=[]) -> None:
        if bool(t) is False:
            if idx_trajectories == 0:
                t_to_predict = self.emb_data["time"]
                x_reduced = self.emb_data["reduced_coordinates"]
                x_to_predict = self.emb_data["observables"]
            else:
                t_to_predict = [self.emb_data["time"][i] for i in idx_trajectories]
                x_reduced = [
                    self.emb_data["reduced_coordinates"][i] for i in idx_trajectories
                ]
                x_to_predict = [
                    self.emb_data["observables"][i] for i in idx_trajectories
                ]

            x_predict = decode_geometry(self.decode, x_reduced)

            prediction_errors = compute_errors(
                reference=x_to_predict, prediction=x_predict, metric=self.error_metric
            )
            self.geometry_predictions = {}
            self.geometry_predictions["time"] = t_to_predict
            self.geometry_predictions["reduced_coordinates"] = x_predict
            self.geometry_predictions["observables"] = x_predict
            self.geometry_predictions["errors"] = prediction_errors
        else:
            if bool(x_reduced) is False:
                x_reduced = encode_geometry(self.encode, x)

            x_predict = decode_geometry(self.decode, x_reduced)

            prediction_errors = compute_errors(
                reference=x, prediction=x_predict, metric=self.error_metric
            )
            geometry_predictions = {}
            geometry_predictions["time"] = t
            geometry_predictions["reduced_coordinates"] = x_reduced
            geometry_predictions["observables"] = x_predict
            geometry_predictions["errors"] = prediction_errors
            return geometry_predictions

    def predict_reduced_dynamics(self, idx_trajectories=0, t=[], x_reduced=[]) -> None:
        if bool(t) is False:
            if idx_trajectories == 0:
                t_to_predict = self.emb_data["time"]
                x_to_predict = self.emb_data["reduced_coordinates"]
            else:
                t_to_predict = [self.emb_data["time"][i] for i in idx_trajectories]
                x_to_predict = [
                    self.emb_data["reduced_coordinates"][i] for i in idx_trajectories
                ]

            t_predict, x_predict = advect(
                dynamics=self.reduced_dynamics.predict,
                t=t_to_predict,
                x=x_to_predict,
                dynamics_type=self.dynamics_type,
            )

            prediction_errors = compute_errors(
                reference=x_to_predict, prediction=x_predict, metric=self.error_metric
            )
            self.reduced_dynamics_predictions = {}
            self.reduced_dynamics_predictions["time"] = t_predict
            self.reduced_dynamics_predictions["reduced_coordinates"] = x_predict
            self.reduced_dynamics_predictions["errors"] = prediction_errors
        else:
            t_predict, x_predict = advect(
                dynamics=self.reduced_dynamics.predict,
                t=t,
                x=x_reduced,
                dynamics_type=self.dynamics_type,
            )

            prediction_errors = compute_errors(
                reference=x_reduced, prediction=x_predict, metric=self.error_metric
            )

            reduced_dynamics_predictions = {}
            reduced_dynamics_predictions["time"] = t_predict
            reduced_dynamics_predictions["reduced_coordinates"] = x_predict
            reduced_dynamics_predictions["errors"] = prediction_errors
            return reduced_dynamics_predictions

    def predict(self, idx_trajectories=0, t=[], x=[], x_reduced=[]) -> None:
        if bool(t) is False:
            if idx_trajectories == 0:
                t_to_predict = self.emb_data["time"]
                x_to_predict = self.emb_data["observables"]
            else:
                t_to_predict = [self.emb_data["time"][i] for i in idx_trajectories]
                x_to_predict = [
                    self.emb_data["observables"][i] for i in idx_trajectories
                ]

            if bool(self.geometry_predictions) is False:
                self.predict_geometry(idx_trajectories)

            if bool(self.reduced_dynamics_predictions) is False:
                self.predict_reduced_dynamics(idx_trajectories)

            x_predict = decode_geometry(
                self.decode, self.reduced_dynamics_predictions["reduced_coordinates"]
            )

            prediction_errors = compute_errors(
                reference=x_to_predict, prediction=x_predict, metric=self.error_metric
            )

            self.predictions = {}
            self.predictions["time"] = t_to_predict
            self.predictions["observables"] = x_predict
            self.predictions["errors"] = prediction_errors
        else:

            geometry_predictions = self.predict_geometry(t=t, x=x, x_reduced=x_reduced)
            x_reduced = geometry_predictions["reduced_coordinates"]
            reduced_dynamics_predictions = self.predict_reduced_dynamics(
                t=t, x_reduced=x_reduced
            )
            t_predict = reduced_dynamics_predictions["time"]
            x_reduced_predict = reduced_dynamics_predictions["reduced_coordinates"]

            x_predict = decode_geometry(self.decode, x_reduced_predict)

            prediction_errors = compute_errors(
                reference=x, prediction=x_predict, metric=self.error_metric
            )
            predictions = {}
            predictions["time"] = t_predict
            predictions["reduced_coordinates"] = x_reduced_predict
            predictions["observables"] = x_predict
            predictions["errors"] = prediction_errors
            return predictions
