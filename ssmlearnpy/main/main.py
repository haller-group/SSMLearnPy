import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import dill
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.pipeline import Pipeline

import ssmlearnpy.reduced_dynamics.normalform as normalform
from ssmlearnpy import LArr
from ssmlearnpy.geometry.coordinates_embedding import coordinates_embedding
from ssmlearnpy.geometry.dimensionality_reduction import (
    BasicReducer,
    LinearChart,
    reduce_dimensions,
)
from ssmlearnpy.geometry.encode_decode import decode_geometry, encode_geometry
from ssmlearnpy.geometry.oblique_projection import *
from ssmlearnpy.reduced_dynamics.advector import advect
from ssmlearnpy.reduced_dynamics.normalform import (
    Dynamics,
    NonlinearCoordinateTransform,
)
from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate
from ssmlearnpy.utils.compute_errors import compute_errors
from ssmlearnpy.utils.config import (
    BaseRegressionConfig,
    GeometryRegressionConfig,
    NormalFormConfig,
    RidgeRegressionConfig,
    SSMConfig,
)
from ssmlearnpy.utils.data import SSMData
from ssmlearnpy.utils.file_handler import get_vectors
from ssmlearnpy.utils.plots import compute_surface
from ssmlearnpy.utils.preprocessing import (
    estimate_ssm_dim,
    get_optimal_timestep,
)
from ssmlearnpy.utils.ridge import (
    Decoder,
    fit_reduced_coords_and_parametrization,
    get_fit_ridge,
    get_fit_ridge_parametric,
    get_matrix,
)

LOGGER = logging.getLogger("SSMLearn")


@dataclass
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

    config: SSMConfig = field(default_factory=SSMConfig)
    encoder: Optional[Union[LinearChart, BasicReducer]] = None
    decoder: Optional[Union[Pipeline, Decoder]] = None
    normalform_transformation: Optional[NonlinearCoordinateTransform] = None
    reduced_dynamics: Optional[Union[Dynamics, Pipeline]] = None
    # TODO what is params. should it be in config, data or here?
    params: Optional[list] = field(default_factory=list)
    data: SSMData = field(default_factory=SSMData)
    data_path: str = ""

    def __post_init__(self) -> None:
        """
        Initializes the SSMLearn class.
        An SSMData object can be passed to the class to resume training from a previous state, or to predict
        from an already trained model.
        If no SSMData object is passed, the class will initialize a new SSMData object and embed it via time-delay.
        """

        assert (not self.data.inputs.empty()) ^ bool(self.data_path), (
            "Please pass input data via either the SSMData class or a path to the data file."
        )

        if self.data.inputs.empty():
            traj, time = self.import_data(self.data_path)
            self.data.inputs.data = traj
            self.data.inputs.time = time

        # This is useful if you already have the phase space, eg. from the high dimensional output
        # of a numerical simulation. If starting from a low-dimensional signal, these should not be skipped.
        if not self.config.bypass_embedding:
            self.preprocess()
            self.embed()
        else:
            assert self.data.embedded.data, (
                "If skipping time delay embedding, you must provide full phase space data in data.embedded.data."
            )

    @staticmethod
    def import_data(path) -> tuple[LArr, LArr]:
        x, t = get_vectors(path)
        return x, t

    def preprocess(self):
        """ """
        data = self.data
        if self.config.ssm_dim is None:
            self.config.ssm_dim = estimate_ssm_dim(data)

        if self.config.coordinates_embeddings_args.shift_steps is None:
            self.config.coordinates_embeddings_args.shift_steps = get_optimal_timestep(
                data
            )

    def embed(self, data: Optional[SSMData] = None) -> SSMData:
        assign_to_self = False
        if data is None:
            data = self.data
            assign_to_self = True
        data = self.data if data is None else data
        assert len(data.inputs.data) > 0, (
            "No input data found. Please provide input data."
        )

        assert self.config.ssm_dim is not None, (
            "SSM dimension is not set. Please set it in the config or run preprocess() to estimate it."
        )
        t, embed, _ = coordinates_embedding(
            data.inputs.time,
            data.inputs.data,
            self.config.ssm_dim,
            **self.config.coordinates_embeddings_args.model_dump(),
        )

        data.embedded.time = t
        data.embedded.data = embed

        if assign_to_self:
            self.data = data
        return data

    def fit_geometry(
        self,
        method: Literal["linearchart", "basic", "oblique-projection"] = "linearchart",
        regression_args: Union[
            BaseRegressionConfig, RidgeRegressionConfig, GeometryRegressionConfig
        ] = BaseRegressionConfig(),
    ) -> None:
        if method == "linearchart":
            # This method ensures that the linear chart used for projection is also
            # used by the encoder

            if isinstance(regression_args, BaseRegressionConfig):
                regression_args = GeometryRegressionConfig.from_shared_args(
                    regression_args
                )

            assert isinstance(regression_args, GeometryRegressionConfig), (
                "regression_args must be of type GeometryRegressionConfig when using LinearChart"
            )

            self.encoder, self.decoder = fit_reduced_coords_and_parametrization(
                self.data.embedded.data,
                n_dim=self.config.ssm_dim,
                **regression_args.model_dump(),
            )
        else:
            if isinstance(regression_args, BaseRegressionConfig):
                regression_args = RidgeRegressionConfig.from_shared_args(
                    regression_args
                )

            assert isinstance(regression_args, RidgeRegressionConfig), (
                "regression_args must be of type RidgeRegressionConfig when using BasicReducer or ObliqueProjection"
            )

            self.fit_encoder(method=method)
            self.fit_decoder(regression_args)

    def fit_encoder(
        self,
        method: Literal["linearchart", "basic", "oblique-projection"] = "linearchart",
    ) -> None:
        """
        Compute the reduced coordinates of the trajectories using the given method.
        method: can be 'basic', 'linearchart' or 'oblique-projection'.
            basic: use the first ssm_dim coordinates of the delay-embedded trajectories
            linearchart: perform an SVD and keep the first ssm_dim coordinates
            oblique-projection: fit the optimal oblique projection matrix and use it to compute the reduced coordinates.
        If the reduced coordinates have already been computed, skip.
        """
        assert self.config.ssm_dim is not None, "SSM dimension is not set."

        training_data = self.data

        if method == "oblique-projection":
            training_data, embedding_config = preprocess_for_oblique_projection(
                self.data, self.config
            )
            t, embed, _ = coordinates_embedding(
                training_data.embedded.time,
                training_data.embedded.data,
                self.config.ssm_dim,
                **embedding_config.model_dump(),
            )
            self.data.embedded.time = t
            self.data.embedded.data = embed

        self.encoder = reduce_dimensions(
            method=method,
            n_dim=self.config.ssm_dim,
        )
        assert training_data.embedded.data, "No embedded signal found."

        self.data.reduced_coordinates.clear()
        self.encoder.fit(training_data.embedded)
        self.data.reduced_coordinates.data = self.encoder.predict(self.data.embedded)

    def fit_decoder(
        self, regression_args: RidgeRegressionConfig = RidgeRegressionConfig()
    ) -> None:
        if self.params:
            self.decoder = get_fit_ridge_parametric(
                self.data.reduced_coordinates.data,
                self.data.embedded.data,
                self.params,
                **regression_args.model_dump(),
            )
        else:
            self.decoder = get_fit_ridge(
                self.data.reduced_coordinates.data,
                self.data.embedded.data,
                **regression_args.model_dump(),
            )

    def encode(self, x):
        """wrapper for encoder.predict. Expects a trajectory of shape (n_features, n_samples)
        returns the reduced coordinates of shape (n_dim, n_samples)
        """
        assert self.encoder is not None, (
            "Encoder not fitted. Please call fit_encoder() first."
        )
        if isinstance(x, list):
            return [self.encoder.predict(_x) for _x in x]
        elif isinstance(x, np.ndarray):
            return self.encoder.predict(x)

    def decode(self, y):
        """wrapper for decoder.predict. Expects a reduced trajectory of shape (n_dim, n_samples)
        returns the full trajectory of shape (n_features, n_samples)
        """
        assert self.decoder is not None, (
            "Decoder not fitted. Please call get_parametrization() first."
        )
        if isinstance(y, list):
            return [self.decoder.predict(_y.T).T for _y in y]
        elif isinstance(y, np.ndarray):
            return self.decoder.predict(y.T).T

    def get_surface(
        self,
        idx_reduced_coordinates=[1, 2],
        idx_observables=1,
        surf_margin=10,
        mesh_step=100,
    ) -> Dict:
        x_data = get_matrix(self.data.reduced_coordinates.data)
        if self.config.ssm_dim == 2:
            U, _, _ = np.linalg.svd(x_data, full_matrices=True)
            max_vals = (1 + surf_margin / 100) * np.amax(np.matmul(U.T, x_data), axis=1)
            transf_mesh_generation = np.matmul(U, np.diag(max_vals))
        else:
            raise NotImplementedError(("Not implemented."))

        surface_dict = compute_surface(
            surface_function=self.decode,
            idx_reduced_coordinates=idx_reduced_coordinates,
            transf_mesh_generation=transf_mesh_generation,
            idx_observables=idx_observables,
            mesh_step=mesh_step,
        )

        return surface_dict

    def fit_reduced_dynamics(
        self,
        data: Optional[SSMData] = None,
        normalform_args: Optional[NormalFormConfig] = None,
        recalculate_polynomial_dynamics=False,
        regression_args: RidgeRegressionConfig = RidgeRegressionConfig(),
    ) -> None:
        """Compute the reduced dynamics from the data supplied to the class.

        Parameters:
            recalculate_polynomial_dynamics (bool, optional): If True, the polynomial dynamics is always calculated. Defaults to False.
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

        if data is None:
            data = self.data

        if self.reduced_dynamics is None or recalculate_polynomial_dynamics:
            X, y = shift_or_differentiate(
                data.reduced_coordinates.data,
                data.embedded.time,
                self.config.dynamics_type,
            )
            if self.params:
                self.reduced_dynamics = get_fit_ridge_parametric(
                    X, y, self.params, **regression_args.model_dump()
                )
            else:
                self.reduced_dynamics = get_fit_ridge(
                    X, y, **regression_args.model_dump()
                )

            linear_part = self.reduced_dynamics.map_info["coefficients"][
                :, : X[0].shape[0]
            ]
            d, v = np.linalg.eig(linear_part)

            self.linear_part = linear_part
            self.reduced_dynamics.map_info["eigenvalues_linear_part"] = d
            self.reduced_dynamics.map_info["eigenvectors_linear_part"] = v
            self.eigenvalues = d
            self.eigenvectors = v
            self.polynomial_reduced_dynamics = deepcopy(self.reduced_dynamics)

        if (
            self.config.dynamics_structure == "normalform" and self.is_oscillatory()
        ):  # compute the normal form transformation after an initial guess has been computed
            if normalform_args is not None:
                self.config.normalform_args = normalform_args

            ndofs = int(self.linear_part.shape[0] / 2)
            if self.config.ssm_dim % 2 != 0:
                raise NotImplementedError(
                    ("Normal form transformation not implemented for odd dimensions.")
                )
            (
                nf_object,
                n_unknowns_dynamics,
                n_unknowns_transformation,
                objective,
            ) = normalform.create_normalform_transform_objective_optimized(
                data.embedded.time,
                data.reduced_coordinates.data,
                self.linear_part,
                degree=self.config.normalform_args.degree,
                do_scaling=self.config.normalform_args.do_scaling,
                tolerance=self.config.normalform_args.tolerance,
                use_center_manifold_style=self.config.normalform_args.use_center_manifold_style,
            )

            # create 3 kinds of initial guesses:
            if self.config.normalform_args.ic_style == "random":
                initial_guess = np.random.rand(
                    (n_unknowns_dynamics + n_unknowns_transformation) * 2
                )  # both real and imaginary parts
            elif self.config.normalform_args.ic_style == "informed":
                initial_guess = normalform.create_normalform_initial_guess(
                    self.reduced_dynamics, nf_object
                )
            elif self.config.normalform_args.ic_style == "zero":
                initial_guess = np.zeros(
                    (n_unknowns_dynamics + n_unknowns_transformation) * 2
                )
            else:
                raise ValueError(
                    f"Unknown initial condition style: {self.config.normalform_args.ic_style}"
                )

            res = least_squares(
                objective,
                initial_guess,
                method=self.config.normalform_args.method,
                jac=self.config.normalform_args.jac,
                max_nfev=self.config.normalform_args.max_iter,
            )
            if not res.success:
                print(f"Optimization did not converge. Message: {res.message}")
                LOGGER.error((f"Optimization did not converge. Message: {res.message}"))
            else:
                LOGGER.info((f"Optimization converged. Message: {res.message}"))
                LOGGER.info(f"Number of iterations: {res.nfev}")
                print(f"Optimization converged. Message: {res.message}")
                print(f"Number of iterations: {res.nfev}")

            unpacked_coeffs = normalform.unpack_optimized_coeffs(
                res.x, ndofs, nf_object, n_unknowns_dynamics, n_unknowns_transformation
            )
            transformation, dynamics = normalform.wrap_optimized_coefficients(
                ndofs,
                nf_object,
                self.config.normalform_args.degree,
                unpacked_coeffs,
                find_inverse=True,
                trajectories=self.data.reduced_coordinates.data,
                raw_coeffs=res.x,
            )
            self.normalform_transformation = transformation
            self.reduced_dynamics = dynamics
            self.reduced_dynamics.map_info["normalform_transformation"] = transformation

    def is_oscillatory(self) -> bool:
        """
        Check if the system is oscillatory by checking if the eigenvalues of the linear part
        of the reduced dynamics are all complex.
        Returns:
            bool: True if the system is oscillatory, False otherwise.
        """
        if self.reduced_dynamics is None:
            raise ValueError(
                "Reduced dynamics not computed. Please call get_reduced_dynamics() first."
            )

        return (np.abs(self.eigenvalues.imag) > 0).all()

    def predict_geometry(
        self,
        data: Optional[SSMData] = None,
    ) -> SSMData:
        assign_to_self = False
        if data is None:
            data = self.data
            assign_to_self = True

        assert data.embedded.data, "No embedded signal found."

        if not data.reduced_coordinates.data:
            data.reduced_coordinates.data = encode_geometry(
                self.encode, data.embedded.data
            )

        data.reduced_coordinates.reconstructed_prev = decode_geometry(
            self.decode, data.reduced_coordinates.data
        )

        data.reduced_coordinates.immediate_errors = compute_errors(
            reference=data.embedded.data,
            prediction=data.reduced_coordinates.reconstructed_prev,
            metric=self.config.error_metric,
        )

        if assign_to_self:
            self.data = data

        return data

    def predict_normalform_reduced_dynamics(
        self,
        data: Optional[SSMData] = None,
    ) -> SSMData:
        assign_to_self = False
        if data is None:
            data = self.data
            assign_to_self = True

        assert data.reduced_coordinates.data, "No reduced coordinates found."

        data.normal_coordinates.data = self.normalform_transformation.inverse_transform(
            data.reduced_coordinates.data
        )

        data.advected_normal_coordinates.data = []
        for t, normal_form in zip(
            data.embedded.time,
            data.normal_coordinates.data,
        ):
            try:
                data.advected_normal_coordinates.data.append(
                    solve_ivp(
                        self.reduced_dynamics.map_info["vectorfield"],
                        [t[0], t[-1]],
                        normal_form[:, 0],
                        t_eval=t,
                        method="DOP853",
                    ).y
                )
                data.advected_normal_coordinates.time.append(t)
            except Exception as e:
                LOGGER.warning("Integration failed when advecting trajectory")
                raise (e)

        data.advected_normal_coordinates.reconstructed_prev = [
            traj.real
            for traj in self.normalform_transformation.transform(
                data.advected_normal_coordinates.data
            )
        ]

        data.advected_normal_coordinates.reconstructed_embedding = self.decode(
            data.advected_normal_coordinates.reconstructed_prev
        )

        # TODO also calculate prediction errors

        if assign_to_self:
            self.data = data
        return data

    def predict_polynomial_reduced_dynamics(
        self, data: Optional[SSMData] = None
    ) -> SSMData:
        assign_to_self = False
        if data is None:
            data = self.data
            assign_to_self = True

        assert data.reduced_coordinates.data, "No reduced coordinates found."

        t_pred, x_pred = advect(
            dynamics=self.polynomial_reduced_dynamics.predict,
            t=data.embedded.time,
            x=data.reduced_coordinates.data,
            dynamics_type=self.config.dynamics_type,
        )

        data.advected_reduced_coordinates.data = x_pred
        data.advected_reduced_coordinates.time = t_pred

        prediction_errors = compute_errors(
            reference=data.reduced_coordinates.data,
            prediction=data.advected_reduced_coordinates.data,
            metric=self.config.error_metric,
        )

        data.advected_reduced_coordinates.immediate_errors = prediction_errors

        if assign_to_self:
            self.data = data

        return data

    def fit_optimal_dynamics(self, data: Optional[SSMData] = None):
        if data is None:
            data = self.data

        if self.is_oscillatory():
            return self.fit_optimal_normalform(data)
        else:
            return self.fit_optimal_polynomial(data)

    def fit_optimal_normalform(self, data: Optional[SSMData] = None):
        _data = self.data if data is None else data
        # Fit linear part
        self.fit_reduced_dynamics(
            recalculate_polynomial_dynamics=True,
            regression_args=RidgeRegressionConfig(poly_degree=2),
        )

        self.data = None
        _ssm = deepcopy(self)
        _config = deepcopy(self.config)
        _data = deepcopy(_data)
        _config.dynamics_structure = "normalform"
        _config.error_metric = "NMTE"

        errors = []
        processed_orders = []
        models: List[SSMLearn] = []

        for order in range(
            self.config.dynamics_polynomial_range[0],
            self.config.dynamics_polynomial_range[1] + 1,
        ):
            LOGGER.info(f"Fitting normal form dynamics for order {order}")
            start_time = time.time()
            _config.normalform_args.degree = order
            _ssm.data = _data
            _ssm.config = _config

            try:
                _ssm.fit_reduced_dynamics()
                _ssm.predict_normalform_reduced_dynamics()
                errors.append(
                    np.mean(
                        compute_errors(
                            reference=_ssm.data.embedded.data,
                            prediction=_ssm.data.advected_normal_coordinates.reconstructed_embedding,
                            metric=_ssm.config.error_metric,
                        )
                    )
                )
            except Exception as e:
                LOGGER.warning(
                    f"Normal form dynamics optimisation for order {order} failed with error: {e}"
                )
                continue

            _ssm.save(
                path=Path(_config.save_directory) / str(order) / "model.dill",
                include_data=_config.save_suboptimal_model_data_obj,
            )
            models.append(_ssm)
            processed_orders.append(order)
            LOGGER.info(
                f"Normal form optimisation for order {order} completed in {time.time() - start_time} seconds "
                f"with mean relative error {errors[-1]}"
            )

            if errors[-1] < _config.reconstruction_error_threshold:
                if _config.optimal_model_early_stopping:
                    break

        index = next(
            (
                i
                for i, e in enumerate(errors)
                if e < _config.reconstruction_error_threshold
            ),
            -1,
        )
        if index == -1:
            LOGGER.warning("Normal form reconstruction threshold not reached")
            index = np.argmin(errors)
        LOGGER.info(f"Optimal normal form order: {processed_orders[index]}")

        optimal_model = models[index]
        optimal_model.config = self.config
        self.update(optimal_model)
        return {
            "orders": processed_orders,
            "errors": errors,
            "selected_order": processed_orders[index],
        }

    def fit_optimal_polynomial(self, data: Optional[SSMData] = None):
        _data = self.data if data is None else data
        self.data = None
        _ssm = deepcopy(self)
        _config = deepcopy(self.config)
        _data = deepcopy(_data)
        _config.dynamics_structure = "generic"
        _config.error_metric = "NMTE"

        errors = []
        models = []
        processed_orders = []

        for order in range(
            self.config.dynamics_polynomial_range[0],
            self.config.dynamics_polynomial_range[1] + 1,
        ):
            start_time = time.time()
            _config.dynamics_poly_degree = order
            _ssm.data = _data
            _ssm.config = _config
            try:
                _ssm.fit_reduced_dynamics(
                    recalculate_polynomial_dynamics=True,
                    regression_args=RidgeRegressionConfig(
                        poly_degree=order,
                    ),
                )
                _ssm.predict_polynomial_reduced_dynamics()
                _ssm.data.advected_reduced_coordinates.reconstructed_embedding = (
                    _ssm.decode(_ssm.data.advected_reduced_coordinates.data)
                )

                errors.append(
                    np.mean(
                        compute_errors(
                            reference=_ssm.data.embedded.data,
                            prediction=_ssm.data.advected_reduced_coordinates.reconstructed_embedding,
                            metric=_ssm.config.error_metric,
                        )
                    )
                )
            except Exception as e:
                LOGGER.warning(
                    f"Polynomial dynamics optimisation for order {order} failed with error: {e}"
                )
                continue

            _ssm.save(
                path=Path(_config.save_directory) / str(order) / "model.dill",
                include_data=_config.save_suboptimal_model_data_obj,
            )
            models.append(_ssm)
            processed_orders.append(order)

            LOGGER.info(
                f"Polynomial dynamics optimisation for order {order} completed in {time.time() - start_time} seconds "
                f"with mean relative error {errors[-1]}"
            )

            # TODO save intermediate model if desired
            # Optimal model is considered to be the simplest (lowest order) model
            # that has a reconstruction error below the threshold.
            if errors[-1] < _config.reconstruction_error_threshold:
                if _config.optimal_model_early_stopping:
                    break

        index = next(
            (
                i
                for i, e in enumerate(errors)
                if e < _config.reconstruction_error_threshold
            ),
            -1,
        )
        if index == -1:
            LOGGER.warning("Reconstruction threshold not reached")
            index = np.argmin(errors)
        LOGGER.info(f"Optimal polynomial order: {processed_orders[index]}")
        optimal_model = models[index]
        optimal_model.config = self.config
        self.update(optimal_model)
        return {
            "orders": processed_orders,
            "errors": errors,
            "selected_order": processed_orders[index],
        }

    def predict(self, data: Optional[SSMData] = None) -> SSMData:
        """
        Assume that only the input data is given, then run through the rest of the pipline.
        """
        assign_to_self = False
        if data is None:
            data = self.data
            assign_to_self = True
        data = self.embed(data)
        data = self.predict_geometry(data)
        if self.is_oscillatory():
            data = self.predict_polynomial_reduced_dynamics(data)
        else:
            data = self.predict_normalform_reduced_dynamics(data)

        if assign_to_self:
            self.data = data

        return data

    def fit(self, data: Optional[SSMData] = None):
        if data is None:
            data = self.data

        # Data is embedded during object initialisation.
        assert data.embedded.data, "No embedded signal found."

        # Computes the mapping from the embedded phase space to the reduced coordinates
        # on the invariant manifold.
        self.fit_geometry(
            regression_args=BaseRegressionConfig(
                poly_degree=self.config.geometry_poly_degree
            )
        )

        data = self.predict_geometry(data)

        # Computes the dynamics on the manifold.
        # Computes both the polynomial dynamics and, if oscillatory,
        # the normal form dynamics.
        self.fit_reduced_dynamics(
            data=data,
            regression_args=RidgeRegressionConfig(
                poly_degree=self.config.dynamics_poly_degree
            ),
            recalculate_polynomial_dynamics=True,
        )

        # Computes either the optimal polynomal dynamics or the optimal normal
        # form dynamics, depending on whether the system is oscillatory
        return self.fit_optimal_dynamics(data)

    def detach_data(self) -> "SSMLearn":
        """
        Detach the data from the SSMLearn object.
        This is useful when you want to save the model without the data.
        """
        self.data = None
        return self

    def save(self, path: Path, include_data=False):
        """
        Save the SSMLearn object to a file.
        If include_data is True, the data is also saved.
        """
        if not include_data:
            data = self.data
            self.data = None

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            dill.dump(self, f)

        if not include_data:
            self.data = data

    @staticmethod
    def load(path: Path) -> "SSMLearn":
        """
        Load the SSMLearn object from a file.
        """
        with open(path, "rb") as f:
            ssm = dill.load(f)

        if not isinstance(ssm, SSMLearn):
            raise TypeError(
                f"Expected SSMLearn object, got {type(ssm)} instead. Please check the file."
            )

        if ssm.data is None:
            ssm.data = SSMData()

        return ssm

    def update(self, other: "SSMLearn") -> None:
        """
        Replace the current SSMLearn object with another one.
        """
        for attr in other.__dict__:
            setattr(self, attr, getattr(other, attr))
