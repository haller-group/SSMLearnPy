import logging
from dataclasses import dataclass, field
import numpy as np
import pickle

from ssmlearnpy.geometry.coordinates_embedding import coordinates_embedding
from ssmlearnpy.geometry.dimensionality_reduction import (
    reduce_dimensions,
    LinearChart,
    BasicReducer,
)
from ssmlearnpy import LArr
from ssmlearnpy.geometry.encode_decode import decode_geometry
from ssmlearnpy.geometry.encode_decode import encode_geometry

from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate
from ssmlearnpy.reduced_dynamics.advector import advect
from ssmlearnpy.reduced_dynamics.normalform import NonlinearCoordinateTransform

from ssmlearnpy.utils.compute_errors import compute_errors
from ssmlearnpy.utils.ridge import (
    get_fit_ridge,
    fit_reduced_coords_and_parametrization,
    get_fit_ridge_parametric,
    Decoder,
)
from ssmlearnpy.utils.ridge import get_matrix
from ssmlearnpy.utils.file_handler import get_vectors
from ssmlearnpy.utils.plots import compute_surface
from ssmlearnpy.utils.data import SSMData, SSMDataAttribute, prev_attribute_map
from ssmlearnpy.utils.config import SSMConfig, NormalFormConfig
from ssmlearnpy.utils.preprocessing import sort_complex_eigenpairs
import ssmlearnpy.reduced_dynamics.normalform as normalform
from scipy.optimize import minimize, least_squares
from scipy.integrate import solve_ivp
from sklearn.pipeline import Pipeline
from copy import deepcopy
from typing import Literal, Optional, Union, Dict, List
from time import time
from pathlib import Path
import ipdb


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

        assert (not self.data.inputs.empty()) ^ bool(
            self.data_path
        ), "Please pass input data via either the SSMData class or a path to the data file."

        if self.data.inputs.empty():
            traj, time = self.import_data(self.data_path)
            self.data.inputs.data = traj
            self.data.inputs.time = time

        # TODO add automatic ssm dimension detection here
        if self.config.ssm_dim is None:
            self.config.ssm_dim = 2

        if self.data.embedded.empty():
            LOGGER.info("Getting coordinates embeddings")
            self.data.embedded.time, self.data.embedded.data, _ = coordinates_embedding(
                self.data.inputs.time,
                self.data.inputs.data,
                self.config.ssm_dim,
                **self.config.coordinates_embeddings_args.model_dump(),
            )

    @staticmethod
    def import_data(path) -> tuple[LArr, LArr]:
        x, t = get_vectors(path)
        return x, t

    def get_reduced_coordinates(
        self,
        method: Literal["linearchart", "basic", "fastssm"] = "linearchart",
        **keyargs,
    ) -> None:
        """
        Compute the reduced coordinates of the trajectories using the given method.
        method: can be 'basic', 'linearchart' or 'fastssm'.
            basic: use the first ssm_dim coordinates of the delay-embedded trajectories
            linearchart: perform an SVD and keep the first ssm_dim coordinates
            fastssm: same as linearchart. We keep the name fastssm to be consistent with the matlab implementation
        If the reduced coordinates have already been computed, skip.
        """
        self.encoder = reduce_dimensions(
            method=method, n_dim=self.config.ssm_dim, **keyargs
        )
        assert self.data.embedded.data, "No embedded signal found."

        # TODO is this behaviour desired? Maybe make recalculating the default behaviour
        if not self.data.reduced_coordinates.data:
            self.data.reduced_coordinates.clear()
            self.encoder.fit(self.data.embedded.data)
            self.data.reduced_coordinates.data = self.encoder.predict(
                self.data.embedded.data
            )
        else:
            LOGGER.info("Reduced coordinates already calculated, skipping.")

    def get_parametrization(self, **regression_args) -> None:
        if (
            self.data.reduced_coordinates.data
        ):  # reduced coordinates have been precomputed
            if self.params:
                self.decoder = get_fit_ridge_parametric(
                    self.data.reduced_coordinates.data,
                    self.data.embedded.data,
                    self.params,
                    **regression_args,
                )
            else:
                self.decoder = get_fit_ridge(
                    self.data.reduced_coordinates.data,
                    self.data.embedded.data,
                    **regression_args,
                )
        else:
            self.encoder, self.decoder = fit_reduced_coords_and_parametrization(
                self.data.embedded.data,
                self.config.ssm_dim,
                **regression_args,
            )  # get both decoder and encoder
            self.data.reduced_coordinates.data = [
                self.encoder.predict(trajectory)
                for trajectory in self.data.embedded.data
            ]

    def encode(self, x):
        """wrapper for encoder.predict. Expects a trajectory of shape (n_features, n_samples)
        returns the reduced coordinates of shape (n_dim, n_samples)
        """
        assert (
            self.encoder is not None
        ), "Encoder not fitted. Please call get_reduced_coordinates() first."
        if isinstance(x, list):
            return [self.encoder.predict(_x) for _x in x]
        elif isinstance(x, np.ndarray):
            return self.encoder.predict(x)

    def decode(self, y):
        """wrapper for decoder.predict. Expects a reduced trajectory of shape (n_dim, n_samples)
        returns the full trajectory of shape (n_features, n_samples)
        """
        assert (
            self.decoder is not None
        ), "Decoder not fitted. Please call get_parametrization() first."
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
        normalform_args: Optional[Dict] = None,
        recalculate_polynomial_dynamics=False,
        **regression_args,
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

        if self.reduced_dynamics is None or recalculate_polynomial_dynamics:
            X, y = shift_or_differentiate(
                self.data.reduced_coordinates.data,
                self.data.embedded.time,
                self.config.dynamics_type,
            )
            if self.params:
                self.reduced_dynamics = get_fit_ridge_parametric(
                    X, y, self.params, **regression_args
                )
            else:
                self.reduced_dynamics = get_fit_ridge(X, y, **regression_args)

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
                self.config.normalform_args = NormalFormConfig.model_validate(
                    normalform_args
                )

            ndofs = int(self.linear_part.shape[0] / 2)
            if self.config.ssm_dim % 2 != 0:
                raise NotImplementedError(
                    (f"Normal form transformation not implemented for odd dimensions.")
                )
            (
                nf_object,
                n_unknowns_dynamics,
                n_unknowns_transformation,
                objective,
            ) = normalform.create_normalform_transform_objective_optimized(
                self.data.embedded.time,
                self.data.reduced_coordinates.data,
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

    def predict_reduced_dynamics(self, data: Optional[SSMData] = None) -> SSMData:

        assign_to_self = False
        if data is None:
            data = self.data
            assign_to_self = True

        assert data.reduced_coordinates.data, "No reduced coordinates found."

        t_pred, x_pred = advect(
            dynamics=self.reduced_dynamics.predict,
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

    def fit_optimal_normalform(
        self, data: Optional[SSMData] = None, min_order: int = 3, max_order: int = 10
    ):

        _data = self.data if data is None else data
        _config = deepcopy(self.config)
        _config.dynamics_structure = "normalform"
        _config.error_metric = "NMTE"

        errors = []
        processed_orders = []
        models = []

        for order in range(min_order, max_order + 1):
            LOGGER.info(f"Fitting normal form dynamics for order {order}")
            start_time = time()
            _ssm = SSMLearn(
                config=_config,
                data=_data,
            )

            _ssm.get_reduced_dynamics()

            _ssm.data.normal_coordinates.data = (
                _ssm.normalform_transformation.inverse_transform(
                    _ssm.data.reduced_coordinates.data
                )
            )

            error_processing_traj = False

            for t, normal_form, embed in zip(
                _ssm.data.embedded.time,
                _ssm.data.normal_coordinates.data,
                _ssm.data.embedded.data,
            ):
                try:
                    if error_processing_traj:
                        break
                    _ssm.data.advected_normal_coordinates.data.append(
                        solve_ivp(
                            _ssm.reduced_dynamics.map_info["vectorfield"],
                            [t[0], t[-1]],
                            normal_form[:, 0],
                            t_eval=t,
                            method="DOP853",
                        ).y
                    )
                except Exception as e:
                    LOGGER.warning(
                        f"Normal form dynamics optimisation for order {order} failed with error: {e}"
                    )
                    error_processing_traj = True

            if error_processing_traj:
                continue

            _ssm.data.advected_normal_coordinates.reconstructed_prev = [
                traj.real
                for traj in _ssm.normalform_transformation.transform(
                    _ssm.data.advected_normal_coordinates.data
                )
            ]

            _ssm.data.advected_normal_coordinates.reconstructed_embedding = _ssm.decode(
                _ssm.data.advected_normal_coordinates.reconstructed_prev
            )

            errors.append(
                np.mean(
                    compute_errors(
                        reference=_ssm.data.embedded.data,
                        prediction=_ssm.data.advected_normal_coordinates.reconstructed_embedding,
                        metric=_ssm.config.error_metric,
                    )
                )
            )
            models.append(_ssm)
            processed_orders.append(order)
            LOGGER.info(
                f"Normal form optimisation for order {order} completed in {time() - start_time} seconds "
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
        self = optimal_model
        return processed_orders, errors

    def fit_optimal_polynomial(
        self, data: Optional[SSMData] = None, min_order: int = 3, max_order: int = 12
    ):

        _data = self.data if data is None else data
        _config = deepcopy(self.config)
        _config.dynamics_structure = "generic"
        _config.error_metric = "NMTE"

        errors = []
        models = []
        processed_orders = []

        for order in range(min_order, max_order + 1):
            start_time = time()
            _ssm = SSMLearn(
                config=_config,
                data=_data,
            )
            try:
                _ssm.predict_reduced_dynamics()
            except Exception as e:
                LOGGER.warning(
                    f"Polynomial dynamics optimisation for order {order} failed with error: {e}"
                )
                continue

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

            models.append(_ssm)
            processed_orders.append(order)

            LOGGER.info(
                f"Polynomial dynamics optimisation for order {order} completed in {time() - start_time} seconds "
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
        self = optimal_model
        return processed_orders, errors

    def run(self, data: Optional[SSMData] = None):
        if data is None:
            data = self.data

        # Data is embedded during object initialisation.
        assert data.embedded.data, "No embedded signal found."

        # Computes the mapping from the embedded phase space to the reduced coordinates
        # on the invariant manifold.
        self.get_parametrization(poly_degree=self.config.geometry_poly_degree)

        # Computes the dynamics on the manifold.
        # Computes both the polynomial dynamics and, if oscillatory,
        # the normal form dynamics.
        self.get_reduced_dynamics(
            poly_degree=self.config.dynamics_poly_degree,
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

    def save(self, path: Path, model_only=True):
        """
        Save the SSMLearn object to a file.
        If model_only is True, only the model is saved, otherwise the data is also saved.
        """
        if model_only:
            data = self.data
            self.data = None

        with open(path, "wb") as f:
            pickle.dump(self, f)

        if model_only:
            self.data = data

    @staticmethod
    def load(path: Path) -> "SSMLearn":
        """
        Load the SSMLearn object from a file.
        """
        with open(path, "rb") as f:
            ssm = pickle.load(f)

        if not isinstance(ssm, SSMLearn):
            raise TypeError(
                f"Expected SSMLearn object, got {type(ssm)} instead. Please check the file."
            )

        if ssm.data is None:
            ssm.data = SSMData()

        return ssm
