import logging

import numpy as np

from ssmlearnpy.geometry.coordinates_embedding import coordinates_embedding
from ssmlearnpy.geometry.dimensionality_reduction import reduce_dimensions
from ssmlearnpy.geometry.encode_decode import decode_geometry
from ssmlearnpy.geometry.encode_decode import encode_geometry

from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate
from ssmlearnpy.reduced_dynamics.advector import advect

from ssmlearnpy.utils.compute_errors import compute_errors
from ssmlearnpy.utils.ridge import get_fit_ridge, fit_reduced_coords_and_parametrization, get_fit_ridge_parametric
from ssmlearnpy.utils.ridge import get_matrix
from ssmlearnpy.utils.file_handler import get_vectors
from ssmlearnpy.utils.plots import compute_surface

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
        - dynamics_structure: structure of the reduced dynamics
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
    """
    def __init__(
        self,
        path_to_trajectories: str=None,
        t: list=None,
        x: list=None,
        params: list=None,
        reduced_coordinates: list=None,
        derive_embdedding=True,
        ssm_dim: int=None,
        coordinates_embeddings_args: dict={},
        dynamics_type = 'flow',
        dynamics_structure = 'generic',
        error_metric = 'NTE'
    ) -> None:
        self.input_data = {}

        if path_to_trajectories:
            self.input_data['time'], self.input_data['observables'] = self.import_data(
                path_to_trajectories
            )

        elif t and x:
            self.input_data['time'] = t
            self.input_data['observables'] = x

        else:
            raise RuntimeError(
                (
                    f"Not enought parameters specified. Found: path_to_trajectories={path_to_trajectories}" +
                    f"t={t}, x={x}. Please either set the path to the trajectories file or pass them to the class"
                )
            )

        self.check_inputs()
        self.emb_data = {}

        if derive_embdedding and not reduced_coordinates:
            logger.info("Getting coordinates embeddings")
            self.emb_data = {}
            self.emb_data['time'], self.emb_data['observables'], _ = coordinates_embedding(
                self.input_data['time'],
                self.input_data['observables'],
                ssm_dim,
                **coordinates_embeddings_args
            )
        else: 
            self.emb_data['time'] = self.input_data['time']
            self.emb_data['observables'] = self.input_data['observables']
        
        self.emb_data['params'] = params
        self.emb_data['reduced_coordinates'] = reduced_coordinates

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

    @staticmethod
    def import_data(path):
        x, t = get_vectors(path)
        return x, t

    def check_inputs(self):
        n_traj_time = len(self.input_data['time'])
        assert(n_traj_time == len(self.input_data['observables']))

    def get_reduced_coordinates(
        self,
        method,
        **keyargs
    ) -> None:
        self.encoder = reduce_dimensions(
            method=method,
            n_dim = self.ssm_dim,
            **keyargs
        )
        if self.emb_data['reduced_coordinates'] is None:
            self.encoder.fit(self.emb_data['observables'])
            self.emb_data['reduced_coordinates']=self.encoder.predict(self.emb_data['observables'])
        else:
            logger.info("Reduced coordinated already passed to SSMLearn, skipping.")


    def get_parametrization(
        self,
        **regression_args
    ) -> None:
        if self.emb_data['reduced_coordinates'] is not None: # reduced coordinates have been precomputed
            if self.emb_data['params'] is not None:
                self.decoder = get_fit_ridge_parametric(
                        self.emb_data['reduced_coordinates'],
                        self.emb_data['observables'],
                        self.emb_data['params'],
                        **regression_args)
            else:
                self.decoder = get_fit_ridge(
                        self.emb_data['reduced_coordinates'],
                        self.emb_data['observables'],
                        **regression_args)
        else:
            self.encoder, self.decoder = fit_reduced_coords_and_parametrization(self.emb_data['observables'],
                                                                                 self.ssm_dim, **regression_args) # get both decoder and encoder
            self.emb_data['reduced_coordinates'] = [self.encoder.predict(trajectory)
                                                     for trajectory in self.emb_data['observables']]
    def get_surface(
        self,
        idx_reduced_coordinates = [1, 2],
        idx_observables = 1,
        surf_margin = 10,
        mesh_step = 100
    ) -> None:

        x_data = get_matrix(self.emb_data['reduced_coordinates'])
        if self.ssm_dim == 2:
            U, _, _ = np.linalg.svd(x_data, full_matrices=True)
            max_vals = (1+surf_margin/100) * np.amax(np.matmul(U.T,x_data), axis = 1)
            transf_mesh_generation = np.matmul(U,np.diag(max_vals))
        else:
            raise NotImplementedError(
            (
                f"Not implemented."
            )
            )

        surface_dict = compute_surface(
            surface_function = self.decoder.predict,
            idx_reduced_coordinates = idx_reduced_coordinates,
            transf_mesh_generation = transf_mesh_generation,
            idx_observables = idx_observables,
            mesh_step = mesh_step
        )

        return surface_dict

    def get_reduced_dynamics(
        self,
        **regression_args
    ) -> None:
        X, y = shift_or_differentiate(
            self.emb_data['reduced_coordinates'], 
            self.emb_data['time'], 
            self.dynamics_type
        )
        if self.emb_data['params'] is not None:
            self.reduced_dynamics = get_fit_ridge_parametric(
                X,
                y,
                self.emb_data['params'],
                **regression_args
            )
        else:
            self.reduced_dynamics = get_fit_ridge(
                X,
                y,
                **regression_args
            )
        lin_part = self.reduced_dynamics.map_info['coefficients'][:,:X[0].shape[0]]
        d, v = np.linalg.eig(lin_part)
        self.reduced_dynamics.map_info['eigenvalues_lin_part'] = d
        self.reduced_dynamics.map_info['eigenvectors_lin_part'] = v

    def predict_geometry(
        self,
        idx_trajectories = 0,
        t = [],
        x = [],
        x_reduced = []
    ) -> None:
        if bool(t) is False:
            if idx_trajectories == 0:
                t_to_predict = self.emb_data['time']
                x_reduced = self.emb_data['reduced_coordinates']
                x_to_predict = self.emb_data['observables']
            else:
                t_to_predict = [self.emb_data['time'][i] for i in idx_trajectories]
                x_reduced = [self.emb_data['reduced_coordinates'][i] for i in idx_trajectories]
                x_to_predict = [self.emb_data['observables'][i] for i in idx_trajectories]

            x_predict = decode_geometry(
                self.decoder.predict,
                x_reduced)
            
            prediction_errors = compute_errors(
                reference=x_to_predict,
                prediction=x_predict,
                metric=self.error_metric
            )
            self.geometry_predictions = {}
            self.geometry_predictions['time'] = t_to_predict
            self.geometry_predictions['reduced_coordinates'] = x_predict
            self.geometry_predictions['observables'] = x_predict
            self.geometry_predictions['errors'] = prediction_errors
        else:
            if bool(x_reduced) is False:
                x_reduced = encode_geometry(
                self.encoder.predict,
                x)  

            x_predict = decode_geometry(
                self.decoder.predict,
                x_reduced)

            prediction_errors = compute_errors(
                reference=x,
                prediction=x_predict,
                metric=self.error_metric
            )
            geometry_predictions = {}
            geometry_predictions['time'] = t
            geometry_predictions['reduced_coordinates'] = x_reduced
            geometry_predictions['observables'] = x_predict
            geometry_predictions['errors'] = prediction_errors
            return geometry_predictions 

    def predict_reduced_dynamics(
        self,
        idx_trajectories = 0,
        t = [],
        x_reduced = []
    ) -> None:
        if bool(t) is False:
            if idx_trajectories == 0:
                t_to_predict = self.emb_data['time']
                x_to_predict = self.emb_data['reduced_coordinates']
            else:
                t_to_predict = [self.emb_data['time'][i] for i in idx_trajectories]
                x_to_predict = [self.emb_data['reduced_coordinates'][i] for i in idx_trajectories]
            
            t_predict, x_predict  = advect(
                dynamics=self.reduced_dynamics.predict,
                t=t_to_predict,
                x=x_to_predict,
                dynamics_type=self.dynamics_type
            )

            prediction_errors = compute_errors(
                reference=x_to_predict,
                prediction=x_predict,
                metric=self.error_metric
            )
            self.reduced_dynamics_predictions = {}
            self.reduced_dynamics_predictions['time'] = t_predict
            self.reduced_dynamics_predictions['reduced_coordinates'] = x_predict
            self.reduced_dynamics_predictions['errors'] = prediction_errors
        else:
            t_predict, x_predict  = advect(
                dynamics=self.reduced_dynamics.predict,
                t=t,
                x=x_reduced,
                dynamics_type=self.dynamics_type
            )

            prediction_errors = compute_errors(
                reference=x_reduced,
                prediction=x_predict,
                metric=self.error_metric
            )

            reduced_dynamics_predictions = {}
            reduced_dynamics_predictions['time'] = t_predict
            reduced_dynamics_predictions['reduced_coordinates'] = x_predict
            reduced_dynamics_predictions['errors'] = prediction_errors
            return reduced_dynamics_predictions

    def predict(
        self,
        idx_trajectories = 0,
        t = [],
        x = [],
        x_reduced = []
    ) -> None:
        if bool(t) is False:
            if idx_trajectories == 0:
                t_to_predict = self.emb_data['time']
                x_to_predict = self.emb_data['observables']
            else:
                t_to_predict = [self.emb_data['time'][i] for i in idx_trajectories]
                x_to_predict = [self.emb_data['observables'][i] for i in idx_trajectories]

            if bool(self.geometry_predictions) is False:
                self.predict_geometry(idx_trajectories)

            if bool(self.reduced_dynamics_predictions) is False:
                self.predict_reduced_dynamics(idx_trajectories)

            x_predict = decode_geometry(
                self.decoder.predict,
                self.reduced_dynamics_predictions['reduced_coordinates'])

            prediction_errors = compute_errors(
                reference=x_to_predict,
                prediction=x_predict,
                metric=self.error_metric
            )

            self.predictions = {}
            self.predictions['time'] = t_to_predict
            self.predictions['observables'] = x_predict
            self.predictions['errors'] = prediction_errors
        else:
            
            geometry_predictions = self.predict_geometry(
                t = t,
                x = x,
                x_reduced = x_reduced
            )
            x_reduced = geometry_predictions['reduced_coordinates']
            reduced_dynamics_predictions = self.predict_reduced_dynamics(
                t = t,
                x_reduced = x_reduced
            )
            t_predict = reduced_dynamics_predictions['time']
            x_reduced_predict = reduced_dynamics_predictions['reduced_coordinates']

            x_predict = decode_geometry(
                self.decoder.predict,
                x_reduced_predict)

            prediction_errors = compute_errors(
                reference=x,
                prediction=x_predict,
                metric=self.error_metric
            )
            predictions = {}
            predictions['time'] = t_predict
            predictions['reduced_coordinates'] = x_reduced_predict
            predictions['observables'] = x_predict
            predictions['errors'] = prediction_errors
            return predictions

    
# if __name__ == '__main__':
#     from ssmlearnpy import SSMLearn

#     ssm = SSMLearn(
#         path_to_trajectories = '/Users/albertocenedese/Desktop/ssmlearnpy/datasets/Trajectories',
#         ssm_dim=2
#     )

#     ssm.get_latent_coordinates(method='basic', n_dim=2)

#     ssm.get_parametrization(poly_degree=18)