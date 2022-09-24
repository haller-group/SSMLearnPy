import logging

#import numpy as np
import sklearn

from ssmlearnpy.utils.file_handler import get_vectors
from ssmlearnpy.geometry.coordinates_embedding import coordinates_embedding
from ssmlearnpy.geometry.dimensionality_reduction import reduce_dimensions
from ssmlearnpy.utils.ridge import get_fit_ridge
from ssmlearnpy.reduced_dynamics.shift_or_differentiate import shift_or_differentiate


logger = logging.getLogger("SSMLearn")

class SSMLearn:
    """
    Write here the main docs of the class

    Parameters:
        - t: list of time of the different trajectories, shape=(n_trajectories,)
        - x: list of trajectories, shape=(n_trajectories, )
    """
    def __init__(
        self,
        path_to_trajectories: str=None,
        t: list=None,
        x: list=None,
        params: list=None,
        reduced_coordinates: list=None,
        derive_embdedding=True,
        im_dim: int=None,
        coordinates_embeddings_args: dict={},
        latent_coordinates_method: str='basic'
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


    @staticmethod
    def import_data(path):
        x, t = get_vectors(path)
        return x, t


    def check_inputs(self):
        n_traj_time = len(self.input_data['time'])
        assert(n_traj_time == len(self.input_data['observables']))


    def get_latent_coordinates(
        self,
        method,
        **keyargs
    ) -> None:
        self.encoder = reduce_dimensions(
            method=method,
            **keyargs
        )
        if self.emb_data['reduced_coordinates'] is None:
            self.encoder.fit(self.emb_data)
            self.emb_data['reduced_coordinates']=self.encoder.predict(self.emb_data)
        else:
            logger.info("Reduced coordinated already passed to SSMLearn, skipping.")


    def get_parametrization(
        self,
        **regression_args
    ) -> None:
        self.decoder = get_fit_ridge(
            self.emb_data['reduced_coordinates'],
            self.emb_data['observables'],
            **regression_args
        )


    def get_reduced_dynamics(
        self,
        type,
        structure = None,
        **regression_args
    ) -> None:
        X, y = shift_or_differentiate(self.emb_data['reduced_coordinates'], self.emb_data['time'], type)
        self.reduced_dynamics = get_fit_ridge(
            X,
            y,
            **regression_args
        )

if __name__ == '__main__':
    from ssmlearnpy import SSMLearn

    ssm = SSMLearn(
        path_to_trajectories = '/Users/albertocenedese/Desktop/ssmlearnpy/datasets/Trajectories',
        ssm_dim=2
    )

    ssm.get_latent_coordinates(method='basic', n_dim=2)

    ssm.get_parametrization(poly_degree=18)