import numpy as np

from ssmlearnpy import SSMLearn

ssm = SSMLearn(
    path_to_trajectories = '/Users/albertocenedese/Desktop/pyssmlearn/datasets/Trajectories',
    ssm_dim=2
)

ssm.get_latent_coordinates(method='basic', n_dim=2)

ssm.get_parametrization(poly_degree=18, cv=5, alpha=[0, 1, 10])