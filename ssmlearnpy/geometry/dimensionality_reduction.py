import numpy as np
from ssmlearnpy.utils.preprocessing import get_matrix
from ssmlearnpy.geometry.oblique_projection import oblique_projection
from ssmlearnpy.utils.data import SSMData, SSMDataAttribute
from ssmlearnpy.utils.config import SSMConfig
from typing import Optional, Union


def reduce_dimensions(method, n_dim: int):
    if method == "basic":
        return BasicReducer(n_dim=n_dim)
    # if method == "fastssm":
    #     return BasicReducer(n_dim=n_dim)
    if method == "linearchart":
        return LinearChart(n_dim=n_dim)
    if method == "oblique_projection":
        return ObliqueProjection(n_dim=n_dim)
    else:
        raise NotImplementedError(
            (
                f"{method} not implemented, please specify a method that have "
                f"already been implemented, otherwise raise an issue to the developers"
            )
        )


class BasicReducer:
    def __init__(
        self,
        n_dim,
    ) -> None:
        self.n_dim = n_dim

    def fit(self, data: SSMDataAttribute):
        pass

    def predict(self, data: SSMDataAttribute):
        return [data_i[: self.n_dim, :] for data_i in data.data]


class LinearChart:
    def __init__(self, n_dim, matrix_representation=None) -> None:
        self.n_dim = n_dim
        self.matrix_representation = matrix_representation

    def fit(self, data: Union[SSMDataAttribute, np.ndarray]):

        if isinstance(data, SSMDataAttribute):
            _data = get_matrix(data.data)
        elif isinstance(data, np.ndarray):
            _data = data
        else:
            raise TypeError(
                (
                    "Data must be of type SSMDataAttribute or np.ndarray, "
                    f"got {type(data)} instead"
                )
            )

        # Centre the data for PCA
        centred_data = _data - np.mean(_data, axis=1)[:, None]
        U, s, v = np.linalg.svd(centred_data, full_matrices=False)
        self.matrix_representation = U[:, : self.n_dim]

    def predict(self, data: Union[SSMDataAttribute, np.ndarray]):
        if self.matrix_representation is None:
            raise RuntimeError(
                (
                    "No projection set for LinearChart. Provide a matrix representation or call .fit() first"
                )
            )

        if isinstance(data, np.ndarray):
            return np.matmul(self.matrix_representation.T, data)

        return [np.matmul(self.matrix_representation.T, data_i) for data_i in data.data]


class ObliqueProjection:
    def __init__(self, n_dim, matrix_representation=None) -> None:
        self.n_dim = n_dim
        self.matrix_representation = matrix_representation

    def fit(self, data: SSMDataAttribute):
        self.matrix_representation = oblique_projection(data)

    def predict(self, data: SSMDataAttribute):
        if self.matrix_representation is None:
            raise RuntimeError(
                (
                    "No projection set for ObliqueProjection. Provide a matrix representation or call .fit() first"
                )
            )
        return [np.matmul(self.matrix_representation.T, data_i) for data_i in data.data]
