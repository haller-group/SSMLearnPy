import numpy as np
from ssmlearnpy.utils.preprocessing import get_matrix


def reduce_dimensions(method, **keyargs):
    if method == "basic":
        return BasicReducer(**keyargs)
    if method == "fastssm":
        return BasicReducer(**keyargs)
    if method == "linearchart":
        return LinearChart(**keyargs)
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

    def fit(self, data, offset=None):
        if offset is not None:
            raise NotImplementedError("Offset not implemented for BasicReducer")
        pass

    def predict(self, data):
        return [data_i[: self.n_dim, :] for data_i in data]


class LinearChart:
    def __init__(self, n_dim, matrix_representation=None) -> None:
        self.n_dim = n_dim
        self.matrix_representation = matrix_representation
        self.offset = None

    def fit(self, data, offset=None):
        # If the data is given as a matrix, the fixed point is assumed to be at the origin
        if self.matrix_representation is not None:
            pass
        if isinstance(data, list):
            if offset is not None:
                self.offset = offset
            data = get_matrix(data)

        # Centre the data for PCA
        centred_data = data - np.mean(data, axis=1)[:, None]
        U, s, v = np.linalg.svd(centred_data, full_matrices=False)
        self.matrix_representation = U[:, : self.n_dim]
        return

    def predict(self, data):
        if self.matrix_representation is None:
            raise RuntimeError(
                (
                    "No projection set for LinearChart. Provide a matrix representation or call .fit() first"
                )
            )
        if isinstance(
            data, list
        ):  # want this to work for a single datamatrix and for a list of trajectories
            if self.offset is not None:
                data = [d - self.offset.reshape(-1, 1) for d in data]
            return [np.matmul(self.matrix_representation.T, data_i) for data_i in data]
        else:
            return np.matmul(self.matrix_representation.T, data)
