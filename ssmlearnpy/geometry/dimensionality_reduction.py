import numpy as np
from ssmlearnpy.utils.ridge import get_matrix


def reduce_dimensions(
        method,
        **keyargs
    ):
    if method == 'basic':
        return BasicReducer(**keyargs)
    if method == 'fastssm':
        return BasicReducer(**keyargs)
    if method == 'linearchart':
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

    def fit(self, data):
        pass

    def predict(self, data):
        return [data_i[:self.n_dim] for data_i in data]

class LinearChart:
    def __init__(
            self,
            n_dim,
            matrix_representation = None
        ) -> None:
        self.n_dim = n_dim
        self.matrix_representation = matrix_representation

    def fit(self, data):
        if self.matrix_representation is not None:
            pass
        x_data = get_matrix(data)
        U, s, v = np.linalg.svd(x_data, full_matrices = False)
        self.matrix_representation = U[:, :self.n_dim]
        return 
        
    def predict(self, data):
        if self.matrix_representation is None:
            raise RuntimeError(
                (
                    "No projection set for LinearChart. Provide a matrix representation or call .fit() first"
                )
            )
        return [np.matmul( self.matrix_representation.T, data_i) for data_i in data]