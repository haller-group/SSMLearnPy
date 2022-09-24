import numpy as np

def reduce_dimensions(
        method,
        **keyargs
    ):
    if method == 'basic':
        return BasicReducer(**keyargs)
    if method == 'fastssm':
        return BasicReducer(**keyargs)
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
        return list(
            map(
                lambda x: x[:self.n_dim, :],
                data['observables']
            )
        )

