from pydantic import BaseModel
from typing import Literal, Optional, Dict, Tuple
from numpy import ndarray


class BaseRegressionConfig(BaseModel):
    """
    poly_degree: int, degree (with respect to the state) of the polynomial to fit
    fit_intercept: bool, whether to include the constant term in the regression
                    if False, this means that the model will be forced to pass through the origin
    alpha: float or list of floats, regularization parameter
    cv: int, number of folds for cross validation. If cv>=2, alpha must be a list
    """

    poly_degree: int = 2
    fit_intercept: bool = False
    alpha: float | list[float] = 0
    cv: int = 2


class GeometryRegressionConfig(BaseRegressionConfig):
    """
    Used by the function
    `fit_reduced_coords_and_parametrization`
    """

    initial_guess: Optional[ndarray] = None
    penalty_linear_cons: float = 1e-5
    penalty_nonlinear_cons: float = 1e-5
    optimize_kwargs: Dict = {
        "method": "lm",
        "ftol": 1e-6,
        "gtol": 1e-6,
        "verbose": 0,
    }

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @staticmethod
    def from_shared_args(
        shared_args: BaseRegressionConfig,
    ) -> "GeometryRegressionConfig":
        """
        Create a GeometryRegressionConfig from a BaseRegressionArgs instance.
        """
        return GeometryRegressionConfig(
            poly_degree=shared_args.poly_degree,
            fit_intercept=shared_args.fit_intercept,
            alpha=shared_args.alpha,
            cv=shared_args.cv,
        )


class RidgeRegressionConfig(BaseRegressionConfig):
    """
    Used by `get_fit_ridge` and `get_fit_ridge_parametric`

    constraints: list of lists: [LHS, RHS] such that model.predict(LHS[i]) == RHS[i].
                model.predict(LHS[i]) and RHS[i] should have the same shape.
                As a result, the last entries in LHS[i] should refer to the parameters.
    do_scaling: bool, whether to apply a StandardScaler to the data before fitting
    """

    constraints: list | None = None
    do_scaling: bool = True

    @staticmethod
    def from_shared_args(shared_args: BaseRegressionConfig) -> "RidgeRegressionConfig":
        """
        Create a RidgeRegressionConfig from a BaseRegressionArgs instance.
        """
        return RidgeRegressionConfig(
            poly_degree=shared_args.poly_degree,
            fit_intercept=shared_args.fit_intercept,
            alpha=shared_args.alpha,
            cv=shared_args.cv,
        )


class ParametricRegressionConfig(RidgeRegressionConfig):
    """
    origin_remains_fixed: bool. If True, then the regression will not contain terms that depend only on the parameter.
                            otherwise the parameter is simply treated as an additional feature.
    poly_degree_parameter: int, degree of the polynomial to fit for the parameters.
                        In general, poly_degree_parameter != poly_degree, but it should be at most poly_degree.
    """

    origin_remains_fixed: bool = True
    poly_degree_parameter: int = 2

    @staticmethod
    def from_shared_args(
        shared_args: BaseRegressionConfig,
    ) -> "ParametricRegressionConfig":
        """
        Create a ParametricRegressionConfig from a BaseRegressionArgs instance.
        """
        return ParametricRegressionConfig(
            poly_degree=shared_args.poly_degree,
            fit_intercept=shared_args.fit_intercept,
            alpha=shared_args.alpha,
            cv=shared_args.cv,
        )


class CoordinatesEmbeddingConfig(BaseModel):
    over_embedding: int = 0
    force_embedding: bool = False
    time_stepping: int = 1
    shift_steps: int = None  # If None, it will be calculated automatically


class NormalFormConfig(BaseModel):
    degree: int = 3
    do_scaling: bool = True
    tolerance: Optional[float] = None  # For resonance condition
    ic_style: Literal["random", "informed", "zero"] = "random"
    max_iter: int = 1000
    method: Literal["lm", "trf"] = "lm"  # TODO see what else is supported
    jac: Literal["2-point", "3-point"] = "2-point"  # TODO see what else is supported
    use_center_manifold_style: bool = True


class SSMConfig(BaseModel):
    """Configuration for the SSMLearn model."""

    dynamics_type: Literal["flow", "map"] = "flow"
    dynamics_structure: Literal["generic", "normalform"] = "generic"
    error_metric: Literal["NTE", "NMTE", "TE", "MTE"] = "NTE"
    geometry_poly_degree: int = 3
    dynamics_poly_degree: int = 5
    reconstruction_error_threshold: Optional[float] = 0.05
    optimal_model_early_stopping: bool = True
    save_directory: str = ""
    dynamics_polynomial_range: Tuple[int, int] = (3, 7)
    save_suboptimal_models: bool = False
    coordinates_embeddings_args: CoordinatesEmbeddingConfig = (
        CoordinatesEmbeddingConfig()
    )
    normalform_args: NormalFormConfig = NormalFormConfig()
    # Optional, because if not provided we will try to deduce it
    ssm_dim: Optional[int] = None
