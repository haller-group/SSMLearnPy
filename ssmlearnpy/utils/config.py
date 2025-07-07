from pydantic import BaseModel
from typing import Literal, Optional, Dict


class CoordinatesEmbeddingConfig(BaseModel):
    over_embedding: int = 0
    force_embedding: bool = False
    time_stepping: int = 1
    shift_steps: int = 1


class NormalFormConfig(BaseModel):
    degree: int = 3
    do_scaling: bool = True
    tolerance: Optional[float] = None  # For resonance condition
    ic_style: Literal["random", "informed", "zero"] = "random"
    max_iter: int = 1000
    method: Literal["lm"] = "lm"  # TODO see what else is supported
    jac: Literal["2-point"] = "2-point"  # TODO see what else is supported
    use_center_manifold_style: bool = True


class SSMConfig(BaseModel):
    """Configuration for the SSMLearn model."""

    dynamics_type: Literal["flow", "map"] = "flow"
    dynamics_structure: Literal["generic", "normalform"] = "generic"
    error_metric: Literal["NTE", "NMTE", "TE", "MTE"] = "NTE"
    geometry_poly_degree: int = 3
    dynamics_poly_degree: int = 5
    reconstruction_error_threshold: Optional[float] = 0.1
    optimal_model_early_stopping: bool = True
    save_directory: str = ""
    save_suboptimal_models: bool = False
    coordinates_embeddings_args: CoordinatesEmbeddingConfig = (
        CoordinatesEmbeddingConfig()
    )
    normalform_args: NormalFormConfig = NormalFormConfig()
    # Optional, because if not provided we will try to deduce it
    ssm_dim: Optional[int] = None
