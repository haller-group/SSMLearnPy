from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class SSMDataAttribute:
    """
    Class to

    Attributes:
        predictions: Trajectories at the current stage of the pipeline.
        immediate_errors: Errors resulting from the immediately preceding transformation.
        cumulative_errors: Errors resulting from the entire pipeline up to the current stage.
    """

    predictions: Optional[List] = field(default_factory=list)
    immediate_errors: Optional[List] = field(default_factory=list)
    cumulative_errors: Optional[List] = field(default_factory=list)


@dataclass
class SSMData:
    """
    Class to abstract data handling from model training and prediction.

    Attributes:
        input_data: Raw input time series.
        emb_data: Time series in time-delay embedding space (reconstructed phase space).
        geometry_predictions: Time series in .
        reduced_dynamics_predictions: Outputs from reduced-order dynamics models.
    """

    # Inputs to the model
    time: Optional[List] = field(default_factory=list)
    input_signal: Optional[List] = field(default_factory=list)

    # Outputs from time-delay embedding
    clipped_time: Optional[List] = field(default_factory=list)
    embedded_signal: Optional[List] = field(default_factory=list)

    # Outputs from projection onto manifold
    reduced_coordinates: Optional[SSMDataAttribute] = None
    # Outputs from advection of polynomial RHS
    reconstructed_reduced_coordinates: Optional[SSMDataAttribute] = None

    # Outputs from near identity transformation to normal coordinates
    normal_coordinates: Optional[SSMDataAttribute] = None
    # Outputs from advection of normal-form RHS
    reconstructed_normal_coordinates: Optional[SSMDataAttribute] = None

    # Miscellaneous
    regression_params: Optional[List] = field(default_factory=list)
