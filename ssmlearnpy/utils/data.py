from typing import Optional, List
from dataclasses import dataclass, field
from numpy import ndarray
from ssmlearnpy import LArr
import numpy as np

prev_attribute_map = {
    "reduced_coordinates": "embedded",
    "advected_reduced_coordinates": "reduced_coordinates",
    "normal_coordinates": "reduced_coordinates",
    "advected_normal_coordinates": "normal_coordinates",
}

@dataclass
class SSMDataAttribute:

    data: LArr = field(default_factory=list)
    reconstructed_prev: LArr = field(default_factory=list)
    reconstructed_embedding: LArr = field(default_factory=list)
    immediate_errors: LArr = field(default_factory=list)
    cumulative_errors: LArr = field(default_factory=list)
    time: LArr = field(default_factory=list)

    def empty(self) -> bool:
        """
        Check if the data attribute is empty.
        """
        return (
            len(self.data) == 0
            and len(self.reconstructed_prev) == 0
            and len(self.immediate_errors) == 0
            and len(self.cumulative_errors) == 0
            and len(self.time) == 0
        )

    def clear(self):
        """
        Clear the data attribute.
        """
        self.data = []
        self.reconstructed_prev = []
        self.reconstructed_embedding = []
        self.immediate_errors = []
        self.cumulative_errors = []
        self.time = []

    def get_data_matrix(self):
        """
        Convert the list of signals in the data attribute to a single matrix
        """

        if len(self.data) == 0:
            return None

        return np.concatenate([np.array(d).reshape(1, -1) for d in self.data], axis=1)
    
    def validate(self):
        if self.time:
            for traj in self.time:
                assert len(traj.shape) == 1, "Time should be a 1D array."


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

    inputs: SSMDataAttribute = field(default_factory=SSMDataAttribute)

    # Only used if oblique_projection is selected
    # linear_regime: SSMDataAttribute = field(default_factory=SSMDataAttribute)

    embedded: SSMDataAttribute = field(default_factory=SSMDataAttribute)

    # Outputs from projection onto manifold
    reduced_coordinates: SSMDataAttribute = field(default_factory=SSMDataAttribute)
    # Outputs from advection of polynomial RHS
    advected_reduced_coordinates: SSMDataAttribute = field(
        default_factory=SSMDataAttribute
    )

    # Outputs from near identity transformation to normal coordinates
    normal_coordinates: SSMDataAttribute = field(default_factory=SSMDataAttribute)
    # Outputs from advection of normal-form RHS
    advected_normal_coordinates: SSMDataAttribute = field(
        default_factory=SSMDataAttribute
    )

    # Miscellaneous
    regression_params: Optional[List] = field(default_factory=list)

    def __post_init__(self):
        """
        Validate the data after initialization.
        """
        self.validate()

    def validate(self):
        """
        Validate the data in the SSMData object.
        """
        for attr in self.__dataclass_fields__:
            if attr != "regression_params":
                assert isinstance(
                    getattr(self, attr), SSMDataAttribute
                ), f"SSMData attribute '{attr}' must be of type SSMDataAttribute."
                getattr(self, attr).validate()

            if attr == "inputs":
                if getattr(self, attr).data:
                    for traj in getattr(self, attr).data:
                        assert(len(traj.shape)==2), "Input signal must be a 2D array"
                        assert(traj.shape[0]<traj.shape[1]), "Shape must be like (signal_dim, signal_length)"
