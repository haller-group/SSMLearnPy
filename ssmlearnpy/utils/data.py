from typing import Optional, List
from dataclasses import dataclass, field
from numpy import ndarray
from ssmlearnpy import LArr

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

    # def validate(self):
    #     assert (
    #         len(self.data) == len(self.immediate_errors) == len(self.cumulative_errors)
    #     ), (
    #         f"Predictions, immediate errors, and cumulative errors must have the same number of trajectories. "
    #         f"Found {len(self.data)} predictions, {len(self.immediate_errors)} immediate errors, and {len(self.cumulative_errors)} cumulative errors."
    #     )
    #     for p, ie, ce in zip(self.data, self.immediate_errors, self.cumulative_errors):
    #         assert len(p) == len(ie) == len(ce), (
    #             f"Each trajectory in predictions, immediate errors, and cumulative errors must have the same length. "
    #             f"Found lengths {len(p)}, {len(ie)}, and {len(ce)} respectively."
    #         )

    # TODO possibly verify dimensions


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

        # if self.input_signal:
        #     assert self.time, "Time data must be provided if input_signal is provided."
        #     assert len(self.time) == len(
        #         self.input_signal
        #     ), f"Found {len(self.time)} time trajectories but {len(self.input_signal)} signal trajectories."
        #     for t, x in zip(self.time, self.input_signal):
        #         assert len(t) == len(
        #             x
        #         ), f"Time vector and signal vector must have the same length. Found {len(t)} and {len(x)}."

        # if self.embedded_signal:
        #     assert (
        #         self.clipped_time
        #     ), "Clipped time data must be provided if embedded_signal is provided."
        #     assert len(self.clipped_time) == len(
        #         self.embedded_signal
        #     ), f"Found {len(self.clipped_time)} clipped time trajectories but {len(self.embedded_signal)} embedded signal trajectories."
        #     for t, x in zip(self.clipped_time, self.embedded_signal):
        #         assert len(t) == len(
        #             x
        #         ), f"Clipped time vector and embedded signal vector must have the same length. Found {len(t)} and {len(x)}."

        # for attr in [
        #     "reduced_coordinates",
        #     "reconstructed_reduced_coordinates",
        #     "normal_coordinates",
        #     "reconstructed_normal_coordinates",
        # ]:
        #     if getattr(self, attr) is not None:
        #         assert isinstance(
        #             getattr(self, attr), SSMDataAttribute
        #         ), f"{attr} must be an instance of SSMDataAttribute."
        #         getattr(self, attr).validate()
