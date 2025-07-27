import numpy as np
import matplotlib.pyplot as plt
from ssmlearnpy.utils.preprocessing import (
    estimate_enduring_frequencies,
    get_optimal_timestep,
)
from ssmlearnpy.utils.data import SSMData, SSMDataAttribute
from ssmlearnpy.geometry.oblique_projection import oblique_projection


def test_estimate_ssm_dim():
    t_max = 10
    t = np.linspace(0, t_max, 1024)
    f1 = 3 + 3 * t / t_max
    f2 = 10 + 15 * t / t_max
    f3 = 30 + 5 * -t / t_max
    signal = (
        np.sin(2 * np.pi * f1 * t)
        + np.sin(2 * np.pi * f2 * t)
        + np.sin(2 * np.pi * f3 * t)
    )

    freqs = estimate_enduring_frequencies(
        signal,
        nperseg=len(signal) // 5,  # Want a minimum of 10 windows with 50% overlap
    )

    assert len(freqs) == 3

    signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    freqs = estimate_enduring_frequencies(
        signal,
        nperseg=len(signal) // 5,
    )
    assert len(freqs) == 2

    signal = np.sin(2 * np.pi * f1 * t)
    freqs = estimate_enduring_frequencies(
        signal,
        nperseg=len(signal) // 5,
    )
    assert len(freqs) == 1


def test_optimal_timestep():
    t_max = 10
    t = np.linspace(0, t_max, 1024)
    f1 = 3 + 3 * t / t_max
    f2 = 10 + 15 * t / t_max
    f3 = 30 + 5 * -t / t_max
    signal = (
        np.sin(2 * np.pi * f1 * t)
        + np.sin(2 * np.pi * f2 * t)
        + np.sin(2 * np.pi * f3 * t)
    )

    signal = signal.reshape(1, -1)

    data = SSMData(inputs=SSMDataAttribute(data=[signal], time=[t]))

    optimal_timestep = get_optimal_timestep(data=data)
    assert optimal_timestep > 0


if __name__ == "__main__":

    # test_estimate_ssm_dim()
    try:
        test_optimal_timestep()
    except Exception as e:
        import ipdb

        print(e)

        ipdb.post_mortem()
