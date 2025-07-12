"""
Utils for SSM computation, including utils related to oblique projection
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
import warnings
from typing import List, Tuple
import matplotlib.pyplot as plt
import scipy.signal as signal
import polars as pl
import ipdb
from logic.ssm_new import get_optimal_timestep
from ssmlearnpy.geometry.coordinates_embedding import coordinates_embedding
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import minimize
import numba
from numba.extending import register_jitable
import nlopt
import time


# @register_jitable(numba.types.Array(numba.types.float64, 1, "C"), numba.types.int64)
@register_jitable
def _find_peaks(x, min_dist_samples) -> np.ndarray:
    return find_peaks(x, distance=min_dist_samples)[0]


# @numba.njit
def pffk(t: np.ndarray, x: np.ndarray, plot=False):
    """
    Estimates instantaneous amplitude, frequency, and damping from a signal.

    Implementation based on the concepts likely described in:
    M. Jin, W. Chen, M. R. W. Brake, and H. Song. Identification of
    instantaneous frequency and damping from transient decay data. Journal of
    Vibration and Acoustics, 142(5):051111.

    Args:
        t (np.ndarray): 1D array of time points.
        x (np.ndarray): 1D array of signal values corresponding to t.
        kmean (int): Window size for moving average smoothing.
        filter_freq_hz (tuple, optional): A tuple (low_hz, high_hz) for
                                           bandpass filtering the signal before
                                           processing. Defaults to None (no filtering).

    Returns:
        tuple: A tuple containing:
            - amp (np.ndarray): Instantaneous amplitude at peak times.
            - freq (np.ndarray): Smoothed instantaneous frequency interpolated at peak times.
            - peak_times (np.ndarray): Time points corresponding to amp, freq, damp.
            - t_zeros (np.ndarray): Time points of zero crossings.

    Raises:
        ValueError: If inputs are not 1D arrays or have mismatching shapes.
        ValueError: If kmean is not a positive integer.
        ValueError: If filter_freq_hz is provided but not as a tuple of two numbers.
        RuntimeWarning: If processing steps fail (e.g., too few zero crossings or peaks).
    """

    # --- Input Validation and Preparation ---
    t = t.squeeze()
    x = x.squeeze()
    # if not isinstance(t, np.ndarray) or t.ndim != 1:
    #     raise ValueError("Input 't' must be a 1D NumPy array.")
    # if not isinstance(x, np.ndarray) or x.ndim != 1:
    #     raise ValueError("Input 'x' must be a 1D NumPy array.")
    # if t.shape != x.shape:
    #     raise ValueError("Inputs 't' and 'x' must have the same shape.")
    # if not isinstance(kmean, int) or kmean <= 0:
    #     raise ValueError("'kmean' must be a positive integer.")

    t = t.astype(float)
    x = x.astype(float)

    # --- Zero Crossing Detection ---
    sign_change = np.where(np.diff(np.sign(x)))[0]  # Indices *before* sign change

    # if len(sign_change) < 5:
    #     raise ValueError(
    #         "Fewer than 5 zero crossings detected, algorithm not well defined."
    #     )

    # Linear interpolation for zero crossings
    t_a = t[sign_change]
    x_a = x[sign_change]
    t_b = t[sign_change + 1]
    x_b = x[sign_change + 1]

    # TODO verify
    # Linear interpolation to find zero crossing times
    t_zeros = t_a - x_a * (t_b - t_a) / (x_b - x_a)

    # Estimate period as twice the time difference between zero crossings
    periods = np.diff(t_zeros) * 2

    # Truncate the signal to the first and last zero crossing, before peak finding
    # and interpolation
    start_idx = sign_change[0]
    end_idx = sign_change[-1] + 1
    t_proc = t[start_idx : end_idx + 1]
    x_proc = x[start_idx : end_idx + 1]

    # --- Peak Finding (Envelope) ---
    xa = np.abs(x_proc)

    # Use half the average zero crossing interval as min peak distance
    # Ensure distance is at least 1 sample
    min_dist_samples = max(1, int(np.round(np.mean(np.diff(sign_change)) / 2.0)))

    peak_indices_rel = _find_peaks(xa, min_dist_samples)

    # --- Quadratic Interpolation for Peak Refinement ---
    refined_amps = []
    refined_times = []

    for idx in peak_indices_rel:
        # Get the 3 points for interpolation
        t_segment = t_proc[idx - 1 : idx + 2]
        x_segment = xa[idx - 1 : idx + 2]  # Use absolute values for envelope peak

        # Create matrix A for Ax = B (fitting y = at^2 + bt + c)
        # Use relative time starting from 0 for numerical stability
        t_rel = t_segment - t_segment[0]
        A = np.vstack([t_rel**2, t_rel, np.ones(3)]).T

        # Solve for coefficients [a, b, c]
        P = np.linalg.solve(A, x_segment)
        a, b, c = P[0], P[1], P[2]

        # Find time and amplitude of vertex
        # If quadratic (a != 0), vertex is at t = -b/(2a)
        if abs(a) > 1e-10:
            t_vertex_rel = -b / (2 * a)
            t_vertex_abs = t_vertex_rel + t_segment[0]
            amp_vertex = a * t_vertex_rel**2 + b * t_vertex_rel + c

            # Only accept if the vertex time is within the segment bounds
            if t_segment[0] <= t_vertex_abs <= t_segment[-1]:
                refined_times.append(t_vertex_abs)
                refined_amps.append(amp_vertex)
        else:
            # If 'a' is near zero, it's nearly linear, peak is likely at center point
            refined_times.append(t_proc[idx])
            refined_amps.append(xa[idx])

    peak_times = np.array(refined_times)
    amp = np.array(refined_amps)

    # Calculate Frequency from periods
    freq = 1.0 / periods

    # Frequency is defined at midpoints between zero crossings
    t_freq = t_zeros[:-1] + np.diff(t_zeros) / 2.0

    # Interpolate to approximate the frequency at the peak times
    freq_peaks = np.interp(peak_times, t_freq, freq)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(t_proc, x_proc, label="Signal")
        plt.plot(t_zeros, np.zeros_like(t_zeros), "ro", label="Zero Crossings")
        plt.plot(peak_times, amp, "go", label="Peaks")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title("Signal with Zero Crossings and Peaks")
        plt.legend()
        plt.grid(True)
        plt.show()

    return amp, freq_peaks, peak_times


def frequency_analysis(
    x: np.ndarray,
    t: np.ndarray,
    num_windows: int = 10,
    epsilon: float = 0.1,
    min_peak_distance: int = 10,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Pythonic version of MATLAB spectrogram analysis with peak detection

    Args:
        x: Input signal (1D array)
        t: Time vector (1D array)
        nwin: Window size for STFT
        epsilon: Relative threshold for peak detection (0-1)
        min_peak_distance: Minimum samples between peaks
        plot: Whether to generate plots

    Returns:
        stft: Short-time Fourier transform (2D array)
        frequencies: Frequency vector (1D array)
        dominant_freqs: List of dominant frequencies at each time point
    """
    x = x.squeeze()
    window_length = int(len(x) / num_windows)
    # Compute STFT (similar to MATLAB's spectrogram)

    w = signal.windows("hann", window_length)

    SFT = signal.ShortTimeFFT()

    fs = 1 / (t[1] - t[0])  # Sampling frequency
    f, t_stft, stft = signal.stft(
        x,
        fs=fs,
        nperseg=window_length,
        return_onesided=False,
    )

    # Convert to power spectral density (similar to MATLAB output)
    power_density = np.abs(stft) ** 2
    frequencies = 2 * np.pi * f  # Convert to rad/s to match MATLAB

    # Find dominant frequencies at each time point
    dominant_freqs = []
    max_pks = None

    for i in range(len(t_stft)):
        slice_pd = power_density[:, i]
        peaks, props = find_peaks(slice_pd, distance=min_peak_distance)

        # Set threshold based on first time point
        if i == 0:
            max_pks = np.max(slice_pd[peaks]) if len(peaks) > 0 else 0
            min_pks = epsilon * max_pks

        # Select dominant peaks
        dominant_mask = slice_pd[peaks] >= min_pks
        dominant_peaks = peaks[dominant_mask]
        dominant_freqs.append(frequencies[dominant_peaks])

        # Optional plotting
        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(frequencies, slice_pd, "k.-", markersize=5, label="Spectrum")
            plt.plot(frequencies[peaks], slice_pd[peaks], "ro", label="All peaks")
            if len(dominant_peaks) > 0:
                plt.plot(
                    frequencies[dominant_peaks],
                    slice_pd[dominant_peaks],
                    "bo",
                    label="Dominant peaks",
                )
            plt.xlabel("Frequency [rad/s]")
            plt.ylabel("Power spectral density [1/Hz]")
            plt.legend()
            plt.title(f"Time = {t_stft[i]:.2f}s")
            plt.grid(True)
            plt.show()

    return stft, frequencies, dominant_freqs


def linear_regime(freqs, lim=0.1):

    # plot_data([i for i in range(len(freqs))], freqs)
    cum_vec = np.empty_like(freqs)
    cum_vec[0] = np.sum(freqs)
    cum_vec[1:] = cum_vec[0] - np.cumsum(freqs)[:-1]

    mean_vect = cum_vec / np.arange(len(freqs), 0, -1)
    # plot_data([i for i in range(len(mean_vect))], mean_vect)
    diff_vect = np.abs(np.diff(mean_vect))
    # plot_data(np.arange(len(diff_vect)), np.log(diff_vect))
    # ipdb.set_trace()
    # plot_data([i for i in range(len(diff_vect))], diff_vect)
    indices = np.where(diff_vect < lim)[0]
    while indices.size == 0:
        lim = np.exp(np.log(lim) + 1)
        indices = np.where(diff_vect < lim)[0]
    return indices[0]


def extract_linear_regime(t, x, lim=0.1):
    amp, freq, peak_times = pffk(t.squeeze(), x.squeeze())
    # plot_data(2 * np.pi * freq, amp)
    idx = linear_regime(freq, lim=5e-9)
    peak_time = peak_times[idx]
    if idx > 0:
        start_idx = np.where(t >= peak_time)[0][0]
    else:
        start_idx = 0
    # print(start_idx)
    end_index = 2 * start_idx
    t_trunc = t.copy()[start_idx:end_index]
    x_trunc = x.copy()[start_idx:end_index]
    return t_trunc, x_trunc


def plot_data(t, x):
    """
    Plot the time series data.

    Args:
        t (np.ndarray): Time vector.
        x (np.ndarray): Signal data.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(t, x)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Time Series Data")
    plt.grid(True)
    plt.show()


def scatter(x, y):
    """
    Scatter plot of x and y data.

    Args:
        x (np.ndarray): x data.
        y (np.ndarray): y data.
    """
    plt.figure(figsize=(10, 4))
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot")
    plt.grid(True)
    plt.show()


def oblique_projection(t, x):
    t, x = extract_linear_regime(t, x)
    xData = to_ssmlearn_format([t], [x])
    # lag = get_optimal_timestep(xData)
    # lag = 15
    lag = 1
    t, y, _ = coordinates_embedding(
        [t.ravel()], [x.reshape(1, -1)], imdim=2, shift_steps=lag, over_embedding=1
    )

    t = t[0]
    y = y[0]
    E = dmd(t, y)
    B = compute_B_nlopt(t, y, E)


def dmd(t, y):
    V1 = y[:, :-1]
    V2 = y[:, 1:]

    # print(y.shape)

    # svd = TruncatedSVD(n_components=4)
    # U = svd.fit_transform(V1)
    # print(V1[:, :5])
    U, D, V = np.linalg.svd(V1, full_matrices=False)
    # print(U)
    # print(D)

    A_tilde = U.T @ V2 @ V.T @ np.diag(1 / D)
    # print(A_tilde)

    # A_tilde = U.T @ V2 @ svd.components_.T @ np.diag(1 / svd.singular_values_)

    lamb, V = np.linalg.eig(A_tilde)
    # Convert discrete-time eigenvalues to continuous-time
    lamb = np.log(lamb) / (t[1] - t[0])
    indices = np.argsort(np.abs(np.real(lamb)))
    lamb = lamb[indices]
    V = V[:, indices]
    Q = U @ V
    E = np.empty((Q.shape[0], 2))
    E[:, 0] = np.real(Q[:, 0])
    E[:, 1] = np.imag(Q[:, 0])
    # print(E)
    return E


def backbone_var(t, y, return_max=False):
    var = np.zeros((y.shape[0]))
    for i in range(y.shape[0]):
        amp, freq, _ = pffk(t.squeeze(), y[i, :])
        # There is some logic in the matlab code that filters out very
        # low amplitudes, but it is incorrectly implemented there so likely
        # not important
        var[i] = np.var(freq)
    if return_max:
        return np.max(var)
    return np.sum(var)


def compute_P(t, y, Q):

    ref_var = backbone_var(t, y, return_max=True)

    def objective(B):
        B = B.reshape(-1, 2)
        y_p = Q @ np.linalg.solve(B.T @ Q, B.T) @ y
        var = backbone_var(t, y_p) / ref_var
        # print(var)
        return var

    res = minimize(
        objective,
        Q.copy().flatten(),
        method="BFGS",
        options={"disp": True, "maxiter": 100},
    )

    B = res.x.reshape(-1, 2)
    return Q @ np.linalg.solve(B.T @ Q, B.T)


def compute_B_nlopt(t, y, Q):

    ref_var = backbone_var(t, y, return_max=True)
    n = Q.flatten().shape[0]
    # TODO This is to correlate correctly with MATLAB, find out why and if necessary
    # NOTE: Doesn't impact convergence
    Q = -Q
    print(Q)
    # ipdb.set_trace()

    def objective(B, grad):
        B = B.reshape(-1, 2)
        y_p = Q @ np.linalg.solve(B.T @ Q, B.T) @ y
        var = backbone_var(t, y_p) / ref_var
        # print(var)
        return var

    # algorithms = [
    #     nlopt.LN_BOBYQA,
    #     nlopt.LN_NEWUOA,
    #     nlopt.LN_SBPLX,
    #     nlopt.LN_COBYLA,
    # ]
    # algorithms = [nlopt.LD_LBFGS]
    algorithms = [nlopt.LN_BOBYQA]

    for algorithm in algorithms:
        opt = nlopt.opt(algorithm, n)
        opt.set_min_objective(objective)
        opt.set_xtol_rel(1e-6)
        opt.set_ftol_rel(1e-6)
        opt.set_ftol_abs(1e-6)
        opt.set_maxeval(100 * n)
        start_time = time.time()
        x_opt = opt.optimize(Q.copy().flatten())
        B = x_opt.reshape(-1, 2)
        P = Q @ np.linalg.solve(B.T @ Q, B.T)
        print(
            f"Algorithm: {nlopt.algorithm_name(algorithm)}, Time: {time.time() - start_time:.2f} seconds"
        )
        print(f"Optimal B: {B}")
        print(f"Optimal P: {P}")
        # print(f"Optimal x: {x_opt}")
        print(f"Objective value: {opt.last_optimum_value()}")
        print(f"Number of iterations: {opt.get_numevals()}")


def to_ssmlearn_format(t, x):
    return [[_t.reshape(1, -1), _x.reshape(1, -1)] for _t, _x in zip(t, x)]


def main():
    # test_file = "oblique_data/nonlinear_beam.csv"
    test_file = "data.csv"
    df = pl.read_csv(test_file, has_header=False)
    t = df.select(pl.nth(0)).to_numpy()
    x = df.select(pl.nth(1)).to_numpy()
    oblique_projection(t, x)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        ipdb.post_mortem()
