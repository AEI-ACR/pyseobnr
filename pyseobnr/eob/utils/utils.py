from typing import cast

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import argrelmin


def iterative_refinement(f, interval, levels=2, dt_initial=0.1, pr=False):
    """
    Attempts to find the peak of Omega/pr iteratively.
    Needed to ensure accurate attachment when the attachment point is the last point of the dynamics.

    Args:
        f (PPoly): derivative of the splined dynamics
        interval (list): interval where to look for the peak
        levels (int): number of iterations (should be greater than 2)
        dt_initial (float): initial time step
        pr (bool): whether the end of the dynamics is due to a peak of pr instead of Omega

    Returns:
        np.array: interpolated dynamics array

    """

    assert levels > 1

    left = interval[0]
    right = interval[1]
    result = None

    for n in range(1, levels + 1):
        dt = dt_initial / (10**n)
        t_fine = np.arange(interval[0], interval[1], dt)
        deriv = np.abs(f(t_fine))

        mins = argrelmin(deriv, order=3)[0]
        if len(mins) > 0:
            result = t_fine[mins[0]]

            interval = max(result - 10 * dt, left), min(result + 10 * dt, right)

        else:
            if pr:
                return interval[-1]
            else:
                return (interval[0] + interval[-1]) / 2

    assert result is not None
    return result


def interpolate_dynamics(dyn_fine, dt=0.1, peak_omega=None, step_back=250.0):
    """
    Interpolate the dynamics to a finer grid.
    This replaces stepping back that was used in older EOB models.

    Args:
        dyn_fine (np.array): dynamics array
        dt (float): time step to which to interpolate
        peak_omega (float): position of the peak (stopping condition for the dynamics)
        step_back (float): step back relative to the end of the dynamics

    Returns:
        np.array: interpolated dynamics array

    """

    if peak_omega:
        start = max(peak_omega - step_back, dyn_fine[0, 0])
        t_new = np.arange(start, peak_omega, dt)

    else:
        t_new = np.arange(dyn_fine[0, 0], dyn_fine[-1, 0], dt)

    intrp = CubicSpline(dyn_fine[:, 0], dyn_fine[:, 1:])
    res = intrp(t_new)

    return np.c_[t_new, res]


def estimate_time_max_amplitude(
    time: np.array, amplitude: np.array, delta_t: float, precision=0.001
) -> float:
    assert time.shape == amplitude.shape
    # the knots are calculated globally, but we may consider a local one around
    # the initial guess of the max
    amplitude_interpolated = CubicSpline(time, amplitude)
    t_max_coarse = time[np.argmax(amplitude)]

    t_fine_peak = np.arange(t_max_coarse - delta_t, t_max_coarse + delta_t, precision)
    amplitude_interpolated_eval = amplitude_interpolated(t_fine_peak)
    idx_max_fine = np.argmax(amplitude_interpolated_eval)
    return cast(float, t_fine_peak[idx_max_fine])
