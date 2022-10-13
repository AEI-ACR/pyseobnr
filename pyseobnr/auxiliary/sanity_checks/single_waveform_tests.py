from typing import Any, Dict

import numpy as np
from scipy.signal import argrelmax


def one_peak_test(t: np.ndarray, h: np.ndarray, interval_end: float) -> None:
    """Check that a quantity has only 1 peak until
    interval_end

    Args:
        t (np.ndarray): The time array
        h (np.ndarray): The quantity as function of tim
        interval_end (float): Last time to consider
    """
    times = np.where(t <= interval_end)
    t = 1 * t[times]
    h = 1 * h[times]
    maxs = argrelmax(h)
    global_max = np.argmax(h)
    assert len(maxs[0]) == 1, "Should only have 1 max in this quantity"
    assert maxs[0] == global_max, f"Local max is not global max,{maxs[0]},{global_max}"


def monotonic_quantity_test(
    t: np.ndarray, h: np.ndarray, interval_end: float, threshold: float = 1e-10
) -> None:
    """Check that thie give quantity is monotonic until interval_end,
    with small deviations allowed, quantified by threshold parameter

    Args:
        t (np.ndarray): The time array
        h (np.ndarray): Quantity as a function of tim
        interval_end (float): Last time to consider
        threshold (float, optional): Allowed deviation from monotonicity. Defaults to 1e-10.
    """
    times = np.where(t <= interval_end)
    t = 1 * t[times]
    h = 1 * h[times]
    forward_diffs = np.diff(h)

    # Allow for small inaccuracies in monotonicity
    conds = (forward_diffs > 0) | (np.abs(forward_diffs) < threshold)

    assert np.alltrue(conds), "The quantity has to be strictly monotonic"


def amplitude_hierarchy_test(t: float, modes: Dict[Any, Any]) -> None:
    """Check that the amplitude of the subdominant modes is
    less than the amplitude of the (2,2) mode until the peak
    of the (2,2) modes

    Args:
        t (np.array): Times
        modes (Dict): The dictionary of modes
    """
    h22 = modes["2,2"]
    amp22 = np.abs(h22)
    interval_end = t[np.argmax(amp22)]
    times = np.where(t <= interval_end)[0]
    amp22 = amp22[times]
    for ell_m in modes.keys():
        if ell_m == "2,2":
            continue
        current_mode = modes[ell_m]
        amp_mode = np.abs(current_mode)
        amp_mode = amp_mode[times]
        assert np.alltrue(
            amp_mode < amp22
        ), f"Amplitude of mode {ell_m} should be less than the (2,2) mode before merger"

