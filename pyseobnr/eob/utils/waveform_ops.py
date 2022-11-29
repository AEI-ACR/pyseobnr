from typing import Any,Dict
import numpy as np

def frame_inv_amp(modes:Dict[Any,Any],ell_max:int=2)->np.ndarray:
    """Compute the frame-invariant amplitude

    Args:
        modes (Dict[Any,Any]): The dictionary of modes
        ell_max (int): The maximum l to use

    Returns:
        np.ndarray: The frame-invariant ampltiude time series
    """
    total = 0.0
    for mode in modes.keys():
        ell,m = mode
        if ell>ell_max:
            continue
        total+=np.abs(modes[mode])**2
    return np.sqrt(total)
