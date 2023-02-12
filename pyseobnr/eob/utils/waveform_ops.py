"""
Additional utility functions.
"""

from typing import Any,Dict
import numpy as np

def frame_inv_amp(modes:Dict[Any,Any],ell_max:int=2,use_symm=True)->np.ndarray:
    """Compute the frame-invariant amplitude.
    By default, assumes that we have aligned-spin symmetry.

    Args:
        modes (Dict[Any,Any]): The dictionary of modes
        ell_max (int): The maximum l to use
        use_symm (bool): Assume up-down aligned symmetry

    Returns:
        np.ndarray: The frame-invariant ampltiude time series
    """
    total = 0.0

    for mode in modes.keys():
        ell,m = mode
        if ell>ell_max:
            continue
        if use_symm:
            total+=2*np.abs(modes[mode])**2
        else:
            total+=np.abs(modes[mode])**2
    return np.sqrt(total)
