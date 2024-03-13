"""
This file contains the coefficients of fits of the amplitude
to 2nd order self-force results
"""
from typing import Any, Dict

import numpy as np


def GSF_amplitude_fits(nu: float) -> Dict[str, Any]:
    """Return the GSF fit coefficients to the amplitude.
    Based only on non-spinning data.
    Note that these are multiplied by the symmetric
    mass ratio nu

    Args:
        nu (float): Symmetric mass ratio

    Returns:
        Dict: Dictionary of coefficients
    """
    coeffs_arrays = [
        21.2,
        -411.0,
        12.0,
        -215.0,
        1.65,
        26.5,
        80.0,
        -3.56,
        15.6,
        -216.0,
        -2.61,
        1.25,
        -35.7,
        0.333,
        -6.5,
        (98 - (1312549797426453052 / 176264081083715625) / nu),
        (18778864 / 12629925) / nu,
        -0.654,
        -3.69,
        18.5 - (2465107182496333 / 460490801971200) / nu,
        (174381 / 67760) / nu,
    ]

    (
        h22_v8,
        h22_v10,
        h33_v8,
        h33_v10,
        h21_v6,
        h21_v8,
        h21_v10,
        h44_v6,
        h44_v8,
        h44_v10,
        h55_v4,
        h55_v6,
        h55_v8,
        h32_v6,
        h32_v8,
        h32_v10,
        h32_vlog10,
        h43_v4,
        h43_v6,
        h43_v8,
        h43_vlog8,
    ) = nu * np.array(coeffs_arrays)
    result_dict = {
        "h22_v8": h22_v8,
        "h22_v10": h22_v10,
        "h33_v8": h33_v8,
        "h33_v10": h33_v10,
        "h21_v6": h21_v6,
        "h21_v8": h21_v8,
        "h21_v10": h21_v10,
        "h44_v6": h44_v6,
        "h44_v8": h44_v8,
        "h44_v10": h44_v10,
        "h55_v4": h55_v4,
        "h55_v6": h55_v6,
        "h55_v8": h55_v8,
        "h32_v6": h32_v6,
        "h32_v8": h32_v8,
        "h32_v10": h32_v10,
        "h32_vlog10": h32_vlog10,
        "h43_v4": h43_v4,
        "h43_v6": h43_v6,
        "h43_v8": h43_v8,
        "h43_vlog8": h43_vlog8,
    }
    return result_dict
