#!/usr/bin/env python
"""This file contains the default settings for various models
Each default setting is a *function* that returns a dictionary.
To be used by models in models.py
"""

from typing import Any, Dict

import lal


def default_NR_LVC_settings() -> Dict[Any, Any]:
    """Return the default settings for the LVC NR approximant

    Returns:
        Dict[Any]: settings dictionary
    """
    settings = dict(
        M=50.0,  # Total mass in solar masses
        distance=400 * lal.PC_SI * 1e6,  # Distance in meters
        dt=2.4627455127717882e-05,  # Sampling rate, in SI units
        modes=[(2, 2)],  # Active modes
    )
    return settings


def default_NR_SXS_settings() -> Dict[Any, Any]:
    """Return the default settings for SXS NR waveforms

    Returns:
        Dict[Any]: settings dictionary
    """
    settings = dict(
        modes=[(2, 2)],  # Active modes
        extrap_order=2,  # Extrapolation order
        alignment_interval_start=2,  # Which peak in Re(h22) is the start
        alignment_interval_end=12,  # Which peak in Re(h22) is the end
        dt=0.1,
    )
    return settings


def default_NR_RIT_settings() -> Dict[Any, Any]:
    """Return the default settings for SXS NR waveforms

    Returns:
        Dict[Any]: settings dictionary
    """
    settings = dict(
        modes=[(2, 2)],  # Active modes
        alignment_interval_start=2,  # Which peak in Re(h22) is the start
        alignment_interval_end=6,  # Which peak in Re(h22) is the end
    )
    return settings


def default_SEOBNRv4HM_settings() -> Dict[Any, Any]:
    """Return the default settings for the SEOBNRv4 approximant

    Returns:
        Dict[Any]: settings dictionary
    """
    settings = dict(
        M=50.0,  # Total mass in solar masses
        distance=400 * lal.PC_SI * 1e6,  # Distance in meters
        dt=2.4627455127717882e-05,  # Sampling rate, in SI units
        debug=False,  # Run in debug mode
    )
    return settings


def default_NRHybSur3dq8_settings() -> Dict[Any, Any]:
    """Return the default settings for the NRHybSur3dq8 approximant

    Returns:
        Dict[Any]: settings dictionary
    """
    settings = dict(
        M=50.0,  # Total mass in solar masses
        dt=0.1,  # Sampling rate in geometric units
        alignment_interval_start=2,  # Which peak in Re(h22) is the start
        alignment_interval_end=12,  # Which peak in Re(h22) is the end
    )
    return settings


def default_NRHybSur2dq15_settings() -> Dict[Any, Any]:
    """Return the default settings for the NRHybSur2dq15 approximant

    Returns:
        Dict[Any]: settings dictionary
    """
    settings = dict(
        M=50.0,  # Total mass in solar masses
        dt=0.1,  # Sampling rate in geometric units
        alignment_interval_start=2,  # Which peak in Re(h22) is the start
        alignment_interval_end=12,  # Which peak in Re(h22) is the end
    )
    return settings


def default_unfaithfulness_phys_v4_settings() -> Dict[Any, Any]:
    settings = dict(
        sigma=0.001,  # the sigma to use in likelihood
    )
    return settings


def default_amplitude_fractional_difference_settings() -> Dict[Any, Any]:
    settings = dict(
        fraction=0.1,  # fraction of the waveform starting at peak going back to consider
        sigma=0.005,  # the sigma to use in likelihood
    )
    return settings


def default_unfaithfulness_flat_settings() -> Dict[Any, Any]:
    settings = dict(sigma=0.01, leading_order=True)
    return settings


def default_delta_time_to_merger_settings() -> Dict[Any, Any]:
    settings = dict(sigma=5)
    return settings


def default_metric_list() -> Dict[Any, Any]:
    return []
    # return [' UnfaithfulnessPhysV4','DeltaTimeToMerger']
