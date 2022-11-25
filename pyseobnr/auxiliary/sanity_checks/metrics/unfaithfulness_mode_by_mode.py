from functools import lru_cache
from typing import Union

import lal
import numpy as np
from pycbc.filter import make_frequency_series
from pycbc.filter.matchedfilter import match
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.psd.analytical import aLIGOZeroDetHighPowerGWINC, EinsteinTelescopeP1600143, CosmicExplorerP1600143

from scipy.signal import argrelmax
from waveform_tools.mismatch.unfaithfulness import generate_waveform
from waveform_tools.mismatch.waveform_parameters import waveform_params
from waveform_tools.mismatch.auxillary_funcs import Ylm
import matplotlib.pyplot as plt


def _planckWindowFunction(t, tol=0.005):
    """f(t) = 0 if t < tol
    = 1 if t > 1 - tol
    = 1/(1 + exp(z)) otherwise, where z = 1/t - 1/(1-t)"""
    safe = (t > tol) * (t < (1.0 - tol))
    # temporarily set unsafe times to 0.5 to avoid dividing by 0
    safeT = safe * t + (1.0 - safe) * 0.5
    safeZ = 1.0 / safeT - 1.0 / (1.0 - safeT)
    return safe * 1.0 / (1.0 + np.exp(safeZ)) + (t >= (1.0 - tol))


def window_function(t, t_s=0.0, t_e=1.0, rolloff=False):
    """Return a Planck window function, evaluated at an array of times t:
    When t <= t_s, f(t) = 0
    When t >= t_e, f(t) = 1
    Otherwise, 0 <= f(t) <= 1, where f is monotonically increasing.

    If rolloff=True, reverse the window so that f(t_s) = 1, f(t_e) = 0.

    This is a modified version of the code from gwtools by C.Galley
    """
    if rolloff:
        return window_function(-t, t_s=-t_e, t_e=-t_s)

    # Rescale the time so that the time interval of interest is always
    # 0 to 1
    scaled_t = (t - t_s) / (t_e - t_s)
    return _planckWindowFunction(scaled_t)


def taper_waveform(ts, h, rollon_end=4, ending_time=35):
    maxs = argrelmax(h, order=20)[0]
    # Roll-on interval
    t_l1 = ts[0]
    t_l2 = ts[maxs[rollon_end]]

    # Roll-off interval
    t_h1 = ts[-1] - ending_time
    t_h2 = ts[-1]

    # Roll-on window
    window1 = window_function(ts, t_s=t_l1, t_e=t_l2)
    # Roll-off window
    window2 = window_function(ts, t_s=t_h1, t_e=t_h2, rolloff=True)
    return h * window1 * window2


def taper_pycbc_series(time_series, rollon_end=4):
    ts = time_series.sample_times.data
    h = time_series.numpy()
    return TimeSeries(
        taper_waveform(ts, h, rollon_end=rollon_end), delta_t=time_series.delta_t
    )


def get_padded_length(h):
    N = len(h)
    return int(2 ** (np.floor(np.log2(N)) + 2))


def condition_waveform(t, h, n=None, convert=True):
    # Taper
    h = taper_waveform(t, h)
    # Pad

    if n is None:
        n_pad = get_padded_length(h)
    else:
        n_pad = int(2 ** (np.floor(np.log2(n)) + 2))
    h.resize(n_pad)
    if convert:
        h = TimeSeries(h, delta_t=np.diff(t)[0])
    return h


def condition_pycbc_series(h, n=None):
    # Taper
    h = taper_pycbc_series(h)
    # Pad

    if n is None:
        n_pad = get_padded_length(h)
    else:
        n_pad = int(2 ** (np.floor(np.log2(n)) + 2))
    h.resize(n_pad)
    return h


@lru_cache(maxsize=128)
def generate_psd(
    length: int, delta_f: float, flow: float, psd_type: str = "aLIGO"
) -> FrequencySeries:
    """A memoized version of different PSDs
    Available:
    'aLIGO' - aLIGOZeroDetHighPowerGWINC
    'flat' - unity PSD


    Args:
        length (int): length of the PSD to generate
        delta_f (float): frequency spacing
        flow (float): low frequency cutoff

    Returns:
        pt.FrequencySeries: the PSD
    """
    if psd_type == "aLIGO":
        return aLIGOZeroDetHighPowerGWINC(length, delta_f, flow)
    elif psd_type == "ET":
        return EinsteinTelescopeP1600143(length, delta_f, flow)
    elif psd_type == "CE":
        return CosmicExplorerP1600143(length, delta_f, flow)
    elif psd_type == "flat":
        return FrequencySeries(np.ones(length), delta_f=delta_f)
    else:
        raise NotImplementedError


def fast_unfaithfulness_mode_by_mode(
    h1: Union[TimeSeries, FrequencySeries],
    h2: Union[TimeSeries, FrequencySeries],
    f_low: float,
    Ms: np.ndarray,
    fmin: float = 10.0,
    fmax: float = 2048.0,
    verbose: bool = False,
    psd_t: str = "aLIGO",
):
    """Compute the simple match (optimisation over time and phase) between
    two waveforms. The waveforms are _assumed to be in geometric units_.
    In time-domain the waveforms are assumed to be already conditioned.
    The spacing/length in time and frequency is assumed to be already consistent.
    For example:
    - if both waveforms are in TD, time spacing is the same
    - if both waveforms are in FD, frequency spacing *and length* are the same
    - if one is in TD and one in FD, it is assumed that calling an FFT on the TD
    waveform will give the right frequency spacing and length.

    The mismatch integral will be done as follows:
    - between f_low and fmax if f_low>fmin
    - between fmin and fmax if f_low<fmin

    Args:
        h1 (Union[TimeSeries, FrequencySeries]): The first waveform in geometric units
        h2 (Union[TimeSeries, FrequencySeries]): The second waveform in geometric units
        f_low (float): The minimum of the strarting frequencies of the 2 waveforms, in geometric units
        Ms (np.ndarray): Masses to use, in solar masses
        fmin (float, optional): Minimum limit of mismatch integral if possible [Hz]. Defaults to 10.0.
        fmax (float, optional): Maximum limit of mismatch integral [Hz]. Defaults to 2048.0.
        verbose (bool, optional): Print extra info. Defaults to False.
        psd_t (str, optional): The type of PSD to use. Defaults to "aLIGO".

    Returns:
        np.ndarray: The array of mismatches
    """

    # Distance scale: feducial but chosen so that the amplitude of the modes
    # is not too far in order of magnitude from the PSD
    dist = 1.0e7 * lal.PC_SI / lal.C_SI
    matches = np.zeros(len(Ms))
    scale_fac = 1.0
    # print(f"f_low={f_low}")
    for i, M in enumerate(Ms):
        # Mass in seconds
        Mt = M * lal.MTSUN_SI

        # physical unit
        f_high_phys = fmax
        f_low_phys = f_low / Mt
        if f_low_phys < fmin:
            f_low_phys = fmin
        if verbose:
            print("\n\n f_low = ", f_low_phys, " Hz, f_high = ", f_high_phys, " Hz")

        if i == 0:
            # First time we are computing things
            # Rescale signal and template to physical units as appropriate for time
            # or frequency domain waveforms
            dist_scale = M * lal.MRSUN_SI * M * lal.MTSUN_SI / dist
            if isinstance(h1, TimeSeries):

                dtM = h1.delta_t * Mt
                x1 = TimeSeries(h1 * Mt / dist, delta_t=dtM)
                h1tilde = make_frequency_series(x1)
            else:
                h1tilde = FrequencySeries(
                    h1.numpy() * dist_scale, delta_f=h1.delta_f / (M * lal.MTSUN_SI)
                )

            if isinstance(h2, TimeSeries):
                dtM = h2.delta_t * Mt
                x2 = TimeSeries(h2 * Mt / dist, delta_t=dtM)
                h2tilde = make_frequency_series(x2)
            else:
                h2tilde = FrequencySeries(
                    h2.numpy() * dist_scale, delta_f=h2.delta_f / (M * lal.MTSUN_SI)
                )

        else:
            # We now have signal and template in frequency domain. Just rescale
            # correctly with the new total mass. Note that the amplitude scaling
            # is not done since it's irrelevant to the mismatch
            scale_fac = 1.0 * M / Ms[i - 1]
            h1old = h1tilde
            h2old = h2tilde
            h1tilde = FrequencySeries(h1old.data, delta_f=h1old.delta_f / scale_fac)
            h2tilde = FrequencySeries(h2old.data, delta_f=h2old.delta_f / scale_fac)

        psd = generate_psd(len(h2tilde), h2tilde.delta_f, f_low_phys, psd_type=psd_t)

        matches[i] = match(
            h2tilde,
            h1tilde,
            psd=psd,
            low_frequency_cutoff=f_low_phys,
            high_frequency_cutoff=f_high_phys,
        )[0]

    return matches


def generate_dominant_mode_pol_LAL(params: waveform_params, f_max: float = 2048):
    # Generate the dominant mode
    hp, hc = generate_waveform(params, f_max=f_max)
    M = params.m1 + params.m2
    # For convenience we now make everything be in geometric units
    if params.domain == "TD":
        # Rescale the amplitude and time array
        amp_prefac = M * lal.MRSUN_SI / params.distance * np.abs(Ylm(2, 2, 0, 0))
        time_prefac = 1 / (M * lal.MTSUN_SI)
        return TimeSeries(hp / amp_prefac, delta_t=hp.delta_t * time_prefac)
        #
    elif params.domain == "FD":
        # Rescale the amplitude and frequency array
        # Amplitude rescaling prefactor. Note the extra factors that come from FFT!
        amp_prefac = (
            M
            * lal.MRSUN_SI
            * M
            * lal.MTSUN_SI
            / params.distance
            * np.abs(Ylm(2, 2, 0, 0))
        )
        freq_prefac = M * lal.MTSUN_SI
        return FrequencySeries(hp / amp_prefac, delta_f=hp.delta_f * freq_prefac)
