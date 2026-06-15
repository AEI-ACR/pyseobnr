"""
Contains functions associated with waveform construction, mostly for merger-ringdown.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Final

import numpy as np
from pygsl_lite import spline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import argrelmax

from ...auxiliary.mode_mixing.auxiliary_functions_modemixing import (
    h_ellm0_nu,
    hdot_ellm0_nu,
    mu,
    omega_ellm0,
    phi_ellm0,
)
from ...models.common import VALID_MODES
from ..fits.EOB_fits import (
    EOBCalculateNQCCoefficients_freeattach,
    EOBCalculateNQCCoefficients_freeattach_BNS,
    EOBNonQCCorrectionImpl,
    compute_QNM,
)
from ..fits.IV_fits import BNSInputValueFits, InputValueFits
from ..fits.MR_fits import MergerRingdownFits
from ..utils.nr_utils import (
    bbh_final_mass_non_precessing_UIB2016,
    bbh_final_spin_non_precessing_HBR2016,
)
from ..utils.universal_relations import SmoothTransitionFunction
from .compute_MR import MRAnzatze, compute_MR_mode_free
from .waveform import compute_factors, unrotate_leading_pn

_default_deviation_dict: Final[dict[str, float]] = {
    f"{ell},{emm}": 0.0 for ell, emm in VALID_MODES
}


def concatenate_modes(hlms_1: dict[Any, Any], hlms_2: dict[Any, Any]) -> dict[Any, Any]:
    """Concatenate 2 dictionaries of waveform modes

    This is used to put together the low and fine sampling waveform modes.

    Note:
        Assumes that the 2 dictionaries have the same keys.

    Args:
        hlms_1 (Dict[Any,Any]): First dictionary of waveform modes to concatenate
        hlms_2 (Dict[Any,Any]): Second dictionary of waveform modes to concatenate

    Returns:
        Dict[Any,Any]: Concatenated modes
    """
    hlms = {}
    for key in hlms_1.keys():
        hlms[key] = np.concatenate((hlms_1[key], hlms_2[key]))
    return hlms


def interpolate_modes_fast(
    t_old: np.ndarray,
    t_new: np.ndarray,
    modes_dict: dict[tuple[int, int], Any],
    phi_orb: np.ndarray,
    m_max: int = 5,
) -> dict[tuple[int, int], Any]:
    """Construct inertial frame modes on a new regularly
    spaced time grid.

    Does this by employing a carrier
    signal, see the idea in [Cotesta2020]_ .

    Uses a custom version of CubicSpline that is faster, but
    cannot handle derivatives or integrals.

    Args:
        t_old (np.ndarray): Original unequally spaced time array
        t_new (np.ndarray): New equally spaced time array
        modes_dict (dict): Dictionary containing *complex* modes
        phi_orb (np.ndarray): Orbital phase
        m_max (int): Max m appearing in the modes

    Returns:
        dict: Dictionary of modes interpolated onto t_new
    """
    modes_intrp: dict[tuple[int, int], Any] = {}

    n = len(t_old)
    intrp_orb = spline.cspline(n)
    intrp_orb.init(t_old, phi_orb)
    phi_orb_interp = intrp_orb.eval_e_vector(t_new)
    tmp_store = np.zeros(len(phi_orb_interp), dtype=np.complex128)
    intrp_re = spline.cspline(n)
    intrp_im = spline.cspline(n)

    factors = np.zeros((m_max, len(phi_orb_interp)), dtype=np.complex128)
    compute_factors(phi_orb_interp, m_max, factors)
    for key, item in modes_dict.items():
        m = key[1]

        tmp = item * np.exp(1j * m * phi_orb)

        intrp_re.init(t_old, tmp.real)
        intrp_im.init(t_old, tmp.imag)
        result_re = intrp_re.eval_e_vector(t_new)
        result_im = intrp_im.eval_e_vector(t_new)
        unrotate_leading_pn(result_re, result_im, factors[m - 1], tmp_store)
        modes_intrp[key] = np.copy(tmp_store)
    return modes_intrp


def compute_IMR_modes(
    t: np.ndarray,
    hlms: dict[tuple[int, int], np.ndarray],
    t_for_compute: np.ndarray,
    hlms_for_compute: dict[tuple[int, int], np.ndarray],
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
    t_attach: float,
    f_nyquist: float,
    lmax_nyquist: int,
    mixed_modes: list[tuple[int, int]] | None = None,
    final_state: list | tuple[float, float] | None = None,
    qnm_rotation: float = 0.0,
    align: bool = True,
    dw_dict: dict[str, float] | None = None,
    domega_dict: dict[str, float] | None = None,
    dtau_dict: dict[str, float] | None = None,
    dtau_22_asym: float = 0.0,
    ivs_mrd: MRAnzatze | None = None,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """Computes the IMR modes given the inspiral modes and the
    attachment time.

    Args:
        t: The interpolated time array of the inspiral modes
        hlms: Dictionary containing the inspiral modes
        t_for_compute: The fine dynamics time array
        hlms_for_compute: The waveform modes on the fine dynamics
        m1: Mass of primary
        m2: Mass of secondary
        chi1: z-component of the primary dimensionless spin
        chi2: z-component of the secondary dimensionless spin
        t_attach: Attachment time
        f_nyquist: Nyquist frequency, needed for checking that RD frequency is resolved
        lmax_nyquist: Determines for which modes the nyquist test is applied for
        mixed_modes: List of mixed modes to consider. Defaults to ``[(3,2),(4,3)]``
        final_state: Final mass and spin of the remnant (as tuple or list of 2 elements).
            Default to ``None``. If ``None``, compute internally.
        qnm_rotation: Factor rotating the QNM mode frequency in the co-precessing frame
            (Eq. 33 of Hamilton et al.)
        align: If True, align the waveform at the peak of the (2,2) mode.
        dw_dict: Dictionary of fractional deviation at instantaneous frequency at the mode
            peak amplitude
        domega_dict: Dictionary of fractional deviations of QNM frequency for each mode
        dtau_dict: Dictionary of fractional deviation of QNM damping time for each mode
        dtau_22_asym: Damping time deviation for the antisymmetric modes, only used to
            get the same ringdown time array as the symmetric mode
        ivs_mrd: fits for the MR ansatze

    Returns:
        time array and dictionary containing the waveform modes

    Note:
        The deviations to the QNM damping time are not acting on the anti-symmetric code:
        the current implementation just ensures that the anti-symmetric modes have the
        same length as to the symmetric ones.
    """

    # We want to attach the ringdown always at the same time,
    # regardless of the sampling rate, i.e. all the functions
    # are evaluated at the true attachment time, not just the
    # closest grid point, as was done in v4.
    # This requires one to be somewhat careful in the construction

    # Dictionary that will hold the final modes
    hIMR = {}

    if mixed_modes is None:
        mixed_modes = [(3, 2), (4, 3)]

    if dw_dict is None:
        dw_dict = {} | _default_deviation_dict

    if domega_dict is None:
        domega_dict = {} | _default_deviation_dict

    if dtau_dict is None:
        dtau_dict = {} | _default_deviation_dict

    # First find the closest point on the time grid which is
    # *before* the attachment time. We do this twice,
    # because for the (5,5) mode the attachment time is
    # different from other modes

    # All modes except (5,5)
    idx = np.argmin(np.abs(t - t_attach))
    if t[idx] > t_attach:
        idx -= 1

    # Time at the grid-point just before the attachment point
    t_match = t[idx]

    # (5,5) mode
    idx_55 = np.argmin(np.abs(t - (t_attach - 10)))
    if t[idx_55] > t_attach - 10:
        idx_55 -= 1
    t_match_55 = t[idx_55]

    # The time spacing. This assumes that we have already
    # interpolated the modes to equal spacing
    dt = t[1] - t[0]
    # N = int(10 / dt) + 1

    # Figure out the duration of the ringdown. Taken to be 30
    # times the damping time of the (2,2) mode
    # To compute QNM, get the final mass and spin

    if final_state:
        final_mass, final_spin = final_state
    else:
        final_mass = bbh_final_mass_non_precessing_UIB2016(m1, m2, chi1, chi2)
        final_spin = bbh_final_spin_non_precessing_HBR2016(
            m1, m2, chi1, chi2, version="M3J4"
        )

    omega_complex = compute_QNM(2, 2, 0, final_spin, final_mass).conjugate()

    # Here we are only interested in the (2,2) mode damping time to estimate
    # the ringdown length. We don't need to compute the co-precessing frame QNM
    # frequencies from the J-frame QNMs as in `compute_MR.py` since this rotation
    # only affects the real part of the frequency and not the damping time.

    # For the antisymmetric modes we use dtau_22_asym only to get the same
    # ringdown time array as the symmetric mode
    damping_time = (
        1 / np.imag(omega_complex) * (1 + dtau_dict["2,2"]) * (1 + dtau_22_asym)
    )

    # The length of the ringdown rounded to closest M
    ringdown_time = int(30 * damping_time)

    # Construct the array on which the ringdown signal is computed
    # Note: since we are attaching at the *actual* attachment point
    # which will fall *between* grid points we need to add an offset
    # to the ringdown time-series so that ansatze in the ringdown is
    # correctly evaluated.

    t_ringdown = np.arange(0, ringdown_time, dt) + (t_match + dt - t_attach)
    t_ringdown_55 = np.arange(0, ringdown_time, dt) + (
        t_match_55 + dt - (t_attach - 10)
    )

    # Get the fits for the MR ansatze
    if ivs_mrd is not None:
        fits_dict = deepcopy(ivs_mrd)
    else:
        MR_fits = MergerRingdownFits(m1, m2, [0.0, 0.0, chi1], [0.0, 0.0, chi2])
        fits_dict = MRAnzatze(
            c1f=MR_fits.c1f(), c2f=MR_fits.c2f(), d1f=MR_fits.d1f(), d2f=MR_fits.d2f()
        )

    # see below for odd m's
    IVfits_omegas = InputValueFits(m1, m2, [0.0, 0.0, chi1], [0.0, 0.0, chi2]).omega()

    # we compute this quantity only once
    idx_interp_22 = np.argmin(np.abs(t_for_compute - t_attach))

    # Placeholder for the IMR modes. Note that by construction
    # this is longer than is needed for the (5,5) mode, since idx_55<idx
    n_samples = idx + 1 + int(ringdown_time // dt) + 10
    N_interp = 5
    for ell_m, mode in hlms_for_compute.items():
        if ell_m == (5, 5):
            t_a = t_attach - 10
            idx_end = idx_55
            t_ring = t_ringdown_55
            idx_interp = np.argmin(np.abs(t_for_compute - t_a))
        else:
            t_a = 1 * t_attach
            idx_end = idx
            t_ring = t_ringdown
            idx_interp = idx_interp_22
        ell, m = ell_m

        amp = np.abs(mode)
        phase = np.unwrap(np.angle(mode))

        # idx_interp = np.argmin(np.abs(t_for_compute - t_a))
        left = max((0, idx_interp - N_interp))
        right = min((idx_interp + N_interp, len(t_for_compute)))

        intrp_amp = InterpolatedUnivariateSpline(
            t_for_compute[left:right], amp[left:right]
        )
        intrp_phase = InterpolatedUnivariateSpline(
            t_for_compute[left:right], phase[left:right]
        )
        amp_max = intrp_amp(t_a)
        damp_max = intrp_amp.derivative()(t_a)
        phi_match = intrp_phase(t_a)

        # To improve the stability of the merger-ringdown for odd-m configurations
        # with a minimum in the amplitude close to the attachment point,
        # we directly use the Input Value fits for the frequency,
        # instead of reading its value from the inspiral phase.
        # If the NQCs were *not* applied this would lead to a
        # discontinuity, and one would need to go back to the
        # previous prescription.

        if m % 2 == 1:
            omega_max = IVfits_omegas[ell, m] * (1.0 + dw_dict[f"{ell},{m}"])
        else:
            omega_max = intrp_phase.derivative()(t_a)

        attach_params = dict(
            amp=amp_max,
            damp=damp_max,
            omega=omega_max,
            final_mass=final_mass,
            final_spin=final_spin,
        )

        hring = compute_MR_mode_free(
            t_ring,
            m1,
            m2,
            chi1,
            chi2,
            attach_params,
            ell,
            m,
            fits_dict,
            f_nyquist,
            lmax_nyquist,
            t_match=0,
            phi_match=phi_match,
            qnm_rotation=qnm_rotation,
            domega=domega_dict[f"{ell},{m}"],
            dtau=dtau_dict[f"{ell},{m}"],
        )

        # Construct the full IMR waveform
        hIMR[(ell, m)] = np.concatenate(
            (
                hlms[(ell, m)][: idx_end + 1],
                hring,
                np.zeros(n_samples - (idx_end + len(hring)), dtype=np.complex128),
            )
        )

    idx_end = idx

    # Now handle mixed modes
    for ell_m in mixed_modes:
        ell, m = ell_m
        hring = compute_mixed_mode(
            m1,
            m2,
            chi1,
            chi2,
            ell,
            m,
            t_for_compute,
            hlms_for_compute,
            final_mass,
            final_spin,
            t_attach,
            t_ringdown,
            fits_dict,
            f_nyquist,
            lmax_nyquist,
            qnm_rotation=qnm_rotation,
            dw_dict=dw_dict,
            domega_dict=domega_dict,
            dtau_dict=dtau_dict,
        )
        # Construct the full IMR waveform
        hIMR[(ell, m)] = np.concatenate(
            (
                hlms[(ell, m)][: idx_end + 1],
                hring,
                np.zeros(n_samples - (idx_end + len(hring)), dtype=np.complex128),
            )
        )

    t_IMR = np.arange(len(hIMR[(2, 2)])) * dt

    if align:
        peak = np.argmax(np.abs(hIMR[(2, 2)]))
        t_IMR -= t_IMR[peak]

    return t_IMR, hIMR


def compute_tidal_tapering_v4T(
    t,
    hlms,
    t_for_compute,
    hlms_for_compute,
    m1,
    m2,
    chi1,
    chi2,
    t_attach,
    f_nyquist,
    lmax_nyquist,
    mixed_modes=[(3, 2), (4, 3)],
    final_state=None,
    qnm_rotation=0.0,
):
    """This computes the IMR modes given the inspiral modes and the
    attachment time.

    Args:
        t (np.ndarray): The interpolated time array of the inspiral modes
        hlms (np.ndarray): Dictionary containing the inspiral modes
        t_for_compute (np.ndarray): The fine dynamics time array
        hlms_for_compute (np.ndarray): The waveform modes on the fine dynamics
        m_1 (float): Mass of primary
        m_2 (float): Mass of secondary
        chi_1 (float): z-component of the primary dimensionless spin
        chi_2 (float): z-component of the secondary dimensionless spin
        t_attach (float): Attachment time
        f_nyquist (float): Nyquist frequency, needed for checking that RD frequency is resolved
        lmax_nyquist (int): Determines for which modes the nyquist test is applied for
        mixed_modes (List): List of mixed modes to consider. Defaults to [(3,2),(4,3)]
        final_state (List): Final mass and spin of the remnant. Default to None. If None,
                            compute internally.
        qnm_rotation (float): Factor rotating the QNM mode frequency in the co-precessing frame
            (Eq. 33 of Hamilton et al.)

    Returns:
        dict: Dictionary containing the waveform modes
    """

    # We want to attach the ringdown always at the same time,
    # regardless of the sampling rate, i.e. all the functions
    # are evaluated at the true attachment time, not just the
    # closest grid point, as was done in v4.
    # This requires one to be somewhat careful in the construction

    # For the tapering we have three distinct regions. We define t_a the
    # attachment time and tau the damping time of our tapering
    # [t[0], t_taper = t_a - 15*tau): Here we just use the interpolated
    # inspiral modes
    # [t_taper = t_a - 15*tau, t_freq = t_a - 12M):
    # Here we have to also take the windowing function on the amplitude into
    # account but use the phase of the inspiral waveform
    # [t_freq = t_a - 12M, t_a):
    # Here we are still windowing the amplitude of the inspiral mode, and use
    # the prescription for the frequency evolution
    # [t_a, t[-1]]:
    # Here we extend the amplitude linearly and window it, and use our
    # prescription for the phase evolution

    # Dictionary that will hold the final modes, which are constructed from the
    # *interpolated* inspiral modes hlms
    hIMR = {}

    ell, m = 2, 2

    # The time spacing. This assumes that we have already
    # interpolated the modes to equal spacing
    dt = np.diff(t)[0]
    # N = int(10 / dt) + 1

    # Take out all of the relevant quantities needed for the
    # construction of the tapering from the *fine* (2,2) mode, as it has
    # higher resolution
    mode22_for_compute = hlms_for_compute[(2, 2)]
    N_interp = 5

    amp_for_compute = np.abs(mode22_for_compute)
    phase_for_compute = np.unwrap(np.angle(mode22_for_compute))

    # Sometimes, the application of NQCs can introduce amplitude or frequency
    # peaks. If this happens, we just cut the waveform to discard them
    idx_amp_peak = np.argmin(-amp_for_compute)
    idx_freq_peak = np.argmin(np.gradient(phase_for_compute))
    idx_peak = np.min([idx_amp_peak, idx_freq_peak])

    # First find the closest point on the time grid which is
    # *before* the attachmnent time.
    idx = np.argmin(np.abs(t - t_attach))
    if t[idx] > t_attach:
        idx -= 1
        t_attach = t[idx]

    # Acount for the fact that the peak happens possibly before the grid point
    if t_for_compute[idx_peak] < t[idx] - dt:
        t_attach = t_for_compute[idx_peak]

        idx = np.argmin(np.abs(t - t_attach))
        if t[idx] > t_attach:
            idx -= 1
            t_attach = t[idx]

    # Get the relevant times for the (2,2) mode
    t_a = 1 * t_attach

    idx_interp = np.argmin(np.abs(t_for_compute - t_a))
    idx_interp_freq = np.argmin(np.abs(t_for_compute - t_a + 12))

    left = np.max((0, idx_interp - N_interp))
    right = np.min((idx_interp + N_interp, len(t_for_compute)))

    left_phase = np.max((0, idx_interp_freq - N_interp))

    intrp_amp = InterpolatedUnivariateSpline(
        t_for_compute[left:right], amp_for_compute[left:right]
    )
    intrp_phase = InterpolatedUnivariateSpline(
        t_for_compute[left_phase:right], phase_for_compute[left_phase:right]
    )

    # The necessary quantities for the (2,2) mode tapering
    amp_max = intrp_amp(t_a)
    damp_max = intrp_amp.derivative()(t_a)
    phi_match = intrp_phase(t_a - 12)
    omega_match = -intrp_phase.derivative()(t_a)
    omega_freq = -intrp_phase.derivative()(t_a - 12)

    # Computation of the asymptotically reached delta_omega and damping time
    # tau
    delta_omega = omega_match - omega_freq
    tau = 0.5 * np.pi / np.abs(omega_match)

    # Length of the tapering, which we set to 10 damping times,
    # which roughly corresponds to a 1/(1+exp(10)) ~ 5e-5 amplitude decrease
    damping_time = 15 * tau
    # The length of the ringdown rounded to closest M
    ringdown_time = int(15 * tau)

    idx_freq = np.argmin(np.abs(t - t_a + 12))
    if t[idx_freq] > t_a - 12 and idx_freq > 0:
        idx_freq -= 1
    idx_tapering = np.min([np.argmin(np.abs(t - t_a + 15 * damping_time)), idx_freq])
    if t[idx_tapering] > t_a - 15 * damping_time and idx_tapering > 0:
        idx_tapering -= 1

    # Placeholder for the IMR modes. Note that by construction
    # this is longer than is needed for the (5,5) mode, since idx_55<idx
    h = np.zeros(idx + 1 + int(ringdown_time // dt) + 10, dtype=np.complex128)
    t_IMR = np.arange(len(h)) * dt + t[0]

    mode22 = hlms[(2, 2)][idx_tapering:]
    amp = np.abs(mode22)
    phase = np.unwrap(np.angle(mode22))

    # The arrays used for the computation of the tapered amplitude and frequency
    new_amp = np.zeros(
        idx - idx_tapering + 1 + int(ringdown_time // dt) + 10, dtype=np.float64
    )
    new_phase = np.zeros(
        idx - idx_tapering + 1 + int(ringdown_time // dt) + 10, dtype=np.float64
    )

    # Keep the original phase up till t_a - 12M
    new_phase[: idx_freq - idx_tapering] = phase[: idx_freq - idx_tapering]

    # print(idx,idx_tapering,t_a,t[-1],idx_freq,damping_time,t - t_a + 15*damping_time)
    # Keep the original amplitude times the windowing function up to the end
    new_amp[: idx - idx_tapering] = (
        amp[: idx - idx_tapering]
        * 1.0
        / (1 + np.exp((t_IMR[idx_tapering:idx] - t_a - 15) / tau))
    )

    # Transition the phase smoothly with the frequeny reaching it's asymptotal value
    t_freq = t_IMR[idx_freq:]
    new_phase[idx_freq - idx_tapering :] = (
        phi_match
        - omega_match * (t_freq - t_a + 12)
        - 12 * delta_omega * (np.exp(-(t_freq - t_a + 12) / 12) - 1)
    )

    # Taper the amplitude by linear extension of the amplitude
    t_tapering = t_IMR[idx:]
    new_amp[idx - idx_tapering :] = (
        (amp_max + damp_max * (t_tapering - t_a))
        * 1.0
        / (1 + np.exp((t_tapering - t_a - 15) / tau))
    )

    # Construct the array on which the ringdown signal is computed
    # Note: since we are attaching at the *actual* attachment point
    # which will fall *between* grid points we need to add an offset
    # to the ringdown time-series so that ansatze in the ringdown is
    # correctly evaluated.

    # t_ringdown = np.arange(0, ringdown_time, dt) + (t_match + dt - t_attach)

    # for ell_m, mode in hlms_for_compute.items():
    # t_a = 1 * t_attach
    # idx_end = idx
    # ell, m = ell_m

    # amp = np.abs(mode)
    # phase = np.unwrap(np.angle(mode))

    # idx_interp = np.argmin(np.abs(t_for_compute - t_a))
    # left = np.max((0, idx_interp - N_interp))
    # right = np.min((idx_interp + N_interp, len(t_for_compute)))

    # intrp_amp = InterpolatedUnivariateSpline(
    #     t_for_compute[left:right], amp[left:right]
    # )
    # intrp_phase = InterpolatedUnivariateSpline(
    #     t_for_compute[left:right], phase[left:right]
    # )
    # amp_max = intrp_amp(t_a)
    # damp_max = intrp_amp.derivative()(t_a)
    # phi_match = intrp_phase(t_a)
    # omega_max = intrp_phase.derivative()(t_a)

    # Construct the full IMR waveform
    hIMR[(ell, m)] = 1 * h
    hIMR[(ell, m)][:idx_tapering] = hlms[(ell, m)][:idx_tapering]
    hIMR[(ell, m)][idx_tapering::] = new_amp * (
        np.cos(new_phase) + 1j * np.sin(new_phase)
    )

    # idx_end = idx

    # t_IMR = np.arange(len(hIMR[(2, 2)])) * dt
    peak = np.argmax(np.abs(hIMR[(2, 2)]))
    t_IMR -= t_IMR[peak]
    return t_IMR, hIMR


def compute_tidal_tapering_v5T(
    t,
    hlms,
    t_for_compute,
    hlms_for_compute,
    m1,
    m2,
    chi1,
    chi2,
    kappaT,
    t_attach,
    fits_dict,
    tau_phase_factor,
):
    """This computes the IMR modes given the inspiral modes and the attachment
    time, using the tidal (v5THM) tapering of the merger-ringdown.

    Args:
        t (np.ndarray): The interpolated time array of the inspiral modes
        hlms (dict): Dictionary containing the interpolated inspiral modes
        t_for_compute (np.ndarray): The fine dynamics time array
        hlms_for_compute (dict): The waveform modes on the fine dynamics
        m1 (float): Mass of primary (M=1 units)
        m2 (float): Mass of secondary (M=1 units)
        chi1 (float): z-component of the primary dimensionless spin
        chi2 (float): z-component of the secondary dimensionless spin
        kappaT (float): Dimensionless tidal coupling parameter kappa^T_2, used to
            interpolate the merger model between the BNS and BBH limits
        t_attach (float): Attachment time
        fits_dict (dict): Dictionary of fitted merger quantities (amplitudes,
            frequencies, ...) used to model the post-attachment modes
        tau_phase_factor (float): Factor multiplying the damping time tau used in
            the phase tapering of the post-attachment region

    Returns:
        tuple(np.ndarray, dict): the IMR time array and the dictionary of IMR
        waveform modes
    """

    # We want to attach the "ringdown" always at the same time,
    # regardless of the sampling rate, i.e. all the functions
    # are evaluated at the true attachment time, not just the
    # closest grid point, as was done in v4.
    # This requires one to be somewhat careful in the construction of the
    # merger modeling and tapering

    # For the tapering we have three distinct regions. We define t_a the
    # attachment time and tau the damping time of our tapering
    # [t[0], t_taper = t_a - 15*tau): Here we just use the interpolated
    # inspiral modes. Note that t_match (the last grid point before
    # attachment) is in here.
    # [t_a, t[-1]]:
    # Here we extend the amplitude linearly and window it, and use our
    # prescription for the phase evolution

    # Dictionary that will hold the final modes, which are constructed from the
    # *interpolated* inspiral modes hlms
    hIMR = {}

    ell, m = 2, 2

    # The time spacing. This assumes that we have already
    # interpolated the modes to equal spacing
    dt = np.diff(t)[0]

    # Take out all of the relevant quantities needed for the
    # construction of the tapering from the *fine* (2,2) mode, as it has
    # higher resolution
    mode22_for_compute = hlms_for_compute[(2, 2)]
    N_interp = 5

    # fine wavefom for lowest differencing error
    amp_for_compute = np.abs(mode22_for_compute)
    phase_for_compute = np.unwrap(np.angle(mode22_for_compute))

    # First find the closest point on the time grid which is
    # *before* the attachment time.
    t_attach = min(t_for_compute[-1], t_attach)
    idx = np.argmin(np.abs(t - t_attach))
    if t[idx] > t_attach:
        idx -= 1

    # t_match is the last grid time before t_a
    t_match = t[idx]

    # Get the relevant times for the (2,2) mode
    t_a = t_attach

    idx_interp = np.argmin(np.abs(t_for_compute - t_a))

    # Check for phase peaks:
    idxs_phase_peak = argrelmax(phase_for_compute)[0]
    if len(idxs_phase_peak) > 0:
        idx_phase_peak = idxs_phase_peak[0]
        if idx_phase_peak < idx_interp:
            # print('Phase peak detected! Will correct t_a appropriately.')
            t_attach = t_for_compute[idx_phase_peak]
            idx = np.argmin(np.abs(t - t_attach))
            if t[idx] > t_attach:
                idx -= 1
            # t_match is the last grid time before t_a
            t_match = t[idx]
            t_a = t_attach

    left = np.max((0, idx_interp - N_interp))
    right = np.min((idx_interp + N_interp, len(t_for_compute)))

    intrp_amp = InterpolatedUnivariateSpline(
        t_for_compute[left:right], amp_for_compute[left:right]
    )
    intrp_phase = InterpolatedUnivariateSpline(
        t_for_compute[left:right], phase_for_compute[left:right]
    )

    # The necessary quantities for the (2,2) mode tapering
    # We will also extract the phase at a later point, as it has to be
    # handled with care in comparison to the coarse phase
    amp_max = intrp_amp(t_a)
    damp_max = intrp_amp.derivative()(t_a)
    omega_match = -intrp_phase.derivative()(t_a)
    domega_match = -intrp_phase.derivative(2)(t_a)

    # also recover the IV fits to be reached
    omega_IV = fits_dict["omega"]
    if omega_IV <= omega_match:
        # print(
        #     "Waveform that has higher omega than IVs at matching time. Skipping boost in tapering."
        # )
        omega_IV = omega_match
    omega_asymp = 1.3 * omega_IV

    # We assume the NQCs to enforce the correct IVs
    A_IV = amp_max

    # The timescale of the damping
    tau = 0.5 * np.pi / np.abs(omega_IV)

    # Boosts: We want to give the frequency and amplitude a little boost before
    # windowing, as NQCs can't reliably model NR. The boost function is
    # \beta = 1 + B_omega * exp((t - t_a)/tau_phase) for the frequency, and
    # \beta = 1 + B * exp((t - t_a)/tau_boost) for the amplitude, which we
    # multiply with the respective waveform quantity up to t_a
    # Next we determine some of the necessary quantities

    # For the frequency boost:
    B_omega = omega_IV / omega_match - 1.0
    delta_omega = omega_asymp - omega_IV
    tau_phase = tau_phase_factor * tau

    # For the amplitude boost:
    dlna = damp_max / amp_max
    tau_amp_ref = 1.5 * tau
    tau_boost = tau_amp_ref / (1 + (np.sign(dlna) - 1) * dlna * tau_amp_ref)
    tau_window = tau_amp_ref / (1 + (np.sign(dlna) + 1) * dlna * tau_amp_ref)

    # These are the *boosted* amplitude derivatives at t_a (pre-windowing)
    boosted_amp_max = A_IV * 2
    boosted_damp_max = A_IV / tau_boost + 2 * damp_max

    # Length of the tapering, which we set to 15 damping times,
    # which roughly corresponds to a 1/(1+exp(15)) ~ exp(-15) ~ 3e-7 error
    damping_time = np.max((tau_phase, tau_boost, tau))
    # The length of the ringdown rounded to closest M
    ringdown_time = int(15 * tau)

    # The time where we begin the boosts
    idx_tapering = np.max([np.argmin(np.abs(t - t_a + 15 * damping_time)), 0])
    if t[idx_tapering] > t_a - 15 * damping_time and idx_tapering > 0:
        idx_tapering -= 1

    # Placeholder for the IMR modes
    h = np.zeros(idx + 1 + int(ringdown_time // dt) + 10, dtype=np.complex128)
    # t_IMR is going to be our final time array
    t_IMR = np.arange(idx + 1 + int(ringdown_time // dt) + 10) * dt + t[0]

    # The arrays used for the computation of the tapered amplitude and frequency
    new_amp = np.zeros(
        idx - idx_tapering + 1 + int(ringdown_time // dt) + 10, dtype=np.float64
    )
    new_phase = np.zeros(
        idx - idx_tapering + 1 + int(ringdown_time // dt) + 10, dtype=np.float64
    )

    # Extract the coarse waveform from the actual grid we will use
    mode22 = hlms[(2, 2)][idx_tapering:]
    amp = np.abs(mode22)
    phase = np.unwrap(np.angle(mode22))

    # Helpful redefined time for boosts and windowing
    t_shift = t_IMR - t_a

    # Compute the *unboosted* phase at t_a, which we will need later
    # Note that the fine and coarse waveform can desync due to low resolution
    # So we align in the beginning

    # It can happen that the fine dynamics are between two coarse dynamics points
    if (t_for_compute[-1] - t_for_compute[0]) < dt:

        t_ref = t_for_compute[3] - t_a

        ref_for_compute = np.searchsorted(t_shift[idx_tapering:], t_ref)

        intrp_phase_sync = InterpolatedUnivariateSpline(
            t_shift[
                idx_tapering
                + ref_for_compute
                - 4 : min(idx_tapering + ref_for_compute + 3, idx + 1)
            ],
            phase[ref_for_compute - 4 : min(ref_for_compute + 3, len(phase))],
        )

        phase_shift = -intrp_phase_sync(t_ref) + phase_for_compute[3]
        phi_match = intrp_phase(t_a) + phase_shift
    else:
        ref = np.where(t_shift[idx_tapering : idx + 2] > t_for_compute[0] - t_a)[0][0]

        ref_for_compute = np.where(t_for_compute - t_a > t_shift[idx_tapering + ref])[
            0
        ][0]

        intrp_phase_sync = InterpolatedUnivariateSpline(
            t_for_compute[
                max(ref_for_compute - 3, 0) : min(
                    ref_for_compute + 3, len(t_for_compute - 1)
                )
            ]
            - t_a,
            phase_for_compute[
                max(ref_for_compute - 3, 0) : min(
                    ref_for_compute + 3, len(t_for_compute - 1)
                )
            ],
        )

        phase_shift = -intrp_phase_sync(t_shift[idx_tapering + ref]) + phase[ref]
        phi_match = intrp_phase(t_a) + phase_shift

    if omega_IV > omega_match:
        # This quantity is the residual from the start, to be used in the integration
        dphi = -B_omega * np.exp(t_shift[idx_tapering] / tau_phase) * phase[0]

        # The following interpolation is needed, such that the dependence on dt
        # vanishes. 1000 data points proves good enough and reproduces the input
        # values to the 1% level
        independent_t = np.linspace(t_shift[idx_tapering], 0.0, 1000)
        dt_new = independent_t[1] - independent_t[0]

        # We also need to augment the fine dynamics, as otherwise we will get errors for
        # low sampling frequencies
        size = max((int(np.round(np.size(t_for_compute) / 1e3)), 1))
        if t_for_compute[0] - t_a > t_shift[idx_tapering]:
            ref_idx = np.where(
                t_shift[idx_tapering : idx + 1] < t_for_compute[0] - t_a
            )[0][-1]
            all_times = np.concatenate(
                (
                    t_shift[idx_tapering : idx_tapering + ref_idx + 1],
                    t_for_compute[::size] - t_a,
                )
            )
            all_phase = np.concatenate(
                (phase[: ref_idx + 1], phase_for_compute[::size] + phase_shift)
            )

            # It can happen that the second derivatives don't match, so in this
            # case we do some smoothing
        else:
            all_times = t_for_compute[::size] - t_a
            all_phase = phase_for_compute[::size] + phase_shift

        # equidistant_integrand = InterpolatedUnivariateSpline(
        #   t_shift[idx_tapering : idx + 1],
        #   phase[:idx - idx_tapering + 1]*np.exp(t_shift[idx_tapering : idx + 1]/tau_phase))(independent_t)
        equidistant_integrand_fine = InterpolatedUnivariateSpline(
            all_times, all_phase * np.exp(all_times / tau_phase)
        )
        equidistant_integrand = equidistant_integrand_fine(independent_t)
        equidistant_integral = dt_new * np.cumsum(equidistant_integrand)
        phase_integral = InterpolatedUnivariateSpline(
            independent_t, equidistant_integral
        )

        # Compute the numerically integrated term
        phase_integrated = (
            B_omega / tau_phase * phase_integral(t_shift[idx_tapering : idx + 1])
        )

        # [t_tapering, t_match]:
        # Compute the boosted phase
        new_phase[: idx - idx_tapering + 1] = (
            phase[: idx - idx_tapering + 1]
            * (1.0 + B_omega * np.exp(t_shift[idx_tapering : idx + 1] / tau_phase))
            - phase_integrated
            + dphi
        )
        # if A < 0:
        #     print('Weird waveform that has higher omega than IVs at matching time. Will try to remedy.')
        #     new_phase[:idx - idx_tapering + 1] = phase[:idx - idx_tapering + 1]

        # Compute the boosted phase and frequency *at the time t_a*, which might be
        # after t_match
        # This is necessary to ensure a smooth frequency throughout the tapering and
        # independently on the dt chosen

        new_phase_interp = InterpolatedUnivariateSpline(
            t_IMR[idx - 3 : idx + 1],
            new_phase[idx - idx_tapering - 3 : idx - idx_tapering + 1],
        )

        phi_a = new_phase_interp(t_match)
        omega_a = -new_phase_interp.derivative(1)(t_match)
        domega_match_boosted = -new_phase_interp.derivative(2)(t_match)

        # phi_a = (phi_match * (1. + B_omega) + dphi - B_omega/tau_phase * equidistant_integral[-1] )
        # omega_a = - (B_omega/tau_phase*phi_match - (1+B_omega)*omega_match - B_omega/tau_phase *
        #               equidistant_integrand_fine(0))
        # # omega_a = omega_IV - A/tau_phase*(phi_match - phase_integral.derivative()(0))
        # omega_a = - (   intrp_phase = InterpolatedUnivariateSpline(
        # t_for_compute[left:right], phase_for_compute[left:right]
        # ))
        # print(f"Frequency residual = {(omega_a - omega_IV)/omega_IV*1e2} %")

        delta_omega = omega_asymp - omega_a

        # Frequency tapering
        # domega_match_boosted = (
        #       (1+A)*domega_match + 2*A/tau_phase*omega_match - A/tau_phase**2*phi_a
        #       + A/tau_phase*phase_integral.derivative(2)(0)
        # )
        # domega_match_boosted = - ( B_omega/tau_phase**2 * phi_match
        #                          - 2 * B_omega/tau_phase*omega_match
        #                          + (1+B_omega)*domega_match
        #                          - B_omega/tau_phase*equidistant_integrand_fine.derivative()(0))
        # domega_match_boosted = - (new_phase[idx - idx_tapering] + new_phase[idx - idx_tapering - 2]
        #                          - 2*new_phase[idx - idx_tapering - 1]) / dt
        tau_freq = delta_omega / domega_match_boosted

    else:
        # Just continue the phase if the IVs are too low
        new_phase[: idx - idx_tapering + 1] = phase[: idx - idx_tapering + 1]

        new_phase_interp = InterpolatedUnivariateSpline(
            t_IMR[idx - 3 : idx + 1],
            phase[idx - idx_tapering - 3 : idx - idx_tapering + 1],
        )

        phi_a = new_phase_interp(t_match)
        omega_match = -new_phase_interp.derivative(1)(t_match)
        domega_match = -new_phase_interp.derivative(2)(t_match)
        delta_omega = omega_asymp - omega_match
        tau_freq = delta_omega / domega_match

    # Keep the boosted amplitude times the windowing function up to t_match
    new_amp[: idx - idx_tapering + 1] = (
        amp[: idx - idx_tapering + 1]
        * 1.0
        / (1 + np.exp((t_IMR[idx_tapering : idx + 1] - t_a) / tau_window))
        * (1.0 + np.exp((t_IMR[idx_tapering : idx + 1] - t_a) / tau_boost))
    )

    # [t_match, t_IMR[-1]]:
    # Transition the phase smoothly with the frequeny reaching it's asymptotal value

    t_freq = t_IMR[idx + 1 :] - t_match
    new_phase[idx - idx_tapering + 1 :] = (
        phi_a
        - omega_asymp * (t_freq)
        - tau_freq * delta_omega * (np.exp(-(t_freq) / tau_freq) - 1)
    )

    # Taper the amplitude by linear extension of the amplitude
    t_tapering = t_IMR[idx + 1 :]

    new_amp[idx - idx_tapering + 1 :] = (
        (boosted_amp_max + boosted_damp_max * (t_tapering - t_a))
        * 1.0
        / (1 + np.exp((t_tapering - t_a) / tau_window))
    )

    # Construct the full IMR waveform
    hIMR[(ell, m)] = 1 * h
    hIMR[(ell, m)][:idx_tapering] = hlms[(ell, m)][:idx_tapering]
    hIMR[(ell, m)][idx_tapering::] = new_amp * (
        np.cos(new_phase) + 1j * np.sin(new_phase)
    )

    for ell_m, mode in hlms_for_compute.items():
        if ell_m != (2, 2):
            t_a = 1 * t_attach
            ell, m = ell_m

            amp_for_compute = np.abs(mode)
            phase_for_compute = np.unwrap(np.angle(mode))

            amp = np.abs(hlms[ell_m])
            phase = np.unwrap(np.angle(hlms[ell_m]))

            idx_interp_freq = np.argmin(np.abs(t_for_compute - t_a + 12))

            left = np.max((0, idx_interp - N_interp))
            right = np.min((idx_interp + N_interp, len(t_for_compute)))

            left_phase = np.max((0, idx_interp_freq - N_interp))

            intrp_amp = InterpolatedUnivariateSpline(
                t_for_compute[left:right], amp_for_compute[left:right]
            )
            intrp_phase = InterpolatedUnivariateSpline(
                t_for_compute[left_phase:right], phase_for_compute[left_phase:right]
            )
            amp_max = intrp_amp(t_a)
            damp_max = intrp_amp.derivative()(t_a)
            phi_match = intrp_phase(t_a - 12)
            omega_match = -intrp_phase.derivative()(t_a)
            omega_freq = -intrp_phase.derivative()(t_a - 12)

            # Computation of the asymptotically reached delta_omega and damping time
            # tau
            delta_omega = omega_match - omega_freq

            idx_freq = np.argmin(np.abs(t - t_a + 12))
            if t[idx_freq] > t_a - 12 and idx_freq > 0:
                idx_freq -= 1

            # The arrays used for the computation of the tapered amplitude and frequency
            new_amp = np.zeros(
                idx - idx_tapering + 1 + int(ringdown_time // dt) + 10, dtype=np.float64
            )
            new_phase = np.zeros(
                idx - idx_tapering + 1 + int(ringdown_time // dt) + 10, dtype=np.float64
            )

            # Keep the original phase up till t_a - 12M
            new_phase[: idx_freq - idx_tapering] = phase[idx_tapering:idx_freq]

            # Keep the original amplitude times the windowing function up to the end
            new_amp[: idx - idx_tapering] = (
                amp[idx_tapering:idx]
                * 1.0
                / (1 + np.exp((t_IMR[idx_tapering:idx] - t_a - 15) / tau))
            )

            # Transition the phase smoothly with the frequeny reaching it's asymptotal value
            t_freq = t_IMR[idx_freq:]
            new_phase[idx_freq - idx_tapering :] = (
                phi_match
                - omega_match * (t_freq - t_a + 12)
                - 12 * delta_omega * (np.exp(-(t_freq - t_a + 12) / 12) - 1)
            )

            # Taper the amplitude by linear extension of the amplitude
            t_tapering = t_IMR[idx:]
            new_amp[idx - idx_tapering :] = (
                (amp_max + damp_max * (t_tapering - t_a))
                * 1.0
                / (1 + np.exp((t_tapering - t_a - 15) / tau))
            )

            # Construct the full IMR waveform
            hIMR[(ell, m)] = 1 * h
            hIMR[(ell, m)][:idx_tapering] = hlms[(ell, m)][:idx_tapering]
            hIMR[(ell, m)][idx_tapering::] = new_amp * (
                np.cos(new_phase) + 1j * np.sin(new_phase)
            )

    # peak = np.argmax(np.abs(hIMR[(2, 2)]))

    t_IMR -= t_attach
    # t_IMR -= t_IMR[peak]
    return t_IMR, hIMR


def compute_non_tapered(
    t: np.ndarray,
    hlms: dict[tuple[int, int], np.ndarray],
    t_for_compute: np.ndarray,
    hlms_for_compute: dict[tuple[int, int], np.ndarray],
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
    kappaT,
    t_attach: float,
    f_nyquist: float,
    lmax_nyquist: int,
):
    """This computes the IMR modes given the inspiral modes and the
    attachment time.

    Args:
        t: The interpolated time array of the inspiral modes
        hlms: Dictionary containing the inspiral modes
        t_for_compute: The fine dynamics time array
        hlms_for_compute (np.ndarray): The waveform modes on the fine dynamics
        m1: Mass of primary
        m2: Mass of secondary
        chi1: z-component of the primary dimensionless spin
        chi2: z-component of the secondary dimensionless spin
        t_attach: Attachment time
        f_nyquist: Nyquist frequency, needed for checking that RD frequency is resolved
        lmax_nyquist: Determines for which modes the nyquist test is applied for

    Returns:
        dict: Dictionary containing the waveform modes
    """

    # We want to attach the ringdown always at the same time,
    # regardless of the sampling rate, i.e. all the functions
    # are evaluated at the true attachment time, not just the
    # closest grid point, as was done in v4.
    # This requires one to be somewhat careful in the construction

    # Dictionary that will hold the final modes
    hIMR = {}

    # First find the closest point on the time grid which is
    # *before* the attachmnent time. We do this twice,
    # because for the (5,5) mode the attachment time is
    # different from other modes

    not_last_point = True

    ell = 2
    m = 2

    # First find the closest point on the time grid which is
    # *before* the attachmnent time.
    idx = np.argmin(np.abs(t - t_attach))
    if t[idx] > t_attach:
        idx -= 1

    if t_attach > t[-1]:
        not_last_point = False

    if not_last_point:
        mode22_for_compute = hlms_for_compute[(2, 2)]
        N_interp = 5

        amp_for_compute = np.abs(mode22_for_compute)
        phase_for_compute = np.unwrap(np.angle(mode22_for_compute))

        # Get the relevant times for the (2,2) mode
        t_a = 1 * t_attach

        idx_interp = np.argmin(np.abs(t_for_compute - t_a))

        left = np.max((0, idx_interp - N_interp))
        right = np.min((idx_interp + N_interp, len(t_for_compute)))

        intrp_amp = InterpolatedUnivariateSpline(
            t_for_compute[left:right], amp_for_compute[left:right]
        )
        intrp_phase = InterpolatedUnivariateSpline(
            t_for_compute[left:right], phase_for_compute[left:right]
        )

        # The necessary quantities for the (2,2) mode tapering
        amp_max = intrp_amp(t_a)
        phi_match = intrp_phase(t_a)

        # Time at the grid-point just before the attacment point
        # t_match = t[idx]

        # The time spacing. This assumes that we have already
        # interpolated the modes to equal spacing
        dt = np.diff(t)[0]

        # Placeholder for the IMR modes. Note that by construction
        # this is longer than is needed for the (5,5) mode, since idx_55<idx
        h = np.zeros(idx + 2, dtype=np.complex128)

        # Construct the full IMR waveform
        hIMR[(ell, m)] = 1 * h
        hIMR[(ell, m)][: idx + 1] = hlms[(ell, m)][: idx + 1]
        hIMR[(ell, m)][idx + 1] = amp_max * (np.cos(phi_match) + 1j * np.sin(phi_match))
        t_IMR = np.arange(len(hIMR[(2, 2)])) * dt
        t_IMR[-1] = t_a

    else:
        dt = np.diff(t)[0]

        h = np.zeros(idx + 1, dtype=np.complex128)
        hIMR[(ell, m)] = 1 * h
        hIMR[(ell, m)] = hlms[(ell, m)][: idx + 1]
        t_IMR = np.arange(len(hIMR[(2, 2)])) * dt

    peak = np.argmax(np.abs(hIMR[(2, 2)]))
    t_IMR -= t_IMR[peak]
    return t_IMR, hIMR


def compute_mixed_mode(
    m1,
    m2,
    chi1,
    chi2,
    ell,
    m,
    t,
    modes,
    final_mass,
    final_spin,
    t_match,
    t_ringdown,
    fits_dict: MRAnzatze,
    f_nyquist,
    lmax_nyquist,
    qnm_rotation=0.0,
    dw_dict: dict | None = None,
    domega_dict: dict | None = None,
    dtau_dict: dict | None = None,
):
    """
    Computes the (3,2) and (4,3) modes, including mode-mixing in the ringdown.

    See Sec. II C of the [SEOBNRv5HM-notes]_ , especially Eqs.(71, 72)

    Args:
        m1 (float): mass of the primary
        m2 (float): mass of the secondary
        chi1 (float): dimensionless spin of the primary
        chi2 (float): dimensionless spin of the secondary
        ell (int): ell index of the desired mode
        m (int): m index of the desired mode
        t (np.ndarray): inspiral time array
        modes (dict): dictionary containing the waveform modes
        final_mass (float): mass of the remnant
        final_spin (float): dimensionless spin of the remnant
        t_match (float): inspiral time at which the merger-ringdown waveform is attached
        t_ringdown (np.ndarray): ringdown time array
        fits_dict (MRAnzatze): dictionary of fit coefficients in the ringdown anzatz
        f_nyquist (float): Nyquist frequency, needed for checking that RD frequency is resolved
        lmax_nyquist (int): Determines for which modes the nyquist test is applied for
        qnm_rotation (float): Factor rotating the QNM mode frequency in the co-precessing
                              frame (Eq. 33 of Hamilton et al.)
        dw_dict (dict): Dictionary of fractional deviation at instantaneous frequency at the mode
                        peak amplitude
        domega_dict (dict): Dictionary of fractional deviations of QNM frequency for each mode
        dtau_dict (dict): Dictionary of fractional deviation of QNM damping time for each mode


    Returns:
        np.ndarray: the merger-ringdown waveform for the mixed modes

    """
    if dw_dict is None:
        dw_dict = {} | _default_deviation_dict

    if domega_dict is None:
        domega_dict = {} | _default_deviation_dict

    if dtau_dict is None:
        dtau_dict = {} | _default_deviation_dict

    # Get spheroidal input values
    # These are constructed from spherical input values

    # Spherical modes in the *inspiral*
    mode_lm = modes[ell, m]
    mode_mm = modes[m, m]

    # If the inspiral spherical (ell,m) mode vanishes,
    # we will also set to 0 the merger-ringdown
    # of the (ell,m) spherical mode and return immediately
    if np.max(np.abs(mode_lm)) < 1e-8:
        hring = np.zeros_like(t_ringdown)
        return hring

    # We must ensure continuity, without necessarily assuming NQCs
    # Thus we compute the values at the matching point via
    # interpolation of the inspiral modes.
    # If the NQCs *are* applied then these values would be identical
    # to NQC input values

    # First the (ell,m) mode
    idx_match = np.argmin(np.abs(t - t_match))
    N = 5
    left = np.max((0, idx_match - N))
    right = np.min((idx_match + N, len(t)))
    amp = np.abs(mode_lm)
    phase = np.unwrap(np.angle(mode_lm))
    intrp_amp = InterpolatedUnivariateSpline(t[left:right], amp[left:right])
    intrp_phase = InterpolatedUnivariateSpline(t[left:right], phase[left:right])
    h = intrp_amp(t_match)
    hd = intrp_amp.derivative()(t_match)
    phi_lm = intrp_phase(t_match)
    om = intrp_phase.derivative()(t_match)

    # Now the (m,m) mode
    amp = np.abs(mode_mm)
    phase = np.unwrap(np.angle(mode_mm))
    intrp_amp = InterpolatedUnivariateSpline(t[left:right], amp[left:right])
    intrp_phase = InterpolatedUnivariateSpline(t[left:right], phase[left:right])
    h_mm = intrp_amp(t_match)
    hd_mm = intrp_amp.derivative()(t_match)
    phi_mm = intrp_phase(t_match)
    om_mm = intrp_phase.derivative()(t_match)

    # To improve the stability of the merger-ringdown for configurations
    # with a minimum in the amplitude close to the attachment point,
    # we directly use the Input Value fits for the frequency,
    # instead of reading its value from the inspiral phase.
    # If the NQCs were *not* applied this would lead to a
    # discontinuity, and one would need to go back to the
    # previous prescription.

    if m % 2 == 1:
        IVfits = InputValueFits(m1, m2, [0.0, 0.0, chi1], [0.0, 0.0, chi2])
        key_str_lm = str(ell) + "," + str(m)
        key_str_mm = str(m) + "," + str(m)
        om = IVfits.omega()[ell, m] * (1.0 + dw_dict[key_str_lm])
        om_mm = IVfits.omega()[m, m] * (1.0 + dw_dict[key_str_mm])

    # Spherical mode we need in the ringdown
    attach_params = dict(
        amp=h_mm,
        damp=hd_mm,
        omega=om_mm,
        final_mass=final_mass,
        final_spin=final_spin,
    )

    hmm_spherical_ringdown = compute_MR_mode_free(
        t_ringdown,
        m1,
        m2,
        chi1,
        chi2,
        attach_params,
        m,  # "m" is intentional here, for 32 mode we need to use the 22, etc
        m,
        fits_dict,
        f_nyquist,
        lmax_nyquist,
        t_match=0 * t_match,
        phi_match=phi_mm,
        qnm_rotation=qnm_rotation,
        domega=domega_dict[f"{m},{m}"],
        dtau=dtau_dict[f"{m},{m}"],
    )

    # Approximation to spheroidal
    mixing_coeff_mm = np.conj(mu(m, m, m, final_spin))
    hmm_spheroidal = hmm_spherical_ringdown / mixing_coeff_mm

    # Now compute the spheroidal inputs we need

    # Ampltidue at peak
    h_ellm0 = h_ellm0_nu(ell, m, final_spin, h, h_mm, phi_lm, phi_mm)
    # Phase at peak
    ph_ellm0 = phi_ellm0(ell, m, final_spin, h, h_mm, phi_lm, phi_mm)

    # Time derivative of amplitude at peak
    hd_ellm0 = hdot_ellm0_nu(
        ell,
        m,
        final_spin,
        h,
        h_mm,
        hd,
        hd_mm,
        om,
        om_mm,
        phi_lm,
        phi_mm,
    )
    # Frequency at peak
    om_ellm0 = omega_ellm0(
        ell,
        m,
        final_spin,
        h,
        h_mm,
        hd,
        hd_mm,
        om,
        om_mm,
        phi_lm,
        phi_mm,
    )

    attach_params.update(
        amp=h_ellm0,
        damp=hd_ellm0,
        omega=om_ellm0,
        final_mass=final_mass,
        final_spin=final_spin,
    )
    # Compute the coefficients+ansatze for spheroidal mode
    # Note that the QNM deviations are applied to the spheroidal modes in this case
    hlm_spheroidal_ringdown = compute_MR_mode_free(
        t_ringdown,
        m1,
        m2,
        chi1,
        chi2,
        attach_params,
        ell,
        m,
        fits_dict,
        f_nyquist,
        lmax_nyquist,
        t_match=0 * t_match,
        phi_match=ph_ellm0,
        qnm_rotation=qnm_rotation,
        domega=domega_dict[f"{ell},{m}"],
        dtau=dtau_dict[f"{ell},{m}"],
    )
    # Reconstruct the spherical mode
    hring = hmm_spheroidal * np.conj(
        mu(m, ell, m, final_spin)
    ) + hlm_spheroidal_ringdown * np.conj(mu(m, ell, ell, final_spin))

    return hring


def NQC_correction(
    inspiral_modes: Dict,
    t_modes: np.ndarray,
    polar_dynamics: list[np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray],
    t_peak: float,
    nrDeltaT: float,
    m_1: float,
    m_2: float,
    chi_1: float,
    chi_2: float,
    dA_dict: dict[str, float] | None = None,
    dw_dict: dict[str, float] | None = None,
):
    """Given the inspiral modes and the dynamics this function
    computes the NQC coefficients at t_peak-nrDeltaT

    Args:
        inspiral_modes (Dict): Dictionary of inspiral modes (interpolated)
        t_modes (np.ndarray): Time array for inspiral modes
        polar_dynamics (np.ndarray): Dynamics array from ODE solver (unequally spaced)
        t_peak (float): The time of the peak of the orbital frequency
        nrDeltaT (float): The shift from peak of the orbital frequency to peak of (2,2) mode
        m_1 (float): Mass of the primary
        m_2 (float): Mass of the secondary
        chi_1 (float): z-component of dimensionless spin of the primary
        chi_2 (float): z-component Dimensionless spin of the secondary
        dA_dict (Dict): Dictionary of fractional deviations of the mode peak amplitude for each mode
        dw_dict (dict): Dictionary of fractional deviation at instantaneous frequency at the mode
                        peak amplitude
    """
    if dA_dict is None:
        dA_dict = {} | _default_deviation_dict

    if dw_dict is None:
        dw_dict = {} | _default_deviation_dict

    # Compute omega

    r, pr, omega_orb = polar_dynamics
    input_value_fits = InputValueFits(m_1, m_2, [0.0, 0.0, chi_1], [0.0, 0.0, chi_2])
    fits_dict = dict(
        amp=input_value_fits.habs(),
        damp=input_value_fits.hdot(),
        ddamp=input_value_fits.hdotdot(),
        omega=input_value_fits.omega(),
        domega=input_value_fits.omegadot(),
    )

    for (ell, m), mode in inspiral_modes.items():
        key_str_lm = f"{ell},{m}"
        fits_dict["amp"][(ell, m)] *= 1.0 + dA_dict[key_str_lm]
        fits_dict["omega"][(ell, m)] *= 1.0 + dw_dict[key_str_lm]

    # Loop over every mode
    nqc_coeffs = {}
    for ell_m, mode in inspiral_modes.items():
        amp = np.abs(mode)
        phase = np.unwrap(np.angle(mode))
        ell, m = ell_m
        if ell == 5 and m == 5:
            # (5,5) mode is special
            extra = -10
        else:
            extra = 0
        # NQC_coeffs = EOBCalculateNQCCoefficientsV4(
        #     amp, phase, r, pr, omega_orb, ell, m, t_peak, t_modes, m1, m2, chi1, chi2
        # )
        # For equal mass, non-spinning cases odd m modes vanish, so don't try to compute NQCs
        if (
            m % 2
            and np.abs(m_1 - m_2) < 1e-4
            and np.abs(chi_1) < 1e-4
            and np.abs(chi_2) < 1e-4
        ) or (m % 2 and np.abs(m_1 - m_2) < 1e-4 and np.abs(chi_1 - chi_2) < 1e-4):
            continue

        else:
            # Compute the NQC coeffs
            NQC_coeffs = EOBCalculateNQCCoefficients_freeattach(
                amp,
                phase,
                r,
                pr,
                omega_orb,
                ell,
                m,
                t_peak,
                t_modes,
                m_1,
                m_2,
                chi_1,
                chi_2,
                nrDeltaT - extra,
                fits_dict,
            )

            NQC_coeffs["a3S"] = 0
            NQC_coeffs["a4"] = 0
            NQC_coeffs["a5"] = 0
            NQC_coeffs["b3"] = 0
            NQC_coeffs["b4"] = 0

        nqc_coeffs[(ell, m)] = deepcopy(NQC_coeffs)

    return nqc_coeffs


def BNS_NQC_correction(
    inspiral_modes: Dict,
    t_modes: np.ndarray,
    polar_dynamics: np.ndarray,
    t_peak: float,
    nrDeltaT: float,
    m_1: float,
    m_2: float,
    chi_1: float,
    chi_2: float,
    kappaT: float,
):
    """Given the inspiral modes and the dynamics this function
    computes the NQC coefficients at t_peak-nrDeltaT

    Args:
        inspiral_modes (Dict): Dictionary of inspiral modes (interpolated)
        t_modes (np.ndarray): Time array for inspiral modes
        polar_dynamics (np.ndarray): Dynamics array from ODE solver (unequally spaced)
        t_peak (float): The time of the peak of the orbital frequency
        nrDeltaT (float): The shift from peak of the orbital frequency to peak of (2,2) mode
        m_1 (float): Mass of the primary
        m_2 (float): Mass of the secondary
        chi_1 (float): z-component of dimensionless spin of the primary
        chi_2 (float): z-component Dimensionless spin of the secondary
        kappaT (float): Tidal deformability kappa_2^T = 3*nu*[(m_1/M)^3*Lambda_1 + (1 <-> 2)]
    """

    # Compute omega

    r = polar_dynamics[0]
    pr = polar_dynamics[1]
    omega_orb = polar_dynamics[2]
    input_value_fits = BNSInputValueFits(
        m_1, m_2, [0.0, 0.0, chi_1], [0.0, 0.0, chi_2], kappaT
    )
    fits_dict = dict(
        amp=input_value_fits.habs(),
        damp=input_value_fits.hdot(),
        ddamp=input_value_fits.hdotdot(),
        omega=input_value_fits.omega(),
        domega=input_value_fits.omegadot(),
    )
    if kappaT < 20:
        nu = m_1 * m_2
        # Transition to v5HM IVs
        input_value_fits_BBH = InputValueFits(
            m_1, m_2, [0.0, 0.0, chi_1], [0.0, 0.0, chi_2]
        )
        fits_dict_BBH = dict(
            amp=input_value_fits_BBH.habs()[2, 2] * nu,
            damp=input_value_fits_BBH.hdot()[2, 2] * nu,
            ddamp=input_value_fits_BBH.hdotdot()[2, 2] * nu,
            omega=-input_value_fits_BBH.omega()[2, 2],
            domega=-input_value_fits_BBH.omegadot()[2, 2],
        )

        scale = SmoothTransitionFunction(kappaT, a=0.0, b=20.0, flip=True)
        fits_dict = dict(
            amp=fits_dict_BBH["amp"] * scale + fits_dict["amp"] * (1 - scale),
            damp=fits_dict_BBH["damp"] * scale + fits_dict["damp"] * (1 - scale),
            ddamp=fits_dict_BBH["ddamp"] * scale + fits_dict["ddamp"] * (1 - scale),
            omega=fits_dict_BBH["omega"] * scale + fits_dict["omega"] * (1 - scale),
            domega=fits_dict_BBH["domega"] * scale + fits_dict["domega"] * (1 - scale),
        )

    # We only have a 22 mode approximant
    nqc_coeffs = {}
    mode = inspiral_modes[(2, 2)]

    amp = np.abs(mode)
    phase = np.unwrap(np.angle(mode))

    # NQC_coeffs = EOBCalculateNQCCoefficientsV4(
    #     amp, phase, r, pr, omega_orb, ell, m, t_peak, t_modes, m1, m2, chi1, chi2
    # )
    # For equal mass, non-spinning cases odd m modes vanish, so don't try to compute NQCs

    # Compute the NQC coeffs
    NQC_coeffs = EOBCalculateNQCCoefficients_freeattach_BNS(
        amp,
        phase,
        r,
        pr,
        omega_orb,
        2,
        2,
        t_peak,
        t_modes,
        m_1,
        m_2,
        chi_1,
        chi_2,
        nrDeltaT,
        fits_dict,
    )

    NQC_coeffs["a3S"] = 0
    NQC_coeffs["a4"] = 0
    NQC_coeffs["a5"] = 0
    NQC_coeffs["b3"] = 0
    NQC_coeffs["b4"] = 0

    nqc_coeffs[(2, 2)] = deepcopy(NQC_coeffs)

    return nqc_coeffs, fits_dict


def apply_nqc_corrections(
    hlms: dict[tuple[int, int], Any],
    nqc_coeffs: dict[tuple[int, int], Any],
    polar_dynamics: list[np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray],
):
    """
    Loop over modes and multiply them by NQC correction

    Args:
        hlms: Dictionary of inspiral modes
        nqc_coeffs: Dictionary of NQC coefficients
        polar_dynamics: Dynamics array

    """
    r, pr, omega_orb = polar_dynamics

    nqc_apply = EOBNonQCCorrectionImpl(r=r, phi=None, pr=pr, pphi=None, omega=omega_orb)

    for key in hlms.keys():
        ell, m = key
        try:
            nqc_coeffs_current_mode = nqc_coeffs[(ell, m)]
        except KeyError:
            continue
        # correction = EOBNonQCCorrection(r, None, pr, None, omega_orb, nqc_coeffs_current_mode)
        correction = nqc_apply.get_nqc_multiplier(coeffs=nqc_coeffs_current_mode)
        hlms[key] *= correction
