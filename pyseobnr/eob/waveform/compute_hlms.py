"""
Contains functions associated with waveform construction, mostly for merger-ringdown.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Final

import numpy as np
from lalinference.imrtgr import nrutils
from pygsl_lite import spline
from scipy.interpolate import InterpolatedUnivariateSpline

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
    EOBNonQCCorrection,
    compute_QNM,
)
from ..fits.IV_fits import InputValueFits
from ..fits.MR_fits import MergerRingdownFits
from .compute_MR import compute_MR_mode_free
from .waveform import compute_factors, unrotate_leading_pn

_default_deviation_dict: Final[dict[str, float]] = {
    f"{ell},{emm}": 0.0 for ell, emm in VALID_MODES
}


def concatenate_modes(hlms_1: Dict[Any, Any], hlms_2: Dict[Any, Any]) -> Dict[Any, Any]:
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
) -> dict[str, Any]:
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
    modes_intrp = {}

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
        modes_intrp[key] = 1 * tmp_store
    return modes_intrp


def compute_IMR_modes(
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
    mixed_modes: list[tuple[int, int]] | None = None,
    final_state=None,
    qnm_rotation=0.0,
    dw_dict: dict[str, float] | None = None,
    domega_dict: dict[str, float] | None = None,
    dtau_dict: dict[str, float] | None = None,
):
    """This computes the IMR modes given the inspiral modes and the
    attachment time.

    Args:
        t (np.ndarray): The interpolated time array of the inspiral modes
        hlms (np.ndarray): Dictionary containing the inspiral modes
        t_for_compute (np.ndarray): The fine dynamics time array
        hlms_for_compute (np.ndarray): The waveform modes on the fine dynamics
        m1 (float): Mass of primary
        m2 (float): Mass of secondary
        chi1 (float): z-component of the primary dimensionless spin
        chi2 (float): z-component of the secondary dimensionless spin
        t_attach (float): Attachment time
        f_nyquist (float): Nyquist frequency, needed for checking that RD frequency is resolved
        lmax_nyquist (int): Determines for which modes the nyquist test is applied for
        mixed_modes (List): List of mixed modes to consider. Defaults to [(3,2),(4,3)]
        final_state (List): Final mass and spin of the remnant. Default to None. If None,
                            compute internally.
        qnm_rotation (float): Factor rotating the QNM mode frequency in the co-precessing frame
                            (Eq. 33 of Hamilton et al.)
        dw_dict (dict): Dictionary of fractional deviation at instantaneous frequency at the mode
                        peak amplitude
        domega_dict (dict): Dictionary of fractional deviations of QNM frequency for each mode
        dtau_dict (dict): Dictionary of fractional deviation of QNM damping time for each mode

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
    dt = np.diff(t)[0]
    # N = int(10 / dt) + 1

    # Figure out the duration of the ringdown. Taken to be 30
    # times the damping time of the (2,2) mode
    # To compute QNM, get the final mass and spin

    if final_state:
        final_mass, final_spin = final_state
    else:
        final_mass = nrutils.bbh_final_mass_non_precessing_UIB2016(m1, m2, chi1, chi2)
        final_spin = nrutils.bbh_final_spin_non_precessing_HBR2016(
            m1, m2, chi1, chi2, version="M3J4"
        )

    omega_complex = compute_QNM(2, 2, 0, final_spin, final_mass).conjugate()

    # Here we are only interested in the (2,2) mode damping time to estimate
    # the ringdown length. We don't need to compute the co-precessing frame QNM
    # frequencies from the J-frame QNMs as in `compute_MR.py` since this rotation
    # only affects the real part of the frequency and not the damping time.
    damping_time = 1 / np.imag(omega_complex) * (1 + dtau_dict["2,2"])
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
    MR_fits = MergerRingdownFits(m1, m2, [0.0, 0.0, chi1], [0.0, 0.0, chi2])
    fits_dict = dict(
        c1f=MR_fits.c1f(), c2f=MR_fits.c2f(), d1f=MR_fits.d1f(), d2f=MR_fits.d2f()
    )

    # Placeholder for the IMR modes. Note that by construction
    # this is longer than is needed for the (5,5) mode, since idx_55<idx
    h = np.zeros(idx + 1 + int(ringdown_time // dt) + 10, dtype=np.complex128)
    N_interp = 5
    for ell_m, mode in hlms_for_compute.items():
        if ell_m == (5, 5):
            t_a = t_attach - 10
            idx_end = idx_55
            t_ring = t_ringdown_55
        else:
            t_a = 1 * t_attach
            idx_end = idx
            t_ring = t_ringdown
        ell, m = ell_m

        amp = np.abs(mode)
        phase = np.unwrap(np.angle(mode))

        idx_interp = np.argmin(np.abs(t_for_compute - t_a))
        left = np.max((0, idx_interp - N_interp))
        right = np.min((idx_interp + N_interp, len(t_for_compute)))

        intrp_amp = InterpolatedUnivariateSpline(
            t_for_compute[left:right], amp[left:right]
        )
        intrp_phase = InterpolatedUnivariateSpline(
            t_for_compute[left:right], phase[left:right]
        )
        amp_max = intrp_amp(t_a)
        damp_max = intrp_amp.derivative()(t_a)
        phi_match = intrp_phase(t_a)
        omega_max = intrp_phase.derivative()(t_a)

        # To improve the stability of the merger-ringdown for odd-m configurations
        # with a minimum in the amplitude close to the attachment point,
        # we directly use the Input Value fits for the frequency,
        # instead of reading its value from the inspiral phase.
        # If the NQCs were *not* applied this would lead to a
        # discontinuity, and one would need to go back to the
        # previous prescription.

        if m % 2 == 1:
            IVfits = InputValueFits(m1, m2, [0.0, 0.0, chi1], [0.0, 0.0, chi2])
            omega_max = IVfits.omega()[ell, m] * (1.0 + dw_dict[f"{ell},{m}"])

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
        hIMR[(ell, m)] = 1 * h
        hIMR[(ell, m)][: idx_end + 1] = hlms[(ell, m)][: idx_end + 1]
        hIMR[(ell, m)][idx_end + 1 : idx_end + 1 + len(hring)] = hring[:]

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
        hIMR[(ell, m)] = 1 * h
        hIMR[(ell, m)][: idx_end + 1] = hlms[(ell, m)][: idx_end + 1]
        hIMR[(ell, m)][idx_end + 1 : idx_end + 1 + len(hring)] = hring[:]

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
    fits_dict,
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
        fits_dict (dict): dictionary of fit coefficients in the ringdown anzatz
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
        m,
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
    polar_dynamics: np.ndarray,
    t_peak: float,
    nrDeltaT: float,
    m_1: float,
    m_2: float,
    chi_1: float,
    chi_2: float,
    dA_dict: dict[str, float],
    dw_dict: dict[str, float],
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

    # Compute omega

    r = polar_dynamics[0]
    pr = polar_dynamics[1]
    omega_orb = polar_dynamics[2]
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


def apply_nqc_corrections(
    hlms: Dict[Any, Any], nqc_coeffs: Dict[Any, Any], polar_dynamics: np.ndarray
):
    """
    Loop over modes and multiply them by NQC correction

    Args:
        hlms (Dict): Dictionary of inspiral modes
        nqc_coeffs (Dict): Dictionary of NQC coefficients
        polar_dynamics (np.ndarray): Dynamics array

    """
    r, pr, omega_orb = polar_dynamics
    for key in hlms.keys():
        ell, m = key
        try:
            NQC_coeffs = nqc_coeffs[(ell, m)]
        except KeyError:
            continue
        correction = EOBNonQCCorrection(r, None, pr, None, omega_orb, NQC_coeffs)
        hlms[key] *= correction
