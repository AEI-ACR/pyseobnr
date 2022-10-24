from copy import deepcopy
from typing import Any, Dict

import numexpr

numexpr.set_num_threads(1)
import numpy as jnp
from lalinference.imrtgr import nrutils
from pygsl import spline
from pyseobnr.auxiliary.mode_mixing.auxiliary_functions_modemixing import *
from scipy.interpolate import CubicSpline
from scipy.interpolate.fitpack2 import InterpolatedUnivariateSpline
from scipy.optimize import root_scalar
from scipy.signal import argrelmax, argrelmin

from ..fits.EOB_fits import *
from ..fits.IV_fits import InputValueFits
from ..fits.MR_fits import MergerRingdownFits
from ..waveform.waveform import *
from .compute_MR import compute_MR_mode_free


def get_omega_peak(dynamics):
    """Find the peak of the orbital frequency.
    If a local max is not found, the last point in
    the dynamics is taken. Otherwise we find the
    critical point

    Args:
        dynamics (np.array): The dynamics, t,r,phi,pr,pphi

    Returns:
        float: The time corresponding to the peak of omega
    """
    t = dynamics[:, 0]
    phi_mine = CubicSpline(t, dynamics[:, 2])
    omega_orb_mine = phi_mine.derivative()(t)
    idx_omega_peak = np.argmax(omega_orb_mine)
    t_omega_peak = t[idx_omega_peak]
    idx_maxs = argrelmax(omega_orb_mine)
    intrp = CubicSpline(t, omega_orb_mine)
    if idx_maxs[-1]:

        intrp = CubicSpline(t, omega_orb_mine)
        g = lambda x: intrp.derivative()(x)
        root = root_scalar(g, bracket=(t[0], t[-1]))

        t_omega_peak = root.root

    return t_omega_peak


def concatenate_modes(hlms_1, hlms_2):
    hlms = {}
    for key in hlms_1.keys():
        hlms[key] = np.concatenate((hlms_1[key], hlms_2[key]))
    return hlms


def interpolate_modes_fast(t_old, t_new, modes_dict, phi_orb, m_max=5):
    """Construct intertial frame modes on a new regularly
    spaced time grid. Does this by employing a carrier
    signal. See the idea in https://arxiv.org/pdf/2003.12079.pdf
    Uses a custom version of CubicSpline that is faster, but
    cannot handle derivatives or integrals.
    Args:
        t_old (np.ndarray): Original unequally spaced time array
        t_new (np.ndarray): New equally spaced time array
        modes_dict (dict): Dictionary containing *complex* modes
        phi_orb (np.ndarray): Orbital phase

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


def iterative_refinement(f, interval, levels=3, dt_initial=0.1):
    left = interval[0]
    right = interval[1]
    for n in range(1, levels + 1):
        dt = dt_initial / (10**n)
        t_fine = np.arange(interval[0], interval[1], dt)
        deriv = np.abs(f(t_fine))

        mins = argrelmin(deriv, order=3)[0]
        if len(mins) > 0:
            result = t_fine[mins[0]]

            interval = max(result - 10 * dt, left), min(result + 10 * dt, right)

        else:

            return (interval[0] + interval[-1]) / 2
    return result


def get_attachment_reference_point(omega_orb, t, guess, step_back=100.0, dt=0.1):
    intrp = CubicSpline(t, omega_orb)
    omega_dot = intrp.derivative()

    # Bracketing interval for minimum
    left = guess - 0.95 * step_back
    right = t[-1]

    # If we are using only fine dynamics make sure
    # we don't guess to deep into inspiral
    if left < t[0]:
        left = t[0]

    t_fine = np.arange(left, right, dt)
    deriv = np.abs(omega_dot(t_fine))

    mins = argrelmin(deriv, order=2)[0]
    if len(mins) > 0:
        result = t_fine[mins[0]]
    else:
        return t[-1]

    result = iterative_refinement(
        omega_dot, [max(result - 10 * dt, left), min(result + 10 * dt, right)]
    )
    return result


def get_attachment_reference_point_pr(pr, t, guess, step_back=100.0):
    intrp = CubicSpline(t, pr)

    # Bracketing interval for minimum
    # print(f"guess={guess},step_back={step_back}")
    left = guess - 0.95 * step_back
    right = t[-1]

    # If we are using only fine dynamics make sure
    # we don't guess to deep into inspiral
    if left < t[0]:
        left = t[0]

    t_fine = np.arange(left, right, 0.01)
    pr_fine = intrp(t_fine)

    mins = argrelmin(pr_fine, order=2)[0]
    if len(mins) > 0:
        result = t_fine[mins[-1]]
        return result
    else:
        return t[-1]


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
    mixed_modes=[(3, 2), (4, 3)],
    final_state=None,
):
    """This computes the IMR modes given the inspiral modes and the
    attachment time.

    Args:
        t (np.ndarray): The time array of the inspiral modes
        hlms (np.ndarray): Dictionary containing the inspiral modes
        chi_1 (float): z-component of the primary dimensionless spin
        chi_2 (float): z-component of the secondary dimensionless spin
        m_1 (float): Mass of primary
        m_2 (float): Mass of secondary

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

    # All modes except (5,5)
    idx = jnp.argmin(jnp.abs(t - t_attach))
    if t[idx] > t_attach:
        idx -= 1

    # Time at the grid-point just before the attacment point
    t_match = t[idx]

    # (5,5) mode
    idx_55 = jnp.argmin(jnp.abs(t - (t_attach - 10)))
    if t[idx_55] > t_attach - 10:
        idx_55 -= 1
    t_match_55 = t[idx_55]

    # The time spacing. This assumes that we have already
    # interpolated the modes to equal spacing
    dt = np.diff(t)[0]
    N = int(10 / dt) + 1

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
    damping_time = 1 / np.imag(omega_complex)
    # The length of the ringdown rounded to closest M
    ringdown_time = int(30 * damping_time)

    # Construct the array on which the ringdown signal is computed
    # Note: since we are attaching at the *actual* attachment point
    # which will fall *between* grid points we need to add an offset
    # to the ringdown time-series so that ansatze in the ringdown is
    # correctly evaluated.

    t_ringdown = jnp.arange(0, ringdown_time, dt) + (t_match + dt - t_attach)
    t_ringdown_55 = jnp.arange(0, ringdown_time, dt) + (
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

        amp = jnp.abs(mode)
        phase = jnp.unwrap(jnp.angle(mode))
        # Here

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

        attach_params = dict(
            amp=amp_max,
            damp=damp_max,
            omega=omega_max,
            final_mass=final_mass,
            final_spin=final_spin,
        )

        hring, philm = compute_MR_mode_free(
            t_ring,
            m1,
            m2,
            chi1,
            chi2,
            attach_params,
            ell,
            m,
            fits_dict,
            t_match=0,
            phi_match=phi_match,
            debug=False,
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
            idx,
            fits_dict,
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
    idx,
    fits_dict,
):

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
    amp = jnp.abs(mode_lm)
    phase = jnp.unwrap(jnp.angle(mode_lm))
    intrp_amp = InterpolatedUnivariateSpline(t[left:right], amp[left:right])
    intrp_phase = InterpolatedUnivariateSpline(t[left:right], phase[left:right])
    h = intrp_amp(t_match)
    hd = intrp_amp.derivative()(t_match)
    phi_lm = intrp_phase(t_match)
    om = intrp_phase.derivative()(t_match)

    # Now the (m,m) mode
    amp = jnp.abs(mode_mm)
    phase = jnp.unwrap(jnp.angle(mode_mm))
    intrp_amp = InterpolatedUnivariateSpline(t[left:right], amp[left:right])
    intrp_phase = InterpolatedUnivariateSpline(t[left:right], phase[left:right])
    h_mm = intrp_amp(t_match)
    hd_mm = intrp_amp.derivative()(t_match)
    phi_mm = intrp_phase(t_match)
    om_mm = intrp_phase.derivative()(t_match)
    # Spherical mode we need in the ringdown
    attach_params = dict(
        amp=h_mm,
        damp=hd_mm,
        omega=om_mm,
        final_mass=final_mass,
        final_spin=final_spin,
    )

    hmm_spherical_ringdown, philm = compute_MR_mode_free(
        t_ringdown,
        m1,
        m2,
        chi1,
        chi2,
        attach_params,
        m,
        m,
        fits_dict,
        t_match=0 * t_match,
        phi_match=phi_mm,
        debug=False,
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
    hlm_spheroidal_ringdown, philm = compute_MR_mode_free(
        t_ringdown,
        m1,
        m2,
        chi1,
        chi2,
        attach_params,
        ell,
        m,
        fits_dict,
        t_match=0 * t_match,
        phi_match=ph_ellm0,
        debug=False,
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
):
    """Given the inspiral modes and the dynamics this function
    computes the NQC coefficients at t_peak-nrDeltaT

    Args:
        inspiral_modes (Dict): Dictionary of inspiral modes (interpolated)
        t_modes (np.ndarray): Time array for inspiral modes
        dynamics (np.ndarray): Dynamics array from ODE solver (unequally spaced)
        t_peak (float): The time of the peak of the orbital frequency
        nrDeltaT (float): The shift from peak of the orbital frequency to peak of (2,2) mode
        m_1 (float): Mass of the primary
        m_2 (float): Mass of the secondary
        chi_1 (float): z-component of dimensionless spin of the primary
        chi_2 (float): z-component Dimensionless spin of the secondary
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
        # NQC_coeffs = EOBCalculateNQCCoefficientsV4(amp,phase,r,pr,omega_orb,ell,m,t_peak,t_modes,m1,m2,chi1,chi2)
        # For equal mass, non-spinning cases odd m modes vanish, so don't try to compute NQCs
        if (
            m % 2
            and np.abs(m_1 - m_2) < 1e-4
            and np.abs(chi_1) < 1e-4
            and np.abs(chi_2) < 1e-4
        ) or (m % 2 and np.abs(m_1 - m_2) < 1e-4 and np.abs(chi_1 - chi_2) < 1e-4):
            continue

            NQC_coeffs["a1"] = 0
            NQC_coeffs["a2"] = 0
            NQC_coeffs["a3"] = 0
            NQC_coeffs["b1"] = 0
            NQC_coeffs["b2"] = 0
            NQC_coeffs["a3S"] = 0
            NQC_coeffs["a4"] = 0
            NQC_coeffs["a5"] = 0
            NQC_coeffs["b3"] = 0
            NQC_coeffs["b4"] = 0

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
                nrDeltaT - extra,  # FIXME for the (5,5) mode
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
    r, pr, omega_orb = polar_dynamics
    for key in hlms.keys():
        ell, m = key
        try:
            NQC_coeffs = nqc_coeffs[(ell, m)]
        except KeyError:
            continue
        correction = EOBNonQCCorrection(r, None, pr, None, omega_orb, NQC_coeffs)
        hlms[key] *= correction
