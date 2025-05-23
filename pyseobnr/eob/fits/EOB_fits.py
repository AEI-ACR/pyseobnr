"""
Contains functions that are used in evaluating different waveform fits.

Also includes computation of NQCs
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import qnm  # initializes the logging and prevents proper setup in __main__
from scipy.interpolate import InterpolatedUnivariateSpline

# Pre-cache the QNM interpolants
modes_qnm = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)]
_qnm_cache = {}
_qnm_cache_mm = {}


def compute_QNM(ell: int, m: int, n: int, af: float, Mf: float):
    """Return the complex QNM frequencies by interpolating
    existing solutions.

    Note that for negative spins we use
    the _positive_ spin negative ``m`` call internally

    Args:
        ell (int): :math:`\\ell` of the desired QNM
        m (int): m of the desired QNM
        n (int): overtone. Only n=0 is supported
        af (float): final spin of the BH
        Mf (float): final mass of the BH

    Returns:
        np.complex128: The complex QNM frequency.
        The frequencies are such that Re(omega)>0 and Im(omega)<0.
    """
    global _qnm_cache, _qnm_cache_mm

    if af > 0:
        if (ell, m) not in _qnm_cache:
            _qnm_cache[(ell, m)] = qnm.modes_cache(s=-2, l=ell, m=m, n=0)
        omega, _, _ = _qnm_cache[(ell, m)](a=af, interp_only=True)
    else:
        if (ell, m) not in _qnm_cache_mm:
            _qnm_cache_mm[(ell, m)] = qnm.modes_cache(s=-2, l=ell, m=-m, n=0)
        omega, _, _ = _qnm_cache_mm[(ell, m)](a=np.abs(af), interp_only=True)
    return omega / Mf


def EOBCalculateRDAmplitudeConstraintedCoefficient1(
    c1f: float, c2f: float, sigmaR: float, amp: float, damp: float, eta: float
):
    """
    Computes c1c coefficient in MR ansatze.

    See Eq(59) in [SEOBNRv5HM-notes]_ .

    Args:
        c1f (float): value of c1f
        c2f (float): value of c1f
        sigmaR (float): real part of the QNM frequency
        amp (float): waveform amplitude at attachment time
        damp (float): waveform amplitude's first derivative at attachment time
        eta (float): reduced mass ratio

    Returns:
        float: The value of c1c
    """
    c1c = 1 / (c1f * eta) * (damp - sigmaR * amp) * np.cosh(c2f) ** 2
    return c1c


def EOBCalculateRDAmplitudeConstraintedCoefficient2(
    c1f: float, c2f: float, sigmaR: float, amp: float, damp: float, eta: float
):
    """
    Computes c2c coefficient in MR ansatze.

    See Eq(60) in [SEOBNRv5HM-notes]_ .

    Args:
        c1f (float): value of c1f
        c2f (float): value of c1f
        sigmaR (float): real part of the QNM frequency
        amp (float): waveform amplitude at attachment time
        damp (float): waveform amplitude's first derivative at attachment time
        eta (float): reduced mass ratio

    Returns:
        float: The value of c2c
    """
    c2c = amp / eta - 1 / (c1f * eta) * (damp - sigmaR * amp) * np.cosh(c2f) * np.sinh(
        c2f
    )
    return c2c


def EOBCalculateRDPhaseConstraintedCoefficient1(
    d1f: float, d2f: float, sigmaI: float, omega: float
):
    """
    Computes d1c coefficient in MR ansatze.

    See Eq(61) in [SEOBNRv5HM-notes]_ .

    Args:
        d1f (float): value of d1f
        d2f (float): value of d1f
        sigmaI (float): imaginary part of the QNM frequency
        omega (float): waveform frequency at attachment time

    Returns:
        float: The value of c2c
    """
    d1c = (omega - sigmaI) * (1 + d2f) / (d1f * d2f)
    return d1c


def EOBCalculateNQCCoefficients_freeattach(
    amplitude,
    phase,
    r,
    pr,
    omega_orb,
    ell,
    m,
    time_peak: float,
    time,
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
    nrDeltaT: float,
    fits_dict: Dict[str, Any],
):
    """
    Computes the NQC coefficients

    See discussion in Sec. II A, around Eq(35), in [SEOBNRv5HM-notes]_ .

    This is just like the SEOBNRv4HM function but allows nrDeltaT to be passed in, instead of
    internally calculated.

    Args:
        amplitude (np.ndarray): amplitude of the relevant mode
        phase (np.ndarray): phase of the relevant mode
        r (np.ndarray): r along the dynamics
        pr (np.ndarray): pr along the dynamics
        omega_orb (np.ndarray): omega_orb along the dynamics
        ell (int): :math:`\\ell` index of the relevant mode
        m (int): m index of the relevant mode
        time_peak (float): "reference" time with respect to which the attachment
            is defined, corresponding to t_ISCO for SEOBRNRv5HM (see Eq. 41 of
            [SEOBNRv5HM-notes]_ )
        time (np.ndarray): time array
        m1 (float): primary mass
        m2 (float): secondary mass
        chi1 (float): primary mass
        chi2 (float): primary mass
        nrDeltaT (float): time difference between the peak of the (2,2) mode and the reference time
        fits_dict (dict): dictionary containing the input-value fits

    Returns:
        Dict: dictionary containing the NQC coefficients
    """

    coeffs = {}

    eta = (m1 * m2) / ((m1 + m2) * (m1 + m2))
    # Eq (3) in LIGO DCC document T1100433v2
    rOmega = r * omega_orb
    q1 = pr**2 / rOmega**2
    q2 = q1 / r
    q3 = q2 / np.sqrt(r)

    # Eq (4) in LIGO DCC document T1100433v2
    p1 = pr / rOmega
    p2 = p1 * pr**2

    # See below Eq(9)
    q1LM = q1 * amplitude
    q2LM = q2 * amplitude
    q3LM = q3 * amplitude

    # Compute the attachment time
    # This is given by Eq.(41) of [SEOBNRv5HM-notes]_
    # with time_peak <-> t_ISCO and nrDeltaT <-> - DeltaT_22
    # i.e. nrDeltaT > 0 means the attachment time is before t_ISCO
    nrTimePeak = time_peak - nrDeltaT
    if nrTimePeak > time[-1]:
        # If the predicted time is after the end of the dynamics
        # use the end of the dynamics
        nrTimePeak = time[-1]
        if ell == 5 and m == 5:
            nrTimePeak -= 10  # (5,5) mode always 10 M before peak of (2,2)
    if nrTimePeak < time[0]:
        # In the eccentric model, for very short systems the attachment time
        # can overshoot the beginning of the fine dynamics, so here we put
        # the attachment time at the beginning of the fine dynamics
        nrTimePeak = time[0]

    # Compute amplitude NQCs (a1,a2,a3) by solving Eq.(10) of T1100433v2
    # Evaluate the Qs at the right time and build Q matrix (LHS)
    idx = np.argmin(np.abs(time - nrTimePeak))
    N = 5
    left = np.max((0, idx - N))
    right = np.min((idx + N, len(time)))

    intrp_q1LM = InterpolatedUnivariateSpline(time[left:right], q1LM[left:right])
    intrp_q2LM = InterpolatedUnivariateSpline(time[left:right], q2LM[left:right])
    intrp_q3LM = InterpolatedUnivariateSpline(time[left:right], q3LM[left:right])

    Q = np.zeros((3, 3))

    Q[:, 0] = intrp_q1LM.derivatives(nrTimePeak)[:-1]
    Q[:, 1] = intrp_q2LM.derivatives(nrTimePeak)[:-1]
    Q[:, 2] = intrp_q3LM.derivatives(nrTimePeak)[:-1]

    # Build the RHS of Eq.(10)
    # Compute the NR fits
    nra = eta * fits_dict["amp"][(ell, m)]
    nraDot = eta * fits_dict["damp"][(ell, m)]
    nraDDot = eta * fits_dict["ddamp"][(ell, m)]

    # Compute amplitude and derivatives at the right time
    intrp_amp = InterpolatedUnivariateSpline(time[left:right], amplitude[left:right])

    amp, damp, ddamp = intrp_amp.derivatives(nrTimePeak)[:-1]

    # Assemble RHS
    amps = np.array([nra - amp, nraDot - damp, nraDDot - ddamp])

    # Solve the equation Q*coeffs = amps
    res = np.linalg.solve(Q, amps)

    coeffs["a1"] = res[0]
    coeffs["a2"] = res[1]
    coeffs["a3"] = res[2]

    # Now we (should) have calculated the a values.
    # We now compute the frequency NQCs (b1,b2) by solving Eq.(11) of T1100433v2
    # Populate the P matrix in LHS of Eq.(11) of T1100433v2
    intrp_p1 = InterpolatedUnivariateSpline(time[left:right], p1[left:right])
    intrp_p2 = InterpolatedUnivariateSpline(time[left:right], p2[left:right])
    P = np.zeros((2, 2))

    P[:, 0] = -intrp_p1.derivatives(nrTimePeak)[1:-1]
    P[:, 1] = -intrp_p2.derivatives(nrTimePeak)[1:-1]

    # Build the RHS of Eq.(11)
    # Compute frequency and derivative at the right time
    intrp_phase = InterpolatedUnivariateSpline(time[left:right], phase[left:right])
    omega, omegaDot = intrp_phase.derivatives(nrTimePeak)[1:-1]

    # Since the phase can be decreasing, we need to take care not to have a -ve frequency
    if omega * omegaDot > 0.0:
        omega = np.abs(omega)
        omegaDot = np.abs(omegaDot)
    else:
        omega = np.abs(omega)
        omegaDot = -np.abs(omegaDot)

    # Compute the NR fits
    nromega = np.abs(fits_dict["omega"][(ell, m)])
    nromegaDot = np.abs(fits_dict["domega"][(ell, m)])

    # Assemble RHS
    omegas = np.array([nromega - omega, nromegaDot - omegaDot])

    # Solve the equation P*coeffs = omegas
    res = np.linalg.solve(P, omegas)
    coeffs["b1"] = res[0]
    coeffs["b2"] = res[1]
    return coeffs


class EOBNonQCCorrectionImpl:
    """Returns the NQC correction array as a function of the NQC coefficients

    The implementation reuses the intermediate calculations.
    """

    def __init__(self, r, phi, pr, pphi, omega):

        sqrtR = np.sqrt(r)
        rOmega = r * omega
        rOmegaSq = rOmega * rOmega
        p = pr

        self.mc1 = p * p / rOmegaSq
        self.mc2 = 1.0 / r
        self.mc3 = 1.0 / (r * sqrtR)
        self.mc4 = 1.0 / (r * r)
        self.mc5 = self.mc3 * self.mc2  # 1.0 / (r * r * sqrtR)

        self.pc1 = p / rOmega
        self.pc2 = p * p * p / rOmega
        self.pc3 = 1.0 / sqrtR
        self.pc4 = self.mc2

    def get_nqc_multiplier(self, coeffs):
        mag = 1.0 + self.mc1 * (
            coeffs["a1"]
            + coeffs["a2"] * self.mc2
            + (coeffs["a3"] + coeffs["a3S"]) * self.mc3
            + coeffs["a4"] * self.mc4
            + coeffs["a5"] / self.mc5
        )
        phase = coeffs["b1"] * self.pc1 + self.pc2 * (
            coeffs["b2"] + coeffs["b3"] * self.pc3 + coeffs["b4"] * self.pc4
        )

        return mag * np.exp(1j * phase)
