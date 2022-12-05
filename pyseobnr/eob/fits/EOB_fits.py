from typing import Any, Dict

import numpy as np
import qnm
from scipy.interpolate import InterpolatedUnivariateSpline

# Pre-cache the QNM interpolants
modes_qnm = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)]
qnm_cache = {}
qnm_cache_mm = {}
for mode in modes_qnm:
    ell, m = mode
    qnm_cache[ell, m] = qnm.modes_cache(s=-2, l=ell, m=m, n=0)
    qnm_cache_mm[ell, m] = qnm.modes_cache(s=-2, l=ell, m=-m, n=0)


def compute_QNM(ell: int, m: int, n: int, af: float, Mf: float):
    """Return the complex QNM frequencies by interpolating
    existing solutions. Note that for negative spins we use
    the _positive_ spin negative m call internally

    Args:
        ell (int): ell of the desired QNM
        m (int): m of the desired QNM
        n (int): overtone. Only n=0 is supported
        af (float): final spin of the BH
        Mf (float): final mass of the BH

    Returns:
        np.complex128: The complex QNM frequency
    """
    if af > 0:
        omega, _, _ = qnm_cache[(ell, m)](a=af, interp_only=True)
    else:
        omega, _, _ = qnm_cache_mm[(ell, m)](a=np.abs(af), interp_only=True)
    return omega / Mf


def EOBCalculateRDAmplitudeConstraintedCoefficient1(
    c1f: float, c2f: float, sigmaR: float, amp: float, damp: float, eta: float
):
    """
    Computes c1c coefficient in MR ansatze (Eq 59 in v5HM doc).

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
    Computes c2c coefficient in MR ansatze (Eq 60 in v5HM doc).

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
    Computes d1c coefficient in MR ansatze (Eq 61 in v5HM doc).

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
    Computes the NQC coefficients (see discussion in Sec. II A, around Eq(35), in v5HM doc)
    This is just like the SEOBNRv4HM function but allows nrDeltaT to be passed in, instead of
    internally calculated.

    Args:
        ...

    Returns:
        Dict: dictionary containing the NQC coefficients
    """

    coeffs = {}

    eta = (m1 * m2) / ((m1 + m2) * (m1 + m2))
    # Eq (3) in T1100433
    rOmega = r * omega_orb
    q1 = pr**2 / rOmega**2
    q2 = q1 / r
    q3 = q2 / np.sqrt(r)

    # Eq (4) in T1100433
    p1 = pr / rOmega
    p2 = p1 * pr**2

    # See below Eq(9)
    q1LM = q1 * amplitude
    q2LM = q2 * amplitude
    q3LM = q3 * amplitude

    # Compute the attachment time
    nrTimePeak = time_peak - nrDeltaT
    if nrTimePeak > time[-1]:
        # If the predicted time is after the end of the dynamics
        # use the end of the dynamics
        nrTimePeak = time[-1]
        if ell == 5 and m == 5:
            nrTimePeak -= 10  # (5,5) mode always 10 M before peak of (2,2)

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

    # Build the RHS
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

    # Now we (should) have calculated the a values. Now we can do the b values
    # Populate the P matrix in Eq. 18 of the LIGO DCC document T1100433v2
    intrp_p1 = InterpolatedUnivariateSpline(time[left:right], p1[left:right])
    intrp_p2 = InterpolatedUnivariateSpline(time[left:right], p2[left:right])
    P = np.zeros((2, 2))

    P[:, 0] = -intrp_p1.derivatives(nrTimePeak)[1:-1]
    P[:, 1] = -intrp_p2.derivatives(nrTimePeak)[1:-1]
    intrp_phase = InterpolatedUnivariateSpline(time[left:right], phase[left:right])
    omega, omegaDot = intrp_phase.derivatives(nrTimePeak)[1:-1]

    # Since the phase can be decreasing, we need to take care not to have a -ve frequency
    if omega * omegaDot > 0.0:
        omega = np.abs(omega)
        omegaDot = np.abs(omegaDot)
    else:
        omega = np.abs(omega)
        omegaDot = -np.abs(omegaDot)

    nromega = np.abs(fits_dict["omega"][(ell, m)])
    nromegaDot = np.abs(fits_dict["domega"][(ell, m)])

    omegas = np.array([nromega - omega, nromegaDot - omegaDot])
    # Solve the equation Q*coeffs = amps
    res = np.linalg.solve(P, omegas)
    coeffs["b1"] = res[0]
    coeffs["b2"] = res[1]
    return coeffs


def EOBNonQCCorrection(r, phi, pr, pphi, omega, coeffs):
    """
    Evaluate the NQC correction, given the coefficients.

    Args:
        ...
        
    Returns:
        ...
        
    """
    sqrtR = np.sqrt(r)
    rOmega = r * omega
    rOmegaSq = rOmega * rOmega
    p = pr
    mag = 1.0 + (p * p / rOmegaSq) * (
        coeffs["a1"]
        + coeffs["a2"] / r
        + (coeffs["a3"] + coeffs["a3S"]) / (r * sqrtR)
        + coeffs["a4"] / (r * r)
        + coeffs["a5"] / (r * r * sqrtR)
    )
    phase = coeffs["b1"] * p / rOmega + p * p * p / rOmega * (
        coeffs["b2"] + coeffs["b3"] / sqrtR + coeffs["b4"] / r
    )

    nqc = mag * np.cos(phase) + 1j * mag * np.sin(phase)
    return nqc
