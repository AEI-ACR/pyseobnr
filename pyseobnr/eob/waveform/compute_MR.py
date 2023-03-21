"""
Contains functions to evaluate the MR ansatze. 
See Eq. (56) and following in https://dcc.ligo.org/DocDB/0186/T2300060/001/SEOBNRv5HM.pdf.
"""

import numpy as np
from ..fits.EOB_fits import *
from ..utils.utils_precession_opt import compute_omegalm_P_frame


def compute_MR_mode_free(
    t, m1, m2, chi1, chi2, attach_params, ell, m, fits_dict,f_nyquist,lmax_nyquist,t_match=0, phi_match=0, qnm_rotation=0.
):
    """
    Evaluate the MR ansatze. See Eqs.(56, 57, 58) in https://dcc.ligo.org/DocDB/0186/T2300060/001/SEOBNRv5HM.pdf.

    Args:
        t (np.ndarray): ringdown time array
        m1 (float): mass of the primary
        m2 (float): mass of the secondary
        chi1 (float): dimensionless spin of the primary
        chi2 (float): dimensionless spin of the secondary
        attach_params (dict): dictionary containing quantities needed to compute the
            MR ansatz (input value fits, final mass and spin)
        ell (int): ell index of the desired mode
        m (int): m index of the desired mode
        t (np.ndarray): inspiral time array
        fits_dict (dict): dictionary of fit coefficients in the ringdown anzatz
        t_match (float): ringdown time at which the merger-ringdown waveform is started
        phi_match (float): value of the phase at t_match
        qnm_rotation (float): Factor rotating the QNM mode frequency in the co-precessing frame (Eq. 33 of Hamilton et al.)

    Returns:
        np.ndarray: the merger-ringdown waveform

    """
    # Step 1 - use the NR fits for amplitude and frequency at attachment time
    eta = m1 * m2 / (m1 + m2) ** 2

    nra = attach_params["amp"]
    nraDot = attach_params["damp"]
    nromega = attach_params["omega"]

    # Step 2 - compute the  QNMs
    final_mass = attach_params["final_mass"]
    final_spin = attach_params["final_spin"]
    omega_complex = compute_QNM(ell, m, 0, final_spin, final_mass).conjugate()
    omega_complex = compute_omegalm_P_frame(omega_complex, m, qnm_rotation)

    # Sanity check on nyquist frequency
    if np.real(omega_complex) > 2*np.pi*f_nyquist and lmax_nyquist >= ell:
        raise ValueError(
                f"Internal function call failed: Input domain error. Ringdown frequency of ({ell},{m}) mode greater than maximum frequency from Nyquist theorem. Decrease the time spacing to resolve this mode."
            )
    sigmaR = -np.imag(omega_complex)
    sigmaI = -np.real(omega_complex)
    # Step 3 - use the fits for the free coefficients in the RD anzatse
    c1f = fits_dict['c1f'][(ell,m)]
    c2f = fits_dict['c2f'][(ell,m)]
    # Ensure that the ampitude of the (2,2) mode has a maximum at t_match
    if ell == 2 and m == 2:
        if sigmaR > 2.0 * c1f * np.tanh(c2f):
            c1f = sigmaR / (2.0 * np.tanh(c2f))

    d1f = fits_dict['d1f'][(ell,m)]
    d2f = fits_dict['d2f'][(ell,m)]
    # Step 4 - compute the constrainted coefficients
    c1c = EOBCalculateRDAmplitudeConstraintedCoefficient1(
        c1f, c2f, sigmaR, nra, nraDot, eta
    )
    c2c = EOBCalculateRDAmplitudeConstraintedCoefficient2(
        c1f, c2f, sigmaR, nra, nraDot, eta
    )
    d1c = EOBCalculateRDPhaseConstraintedCoefficient1(d1f, d2f, sigmaI, nromega)



    # Step 5 - assemble the amplitude and phase
    # First the ansatze part
    Alm = c1c * np.tanh(c1f * (t - t_match) + c2f) + c2c
    philm = phi_match - d1c * np.log(
        (1 + d2f * np.exp(-d1f * (t - t_match))) / (1 + d2f)
    )

    test_omega = -np.real(omega_complex) + 1j * np.imag(omega_complex)
    # hlm = eta*Alm*np.exp(1j*philm)*np.exp(1j*omega_complex*(t-t_match))
    hlm = eta * Alm * np.exp(1j * philm) * np.exp(1j * test_omega * (t - t_match))
    return hlm
