#!/usr/bin/env python3

import numpy as np
from scipy.optimize import root, root_scalar

def IC_cons_augm(u, omega, H, chi1_v, chi2_v, chi1_LN, chi2_LN, chi1_L, chi2_L, m_1, m_2):
    """ The equations defining the 'conservative'
    part of the QC initial conditions, namely
    for r and pphi.

    These are Eq(4.8,4.9) in https://arxiv.org/pdf/gr-qc/0508067.pdf

    Args:
        u ([np.ndarray]): The unknowns, r,pphi
        omega ([float): Desired starting orbital frequency, in gemoetric units
        H (function): The Hamiltonian to use (an instance of Hamiltonian class)
        chi_1 (float): z-component of the primary spin
        chi_2 (float): z-component of the secondary spin
        m_1 (float): mass of the primary
        m_2 (float): mass of the secondary

    Returns:
        [np.ndarray]: The desired equations evaluated at u
    """
    r, pphi = u

    q = np.array([r, 0.0])
    p = np.array([0.0, pphi])

    #print(q, p, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L)
    grad = H.dynamics(q, p, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L)
    dHdr = grad[0]
    dHdpphi = grad[3]
    diff = np.array([dHdpphi - omega, dHdr])
    return diff


def IC_diss_augm(u, r, pphi, H, RR, chi1_v, chi2_v, chi1_LN, chi2_LN, chi1_L, chi2_L, m_1, m_2, params):
    """Initial conditions for the "dissipative" part,
    namely pr

    This is basically Eq(4.15) in https://arxiv.org/pdf/gr-qc/0508067.pdf
    Args:
        u (float): Guess for pr
        r (floart): Starting separation
        pphi (float): Starting angular momentum
        H (Hamiltonian): The Hamiltonian object to use
        RR (function): Function that returns the RR force. Must have same signature as the Hamiltonian
        chi_1 (float): z-component of the dimensionless spin of primary
        chi_2 (float): z-component of the dimensionless spin of secondary
        m_1 (float): Mass of primary
        m_2 (float): Mass of secondary

    Returns:
        float: The equation for pr
    """
    pr = u
    q = np.array([r, 0.0])
    p = np.array([pr, pphi])
    hess = H.hessian(q, p,chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L)
    d2Hdr2 = hess[0, 0]
    d2HdrdL = hess[3, 0]
    dLdr = -d2Hdr2 / d2HdrdL
    p_circ = np.array([0.0, p[1]])
    dynamics = H.dynamics(q, p, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L)
    H_val = dynamics[4]
    omega = dynamics[3]
    omega_circ = H.omega(q, p_circ, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L)
    RR_f = RR.RR(q, p, omega, omega_circ, H_val, params)
    csi = dynamics[5]

    rdot = 1 / csi * RR_f[1] / dLdr
    dHdpr = dynamics[2]
    return rdot - dHdpr

def computeIC_augm(omega, H, RR, chi1_v, chi2_v, chi_1, chi_2, m_1, m_2, **kwargs):
    """Compute the initial conditions for an aligned-spin BBH binary

    Args:
        omega (float): Initial *orbital* frequency in geometric units
        H (Hamiltonian): The Hamiltonian object to use
        RR (function): The RR force to use
        chi_1 (float): z-component of the dimensionless spin of primary
        chi_2 (float): z-component of the dimensionless spin of secondary
        m_1 (float): Mass of primary
        m_2 (float): Mass of secondary

    Returns:
        tuple: The initial conditions: (r,pphi,pr)
    """
    # Initial guess from Newtonian gravity

    params = kwargs["params"]


    chi1_LN, chi2_LN = params.p_params.chi_1, params.p_params.chi_2
    chi1_L, chi2_L = params.p_params.chi1_L, params.p_params.chi2_L

    r_guess = omega ** (-2.0 / 3)
    z = [r_guess, np.sqrt(r_guess)]
    # print(f"Initial guess is {z}")
    # The conservative bit: solve for r and pphi
    res_cons = root(IC_cons_augm, z, args=(omega, H, chi1_v, chi2_v, chi1_LN, chi2_LN, chi1_L, chi2_L, m_1, m_2), tol=1e-12)
    r0, pphi0 = res_cons.x
    # print(f"Computed conservative stuff, {r0},{pphi0}")

    res_diss = root_scalar(
        IC_diss_augm,
        bracket=[-3e-2, 0],
        args=(r0, pphi0, H, RR, chi1_v, chi2_v, chi1_LN, chi2_LN, chi1_L, chi2_L, m_1, m_2,kwargs["params"]),
        xtol=1e-12,
        rtol=1e-10,
    )
    # Now do the dissipative bit: solve for pr
    pr0 = res_diss.root
    # print(f"Computed dissipative stuff, {pr0}")
    return r0, pphi0, pr0
