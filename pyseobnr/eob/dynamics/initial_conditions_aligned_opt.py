#!/usr/bin/env python3
"""
Computes the aligned-spin initial conditions in polar coordinates.
"""

import logging
import numpy as np
from scipy.optimize import root, root_scalar

from rich.logging import RichHandler
from rich.traceback import install

# Setup the logger to work with rich
logger = logging.getLogger(__name__)
logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))
logger.setLevel("INFO")
# Setup rich to get nice tracebacks
install()


def IC_cons(u, omega, H, chi_1, chi_2, m_1, m_2):
    """The equations defining the 'conservative'
    part of the QC initial conditions, namely
    for r and pphi.

    This is Eq(60) in [Khalil2021]_ .

    Args:
        u ([np.ndarray]): The unknowns, r,pphi
        omega (float): Desired starting orbital frequency, in geometric units
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
    grad = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
    dHdr = grad[0]
    dHdpphi = grad[3]
    diff = np.array([dHdpphi - omega, dHdr])
    return diff


def IC_diss(u, r, pphi, H, RR, chi_1, chi_2, m_1, m_2, params):
    """Initial conditions for the "dissipative" part,
    namely pr.

    This is Eq(68) in [Khalil2021]_ .

    Note that RR_f[1] is 1/Omega*dE/dt

    Args:
        u (float): Guess for pr
        r (float): Starting separation
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
    hess = H.hessian(q, p, chi_1, chi_2, m_1, m_2)
    d2Hdr2 = hess[0, 0]
    d2HdrdL = hess[3, 0]
    dLdr = -d2Hdr2 / d2HdrdL
    p_circ = np.array([0.0, p[1]])
    dynamics = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
    H_val = dynamics[4]
    omega = dynamics[3]
    omega_circ = H.omega(q, p_circ, chi_1, chi_2, m_1, m_2)
    RR_f = RR.RR(q, p, omega, omega_circ, H_val, params)
    xi = dynamics[5]

    rdot = 1 / xi * RR_f[1] / dLdr
    dHdpr = dynamics[2]
    return rdot - dHdpr


def computeIC_opt(omega, H, RR, chi_1, chi_2, m_1, m_2, **kwargs):
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
    r_guess = omega ** (-2.0 / 3)
    z = [r_guess, np.sqrt(r_guess)]
    # print(f"Initial guess is {z}")
    # The conservative bit: solve for r and pphi
    res_cons = root(IC_cons, z, args=(omega, H, chi_1, chi_2, m_1, m_2), tol=6e-12)
    if not res_cons.success:
        logger.error(
            f"The solution for the conservative part of initial conditions failed for"
            f" m1={m_1},m2={m_2},chi1={chi_1},chi2={chi_2},omega={omega}"
        )

    r0, pphi0 = res_cons.x
    # print(f"Computed conservative stuff, {r0},{pphi0}")

    res_diss = root_scalar(
        IC_diss,
        bracket=[-3e-2, 0],
        args=(r0, pphi0, H, RR, chi_1, chi_2, m_1, m_2, kwargs["params"]),
        xtol=1e-12,
        rtol=1e-10,
    )
    if not res_diss.converged:
        logger.error(
            "The solution for the dissipative part of initial conditions failed for"
            f" m1={m_1},m2={m_2},chi1={chi_1},chi2={chi_2},omega={omega}"
        )
    # Now do the dissipative bit: solve for pr
    pr0 = res_diss.root
    return r0, pphi0, pr0
