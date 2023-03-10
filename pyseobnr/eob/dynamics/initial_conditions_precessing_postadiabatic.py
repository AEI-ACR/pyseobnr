#!/usr/bin/env python3
"""
Computes post-adiabatic initial conditions in polar coordinates.
"""

from typing import Callable
from ..hamiltonian import Hamiltonian
import numpy as np

from .postadiabatic_C_prec import compute_postadiabatic_dynamics
from ..fits.fits_Hamiltonian import dSO as dSO_poly_fit
from .initial_conditions_aligned_precessing import computeIC_augm

def compute_IC_PA(
    omega_ref: float,
    omega_start: float,
    H: Hamiltonian,
    RR: Callable,
    chi1_v: np.ndarray,
    chi2_v: np.ndarray,
    m_1: float,
    m_2: float,
    splines: dict,
    t_pn: np.array,
    dynamics_pn: np.array,
    **kwargs,
):
    """Compute the postadiabatic initial conditions for a precessing-spin BBH binary

    Args:
        omega_ref (float): Reference *orbital* frequency in geometric units at which the spins are defined
        omega_start (float): Initial *orbital* frequency in geometric units
        H (Hamiltonian): The Hamiltonian object to use
        RR (function): The RR force to use
        chi1_v (np.ndarray): Dimensionless spin vector of the primary
        chi2_v (np.ndarray): Dimensionless spin vector of the secondary
        m_1 (float): Mass of primary
        m_2 (float): Mass of secondary
        splines (dict): Dictionary containing the splines in orbital frequency of the vector components of the spins, LN and L as
                        well as the spin projections onto LN and L
        t_pn (np.array): Time array of the PN evolution of the spins and Newtonian angular momentum.
        dynamics_pn (np.array): Array of the spin-precessing PN evolution. It contains the Newtonian angular momentum, the dimensionful spin vectors and the PN orbital frequency.

    Returns:
        tuple: The initial conditions: (r,pphi,pr)
    """

    params = kwargs["params"]
    postadiabatic_type = kwargs["postadiabatic_type"]
    PA_success = False

    try :
        postadiabatic_dynamics, omega_pa = compute_postadiabatic_dynamics(
            omega_ref,
            omega_start,
            H, RR,
            chi1_v, chi2_v,
            m_1, m_2,
            splines,
            t_pn,
            dynamics_pn,
            tol=1e-12,
            params=params,
            order=8,
            postadiabatic_type=postadiabatic_type,
            window_length=10,
            only_first_n=12,
        )

        r0 = postadiabatic_dynamics[0, 1]
        pr0 = postadiabatic_dynamics[0, 3]
        pphi0 = postadiabatic_dynamics[0, 4]


        PA_success = True
    except:

        # If PA fails use the adiabatic initial conditions
        r0, pphi0, pr0 = computeIC_augm(
                omega_start,
                H,
                RR,
                chi1_v,
                chi2_v,
                m_1,
                m_2,
                params=params,
            )
        params.p_params.omega = omega_start

    if PA_success:
        # Note that omega and omega_pa[0] may differ due to numerical noise
        # we set the initial frequency to omega_pa[0] to avoid interpolation errors
        params.p_params.omega = omega_pa[0]

    # Update parameters
    X1 = params.p_params.X_1
    X2 = params.p_params.X_2

    tmp = splines["everything"](params.p_params.omega)
    chi1_LN_start = tmp[0]
    chi2_LN_start = tmp[1]
    chi1_L_start = tmp[2]
    chi2_L_start = tmp[3]
    chi1_v_start = tmp[4:7]
    chi2_v_start = tmp[7:10]
    lN_start = tmp[10:13]

    params.p_params.chi1_v[:] = chi1_v_start
    params.p_params.chi2_v[:] = chi2_v_start

    params.p_params.chi_1, params.p_params.chi_2 = chi1_LN_start, chi2_LN_start
    params.p_params.chi1_L, params.p_params.chi2_L = chi1_L_start, chi2_L_start
    params.p_params.lN[:] = lN_start

    params.p_params.update_spins(chi1_LN_start, chi2_LN_start)

    ap_start = chi1_LN_start * X1 + chi2_LN_start * X2
    am_start = chi1_LN_start * X1 - chi2_LN_start * X2

    dSO_start = dSO_poly_fit(params.p_params.nu, ap_start, am_start)
    H.calibration_coeffs["dSO"] = dSO_start

    # Evaluate omega_circ and the Hamiltonian at the intial conditions
    q = np.array([r0, 0.0])
    p = np.array([pr0, pphi0])
    p_circ = np.array([0.0, p[1]])
    dynamics = H.dynamics(q, p, chi1_v_start, chi2_v_start, m_1, m_2, chi1_LN_start, chi2_LN_start, chi1_L_start, chi2_L_start)
    H_val = dynamics[4]
    omega_circ = H.omega(q, p_circ, chi1_v_start, chi2_v_start, m_1, m_2, chi1_LN_start, chi2_LN_start, chi1_L_start, chi2_L_start)
    params.p_params.omega_circ = omega_circ
    params.p_params.H_val = H_val

    #print(f"PA: r0 = {r0}, pphi0 = {pphi0}, pr0 = {pr0}")

    return r0, pphi0, pr0
