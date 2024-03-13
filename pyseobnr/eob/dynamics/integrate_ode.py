#!/usr/bin/env python
"""
Contains functions associated with evolving the equations of motion
"""

import numpy as np
import pygsl_lite.errno as errno
import pygsl_lite.odeiv2 as odeiv2
from numba import jit
from scipy.interpolate import CubicSpline

from ..utils.utils import interpolate_dynamics, iterative_refinement
from .initial_conditions_aligned_opt import computeIC_opt
from .rhs_aligned import augment_dynamics, get_rhs

step = odeiv2.pygsl_lite_odeiv2_step
_control = odeiv2.pygsl_lite_odeiv2_control
evolve = odeiv2.pygsl_lite_odeiv2_evolve


class control_y_new(_control):
    def __init__(self, eps_abs, eps_rel):
        a_y = 1
        a_dydt = 1
        super().__init__(eps_abs, eps_rel, a_y, a_dydt, None)


@jit(nopython=True)
def h_max(r):
    return 1


def ODE_system_RHS_opt(t: float, z: np.ndarray, args) -> np.ndarray:
    """Return the dynamics equations for aligned-spin systems

    Args:
        t (float): The current time
        z (np.array): The dynamics variables, stored as (q,p)
        H (Hamiltonian): The Hamiltonian object to use
        RR (function): The RR force to use. Must have same signature as Hamiltonian
        chi_1 (float): z-component of the primary spin
        chi_2 (float): z-component of the secondary spin
        m_1 (float): mass of the primary
        m_2 (float): mass of the secondary

    Returns:
        np.array: The dynamics equations, including RR
    """
    return get_rhs(t, z, *args)


def compute_dynamics_opt(
    omega0,
    H,
    RR,
    chi_1,
    chi_2,
    m_1,
    m_2,
    rtol=1e-11,
    atol=1e-12,
    backend="solve_ivp",
    params=None,
    step_back=100,
    max_step=0.1,
    min_step=1.0e-9,
    y_init=None,
    r_stop=None,
):
    """
    Main function to integrate the dynamics
    The RHS of the equations are given in Eq(2) of [Buades2021]_ .
    See rhs_aligned.pyx for more details.
    Uses GSL Dormand-Prince 8th order integrator.
    Args:
        omega0 (float): initial orbital frequency
        H (Hamiltonian): The Hamiltonian object to use
        RR (function): The RR force to use. Must have same signature as Hamiltonian
        chi_1 (float): z-component of the primary spin
        chi_2 (float): z-component of the secondary spin
        m_1 (float): mass of the primary
        m_2 (float): mass of the secondary
        rtol (float): relative tolerance
        atol (float): absolute tolerance
        step_back (float): step back for the start of the fine dynamics
        r_stop (float): minimum final separation for the dynamics

    Returns:
        np.array, np.array: coarse and fine dynamics arrays
    """

    sys = odeiv2.system(
        ODE_system_RHS_opt, None, 4, [H, RR, chi_1, chi_2, m_1, m_2, params]
    )

    T = odeiv2.step_rk8pd
    s = step(T, 4)
    c = control_y_new(atol, rtol)
    e = evolve(4)

    t = 0
    t1 = 2.0e9
    if r_stop < 0:
        r_stop = 1.4

    if y_init is None:
        r0, pphi0, pr0 = computeIC_opt(
            omega0, H, RR, chi_1, chi_2, m_1, m_2, params=params
        )
        y0 = np.array([r0, 0.0, pr0, pphi0])
    else:
        y0 = y_init.copy()
    y = y0
    if y_init is None:
        h = 2 * np.pi / omega0 / 5
    else:
        h = 0.5
    omega_previous = omega0
    res_gsl = []
    ts = []

    ts.append(0.0)
    res_gsl.append(y)

    p_circ = np.zeros(2)
    peak_omega = False
    peak_pr = False
    while t < t1:
        # Take a step
        status, t, h, y = e.apply(c, s, sys, t, t1, h, y)
        if status != errno.GSL_SUCCESS:
            print("break status", status)
            break
        # Compute the error for the step controller
        e.get_yerr()

        # Append the last step
        res_gsl.append(y)
        ts.append(t)

        # Compute the RHS after the step is done

        r = y[0]

        # Check if the proposed step is larger than the maximum timestep
        # h_mx = h_max(r)
        # h = h / h_mx

        # Handle termination conditions
        if r <= 6:
            deriv = ODE_system_RHS_opt(t, y, [H, RR, chi_1, chi_2, m_1, m_2, params])
            drdt = deriv[0]
            omega = deriv[1]
            dprdt = deriv[2]
            """
            h_small = np.max((0.01,2*np.pi/(2.*omega) / (1 + np.exp(-(r - 4) / 0.13))))
            if h > h_small:
                h = h_small
            """

            if omega < omega_previous:
                peak_omega = True
                break
            if drdt > 0:
                break
            if dprdt > 0:
                peak_pr = True
                break
            if r <= r_stop:
                break
            if r < 3:
                q_vec = y[:2]
                p_circ[1] = y[-1]
                omega_circ = H.omega(q_vec, p_circ, chi_1, chi_2, m_1, m_2)
                if omega_circ > 1:
                    break
            omega_previous = omega

    ts = np.array(ts)
    dyn = np.array(res_gsl)

    if peak_omega:
        t_desired = ts[-1] - step_back - 50
    else:
        t_desired = ts[-1] - step_back

    idx_close = np.argmin(np.abs(ts - t_desired))
    if ts[idx_close] > t_desired:
        idx_close -= 1

    # Guard against the case where when using PA dynamics,
    # there is less than step_back time between the start
    # of the ODE integration and the end of the dynamics
    # In that case make the fine dynamics be _all_ dynamics
    # except the 1st element
    if t_desired < ts[1]:
        idx_close = 1
        step_back = ts[-1] - ts[idx_close]
    dyn_coarse = np.c_[ts[:idx_close], dyn[:idx_close]]
    dyn_fine = np.c_[ts[idx_close:], dyn[idx_close:]]

    # print(f"t_desired={t_desired}")
    # print(f"len(dyn_coarse)={len(dyn_coarse)}")
    # print(f"len(dyn_fine)={len(dyn_fine)}")
    # print(f"End time: {ts[-1]}")
    # print(f"idx_close:{idx_close}, t[idx_close]={ts[idx_close]}")

    dyn_coarse = augment_dynamics(dyn_coarse, chi_1, chi_2, m_1, m_2, H)
    dyn_fine = augment_dynamics(dyn_fine, chi_1, chi_2, m_1, m_2, H)
    t_peak = None
    if peak_omega:
        intrp = CubicSpline(dyn_fine[:, 0], dyn_fine[:, -2])
        left = dyn_fine[0, 0]
        right = dyn_fine[-1, 0]
        t_peak = iterative_refinement(intrp.derivative(), [left, right])

    if peak_pr:
        intrp = CubicSpline(dyn_fine[:, 0], dyn_fine[:, 3])
        left = dyn_fine[-1, 0] - 10
        right = dyn_fine[-1, 0]
        t_peak = iterative_refinement(intrp.derivative(), [left, right], pr=True)

    dyn_fine = interpolate_dynamics(
        dyn_fine[:, :-3], peak_omega=t_peak, step_back=step_back
    )
    dyn_fine = augment_dynamics(dyn_fine, chi_1, chi_2, m_1, m_2, H)

    return dyn_coarse, dyn_fine
