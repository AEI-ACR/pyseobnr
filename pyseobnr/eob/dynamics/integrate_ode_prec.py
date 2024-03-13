"""
Contains functions associated with evolving the equations of motion.
"""

from typing import Callable

import numpy as np
import pygsl_lite.errno as errno
import pygsl_lite.odeiv2 as odeiv2
from numba import *
from scipy.interpolate import CubicSpline

from ..fits.fits_Hamiltonian import dSO as dSO_poly_fit
from ..hamiltonian import Hamiltonian
from ..utils.containers import EOBParams
from ..utils.math_ops_opt import *
from ..utils.utils import interpolate_dynamics, iterative_refinement
from .initial_conditions_aligned_precessing import computeIC_augm
from .pn_evolution_opt import (
    build_splines_PN,
    compute_omega_orb,
    compute_quasiprecessing_PNdynamics_opt,
    rhs_wrapper,
)

# Test cythonization of PN equations


step = odeiv2.pygsl_lite_odeiv2_step
_control = odeiv2.pygsl_lite_odeiv2_control
evolve = odeiv2.pygsl_lite_odeiv2_evolve


class control_y_new(_control):
    def __init__(self, eps_abs, eps_rel):
        a_y = 1
        a_dydt = 1
        _control.__init__(self, eps_abs, eps_rel, a_y, a_dydt, None)


# Function to terminate the EOB evolution
def check_terminal(
    r: float,
    omega: float,
    drdt: float,
    dprdt: float,
    omega_circ: float,
    omega_previous: float,
    r_previous: float,
    omegaPN_f: float,
) -> int:
    """
    Check termination condition of the EOB evolution.

    Args:
        r (float): Orbital separation
        omega (float): Orbital frequency
        drdt (float): Time derivative of r
        dprdt (float): Time derivative of prstar
        omega_circ (float): Circular orbital frequency
        omega_previous (float): Orbital frequency at the previous timestep
        r_previous (float): Orbital separation at the previous timestep
        omegaPN_f (float): Final orbital frequency reached in the spin-precessing PN evolution

    Returns:
        (int): If >0 terminates EOB dynamics
    """

    if r <= 1.4:
        # print(f"r = {r} < 1.4, omega = {omega}")
        return 1

    if np.isnan(omega) or np.isnan(drdt) or np.isnan(dprdt) or np.isnan(omega_circ):
        # print(f"nan omega: r = {r}, omega = {omega}")
        return 2

    if omega < omega_previous:
        # print(f"r = {r}, omega = {omega}, omega_previous = {omega_previous}")
        return 3

    if r < 3:
        if drdt > 0:
            # print(f"drdt>0 : r = {r}, omega = {omega}, omega_previous = {omega_previous}")
            return 5

        if dprdt > 0:
            # print(f"dprdt >0 : r = {r}, omega = {omega}, omega_previous = {omega_previous}")
            return 6

        if omega_circ > 1:
            # print(f"omega_circ >1 : r = {r}, omega = {omega}, omega_previous = {omega_previous}")
            return 7

        if r > r_previous:
            # print(f"r> r_previous : r = {r}, omega = {omega}, omega_previous = {omega_previous}, r_previous = {r_previous}")
            return 8

    if omega > omegaPN_f:
        # print(f"omega> omegaPN_f : r = {r}, omega = {omega}, omegaPN_f = {omegaPN_f}")
        return 9

    return 0


def compute_dynamics_prec_opt(
    omega_ref: float,
    omega_start: float,
    omegaPN_f: float,
    H: Hamiltonian,
    RR: Callable,
    m_1: float,
    m_2: float,
    splines: dict,
    t_pn: np.array,
    dynamics_pn: np.array,
    params: EOBParams,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    step_back: float = 250.0,
    y_init=None,
    initial_conditions: str = "adiabatic",
    initial_conditions_postadiabatic_type: str = "analytic",
):
    """
    Function to perform a non-precessing EOB evolution with the spins modified
    at every timestep according to the values from the precessing-spin PN
    evolution.

    Args:
        omega_ref (float): Reference orbital frequency at which the spins are defined
        omega_start (float): Starting orbital frequency
        omegaPN_f (float): Final orbital frequency from the precessing-spin PN evolution
        H (Hamiltonian): Hamiltonian class
        RR (Callable): RR force
        m_1 (float): Mass component of the primary
        m_2 (float): Mass component of the secondary
        splines (dict): Dictionary containing the splines in orbital frequency of the vector components of the spins, LN and L as
                        well as the spin projections onto LN and L
        t_pn (np.array): Time array of the PN evolution of the spins and Newtonian angular momentum.
        dynamics_pn (np.array): Array of the spin-precessing PN evolution. It contains the Newtonian angular momentum, the dimensionful spin vectors and the PN orbital frequency.
        params (EOBParams): Container of additional inputs
        rtol (float, optional): Relative tolerance for EOB integration. Defaults to 1e-12
        atol (float, optional): Absolute tolerance for EOB integration. Defaults to 1e-12
        step_back (float, optional): Amount of time to step back for fine interpolation. Defaults to 250.
        y_init (np.ndarray, optional): Initial condition vector (r,phi,pr,pphi).
        initial_conditions (str, optional): Type of initial conditions for the ODE evolution ('adiabatic' or 'postadiabatic').
        initial_conditions_postadiabatic_type (str, optional): Type of postadiabatic initial conditions for the ODE evolution ('analytic' or 'numeric').

    Returns:
        (tuple): Low and high sampling rate dynamics, unit Newtonian orbital angular momentum, assembled dynamics
                 and the index splitting the low and high sampling rate dynamics
    """

    # Step 3 : Evolve EOB dynamics

    # Step 3.1) Update spin parameters and calibration coeffs (only dSO) at the start of EOB integration

    X1 = params.p_params.X_1
    X2 = params.p_params.X_2

    tmp = splines["everything"](omega_start)
    chi1_LN_start = tmp[0]
    chi2_LN_start = tmp[1]
    chi1_L_start = tmp[2]
    chi2_L_start = tmp[3]
    chi1_v_start = tmp[4:7]
    chi2_v_start = tmp[7:10]

    params.p_params.omega = omega_start
    params.p_params.chi1_v[:] = chi1_v_start
    params.p_params.chi2_v[:] = chi2_v_start

    params.p_params.chi_1, params.p_params.chi_2 = chi1_LN_start, chi2_LN_start
    params.p_params.chi1_L, params.p_params.chi2_L = chi1_L_start, chi2_L_start

    params.p_params.update_spins(chi1_LN_start, chi2_LN_start)

    ap_start = chi1_LN_start * X1 + chi2_LN_start * X2
    am_start = chi1_LN_start * X1 - chi2_LN_start * X2

    dSO_start = dSO_poly_fit(params.p_params.nu, ap_start, am_start)
    H.calibration_coeffs["dSO"] = dSO_start

    # Step 3.2: compute the initial conditions - uses the aligned-spin ID
    if y_init is None:
        if initial_conditions == "adiabatic":
            r0, pphi0, pr0 = computeIC_augm(
                omega_start,
                H,
                RR,
                chi1_v_start,
                chi2_v_start,
                m_1,
                m_2,
                params=params,
            )

        elif initial_conditions == "postadiabatic":
            from .initial_conditions_precessing_postadiabatic import compute_IC_PA

            r0, pphi0, pr0 = compute_IC_PA(
                omega_ref,
                omega_start,
                H,
                RR,
                chi1_v_start,
                chi2_v_start,
                m_1,
                m_2,
                splines,
                t_pn,
                dynamics_pn,
                params=params,
                postadiabatic_type=initial_conditions_postadiabatic_type,
            )
        y0 = np.array([r0, 0.0, pr0, pphi0])

    else:
        y0 = y_init.copy()
        r0 = y0[0]

    # Step 3.3: now integrate the dynamics
    sys = odeiv2.system(rhs_wrapper, None, 4, [H, RR, m_1, m_2, params])

    T = odeiv2.step_rkf45
    s = step(T, 4)
    c = control_y_new(atol, rtol)
    e = evolve(4)

    t = 0
    t1 = 1.0e9

    y = y0

    if y_init is None:
        h = 2 * np.pi / omega_start / 100.0
    else:
        h = 0.1

    # Use an agnostic and small initial step for the integrator
    h = 0.1
    res_gsl = []
    ts = []
    omegas = []
    augm_dyn = []

    # Convert to numpy array to get the correct value
    augm_dyn.append(
        [
            params.p_params.H_val,
            params.p_params.omega,
            params.p_params.omega_circ,
            params.p_params.chi_1,
            params.p_params.chi_2,
        ]
    )
    ts.append(0.0)
    res_gsl.append(y)
    omegas.append(params.p_params.omega)

    omega_previous = omega_start
    r_previous = r0

    X1 = params.p_params.X_1
    X2 = params.p_params.X_2

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

        r = y[0]

        if np.isnan(r):
            # print(f"t = {t}, r = {r}, h = {h}")
            break
        else:
            # Append the last step
            res_gsl.append(y)
            ts.append(t)
            augm_dyn.append(
                [
                    params.p_params.H_val,
                    params.p_params.omega,
                    params.p_params.omega_circ,
                    params.p_params.chi_1,
                    params.p_params.chi_2,
                ]
            )

        # Handle termination conditions
        if r <= 6:
            deriv = rhs_wrapper(t, y, [H, RR, m_1, m_2, params])
            drdt = deriv[0]
            omega = deriv[1]
            omega_previous = omegas[-1]
            if np.isnan(omega):
                res_gsl = res_gsl[:-1]
                ts = ts[:-1]
                augm_dyn = augm_dyn[:-1]
                break

            else:
                omegas.append(omega)
                dprdt = deriv[2]
                omega_circ = params.p_params.omega_circ

            # check termination conditions
            termination = check_terminal(
                r, omega, drdt, dprdt, omega_circ, omega_previous, r_previous, omegaPN_f
            )
            if termination:
                if termination == 3:
                    peak_omega = True
                if termination == 6:
                    peak_pr = True
                break

            omega_previous = omega

        else:
            omega = compute_omega_orb(t, y, H, RR, m_1, m_2, params)
            omegas.append(omega)

        # Update spin parameters and calibration coeffs (only dSO)
        params.p_params.omega = omega
        omega_circ = params.p_params.omega_circ

        # Update spins according to the new value of orbital frequency
        tmp = splines["everything"](omega)
        chi1_LN = tmp[0]
        chi2_LN = tmp[1]
        chi1_L = tmp[2]
        chi2_L = tmp[3]
        chi1_v = tmp[4:7]
        chi2_v = tmp[7:10]
        tmp_LN = tmp[10:13]

        params.p_params.chi1_v[:] = chi1_v
        params.p_params.chi2_v[:] = chi2_v
        # params.p_params.lN[:] = tmp_LN#/my_norm(tmp_LN)

        ap = chi1_LN * X1 + chi2_LN * X2
        am = chi1_LN * X1 - chi2_LN * X2

        # Check for possible extrapolation in the spin values
        if abs(ap) > 1 or abs(am) > 1:
            # print(f"r = {r}, omega = {omega}, ap = {ap}, am = {am}")
            break

        dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)
        H.calibration_coeffs["dSO"] = dSO_new

        params.p_params.chi_1 = chi1_LN
        params.p_params.chi_2 = chi2_LN

        params.p_params.chi1_L = chi1_L
        params.p_params.chi2_L = chi2_L

        # Remember to update the derived spin quantities which enter the flux
        params.p_params.update_spins(params.p_params.chi_1, params.p_params.chi_2)

    # Step 3.4: After the integration assemble quantities and apply a roll-off of the timestep
    #           to avoid abrupt changes in the splines
    ts = np.array(ts)
    dyn = np.array(res_gsl)
    omega_eob = np.array(omegas)
    augm_dyn = np.array(augm_dyn)

    dyn = np.c_[dyn, augm_dyn]

    # Step 3.5: Iterative procedure to find the peak of omega or pr to have a more robust model
    #           under perturbations

    ##################################################################

    # Estimate of the starting point of fine integration
    # Same refinement as in the aligned-spin model
    t_stop = ts[-1]
    if peak_omega:
        t_desired = t_stop - step_back - 50
    else:
        t_desired = t_stop - step_back

    idx_close = np.argmin(np.abs(ts - t_desired))
    t_fine = ts[idx_close:]
    dyn_fine = np.c_[t_fine, dyn[idx_close:]]
    omega_fine = omega_eob[idx_close:]

    t_peak = None
    if peak_omega:
        idx_max = np.argmax(omega_fine)
        omega_peak = omega_fine[idx_max]
        t_peak = t_fine[idx_max]

        if idx_max != len(omega_fine):
            omega_idxm1 = omega_fine[idx_max - 1]
            t_idxm1 = t_fine[idx_max - 1]

            omega_idx0 = omega_peak
            t_idx0 = t_fine[idx_max]

            omega_idx1 = omega_fine[idx_max + 1]
            t_idx1 = t_fine[idx_max + 1]

            t_peak, omega_peak = parabolaExtrema(
                omega_idx0, omega_idxm1, omega_idx1, t_idx0, t_idxm1, t_idx1
            )

    if peak_pr:
        intrp = CubicSpline(t_fine, dyn_fine[:, 3])
        left = t_fine[-1] - 10
        right = t_fine[-1]
        t_peak = iterative_refinement(intrp.derivative(), [left, right], pr=True)

    t_roll, dyn_roll, omega_roll = ts, dyn, omega_eob
    res = np.c_[t_roll, dyn_roll, omega_roll]

    if peak_omega or peak_pr:
        t_start = max(t_peak - step_back, dyn_fine[0, 0])
        idx_fine_start = np.argmin(np.abs(t_roll - t_start))

        idx_close = idx_fine_start

        t_fine = t_roll[idx_close:]
        dyn_fine = np.c_[t_fine, dyn_roll[idx_close:]]
        omega_fine = omega_roll[idx_close:]

    t_roll, dyn_roll, omega_roll, idx_close = transition_dynamics_v2(
        ts, dyn, omega_eob, idx_close
    )

    t_fine = t_roll[idx_close:]
    dyn_fine = np.c_[t_fine, dyn_roll[idx_close:]]
    omega_fine = omega_roll[idx_close:]

    dynamics_low = np.c_[t_roll[:idx_close], dyn_roll[:idx_close]]
    omega_low = omega_roll[:idx_close]

    # Add LN to the array so that it is also interpolated onto the fine sampling rate dynamics
    dyn_fine = interpolate_dynamics(dyn_fine, peak_omega=t_peak, step_back=step_back)

    # Remove points where omega starts decreasing as we need it
    # to be monotonically increasing as it is used for interpolation

    omega_dyn = dyn_fine[:, 6]
    om_diff = np.diff(omega_dyn)
    idx_omdiff = np.where(om_diff < 0)[0]

    if idx_omdiff.size != 0:
        idx_final = idx_omdiff[0]
        dyn_fine = dyn_fine[: idx_final + 1]

    # Define the dynamics
    dynamics_fine = dyn_fine

    # Full dynamics array
    dynamics = np.vstack((dynamics_low, dynamics_fine))

    # Return EOB dynamics, LN vectors,  PN stuff
    return (dynamics_low, dynamics_fine, dynamics, idx_close)


def transition_dynamics_v2(
    ts: np.ndarray, dyn: np.ndarray, omega_eob: np.ndarray, idx_restart: int
):
    """
    Function to transition from a point dyn1,t1final to a point (dyn2,tfinal2)

    Args:

         ts (np.ndarray): Time array
         dyn (np.ndarray): Dynamics array  (r,phi,pr,pphi)
         omega_eob (np.ndarray): Orbital frequency array
         idx_restart (int): Index which separates the low sampling rate dynamics and the high sampling rate dynamics.

    Returns:
         (tuple): Time array, dynamics, orbital frequency and index splitting the low and high sampling rate dynamics
    """

    # If the closest point within step back is actually the last point, step back more
    dyn_low = dyn[:idx_restart, :]
    dyn_fine = dyn[idx_restart:, :]

    t_low = ts[:idx_restart]
    t_fine = ts[idx_restart:]

    t_low_last = t_low[-1]
    t_fine_init = t_fine[0]

    if t_fine_init - t_low_last > 2:
        dt_low_last = t_low[-1] - t_low[-2]
        dt_fine = t_fine[1] - t_fine[0]
        dt_fine = 0.1

        t = t_fine[0] - dt_fine
        t_new = []

        step_multiplier = 1.3
        dt = dt_fine

        while True:
            t_new.append(t)

            if step_multiplier * dt < dt_low_last:
                dt *= step_multiplier

            t -= dt

            if t < t_low_last:
                break

        t_new = t_new[::-1]

        window = 50
        while idx_restart < window:
            window -= 5

        t_middle = ts[idx_restart - window : idx_restart + window]
        dyn_window = dyn[idx_restart - window : idx_restart + window]

        dyn_interp = CubicSpline(t_middle, dyn_window[:, :])
        omega_interp = CubicSpline(
            t_middle, omega_eob[idx_restart - window : idx_restart + window]
        )

        dyn_middle = dyn_interp(t_new)
        omega_middle = omega_interp(t_new)

        # Separate low and high SR omega
        omega_low = omega_eob[:idx_restart]
        omega_fine = omega_eob[idx_restart:]

        time = np.concatenate((t_low, t_new[:-1], t_fine))
        dynamics = np.vstack((dyn_low, dyn_middle[:-1, :], dyn_fine))
        omega = np.concatenate((omega_low, omega_middle[:-1], omega_fine))

        idx_restart_v1 = (
            len(omega_low) + len(omega_middle) - 1
        )  # idx_restart -1  + window_length
    else:
        dynamics = dyn
        time = ts
        omega = omega_eob
        idx_restart_v1 = idx_restart

    return time, dynamics, omega, idx_restart_v1


def parabolaExtrema(
    ff0: float, ffm1: float, ff1: float, tt0: float, ttm1: float, tt1: float
):
    """
    Compute the extremum of a parabola from 3 points (idx-1, idx, idx+1).

    Args:
        ff0 (float): value of the function at the index idx
        ffm1 (float): value of the function at the index idx-1
        ff1 (float): value of the function at the index idx+1
        tt0 (float): value of the time array at the index idx
        ttm1 (float): value of the time array at the index idx-1
        tt1 (float): value of the time array at the index idx+1

    Returns:
        tuple: time of the extremum, and value of the function at the extremum
    """

    aa = (ffm1 * (tt0 - tt1) + ff0 * (tt1 - ttm1) + ff1 * (-tt0 + ttm1)) / (
        (tt0 - tt1) * (tt0 - ttm1) * (tt1 - ttm1)
    )
    bb = (
        ffm1 * (-tt0 * tt0 + tt1 * tt1)
        + ff1 * (tt0 * tt0 - ttm1 * ttm1)
        + ff0 * (-tt1 * tt1 + ttm1 * ttm1)
    ) / ((tt0 - tt1) * (tt0 - ttm1) * (tt1 - ttm1))
    cc = (
        ffm1 * tt0 * (tt0 - tt1) * tt1
        + ff0 * tt1 * (tt1 - ttm1) * ttm1
        + ff1 * tt0 * ttm1 * (-tt0 + ttm1)
    ) / ((tt0 - tt1) * (tt0 - ttm1) * (tt1 - ttm1))

    textremum = -bb / (2.0 * aa)
    ff_extremum = aa * textremum * textremum + bb * textremum + cc

    return textremum, ff_extremum


def compute_dynamics_quasiprecessing(
    omega_ref: float,
    omega_start: float,
    H: Hamiltonian,
    RR: Callable,
    m_1: float,
    m_2: float,
    chi_1: np.ndarray,
    chi_2: np.ndarray,
    params: EOBParams,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    step_back: float = 250,
    y_init=None,
    initial_conditions=None,
    initial_conditions_postadiabatic_type=None,
):
    """
    Compute the dynamics starting from omega_start, with spins
    defined at omega_ref.

    First, PN evolution equations are integrated (including backwards in time)
    to get spin and orbital angular momentum. From that we construct splines
    either in time or orbital frequency for the PN quantities. Given the splines
    we now integrate aligned-spin EOB dynamics where at every step the projections
    of the spins onto orbital angular momentum is computed via the splines.

    Args:
        omega_ref (float): Reference frequency
        omega_start (float): Starting frequency
        H (Hamiltonian): Hamiltonian to use
        RR (Callable): RR force to use
        m_1 (float): Mass of primary
        m_2 (float): Mass of secondary
        chi_1 (np.ndarray): Dimensionless spin of the primary
        chi_2 (np.ndarray): Dimensionless spin of the secondary
        params (EOBParams): Container of additional inputs
        rtol (float, optional): Relative tolerance for EOB integration. Defaults to 1e-12.
        atol (float, optional): Absolute tolerance for EOB integration. Defaults to 1e-12.
        step_back (float, optional): Amount of time to step back for fine interpolation. Defaults to 250.
        y_init (np.ndarray, optional): Initial condition vector (r,phi,pr,pphi).
        initial_conditions (str, optional): Type of initial conditions for the ODE evolution ('adiabatic' or 'postadiabatic').
        initial_conditions_postadiabatic_type (str, optional): Type of postadiabatic initial conditions for the ODE evolution ('analytic' or 'numeric').

    Returns:
        tuple: Aligned-spin EOB dynamics, PN time, PN dynamics, PN splines
    """

    # Step 1: Compute PN dynamics

    if initial_conditions == "postadiabatic":
        combined_t, combined_y = compute_quasiprecessing_PNdynamics_opt(
            omega_ref, 0.9 * omega_start, m_1, m_2, chi_1, chi_2
        )
    else:
        combined_t, combined_y = compute_quasiprecessing_PNdynamics_opt(
            omega_ref, omega_start, m_1, m_2, chi_1, chi_2
        )
    # Compute last value of the orbital frequency from the PN evolution (to be used for
    # the termination conditions of the non-precessing evolution)
    omegaPN_f = combined_y[:, -1][-1]

    # Step 2: Interpolate PN dynamics
    splines = build_splines_PN(combined_t, combined_y, m_1, m_2, omega_start)

    # Step 3: Evolve the non-precessing EOB evolution equations
    (dynamics_low, dynamics_fine, dynamics, idx_restart) = compute_dynamics_prec_opt(
        omega_ref,
        omega_start,
        omegaPN_f,
        H,
        RR,
        m_1,
        m_2,
        splines,
        combined_t,
        combined_y,
        params,
        rtol=rtol,
        atol=atol,
        step_back=step_back,
        y_init=y_init,
        initial_conditions=initial_conditions,
        initial_conditions_postadiabatic_type=initial_conditions_postadiabatic_type,
    )

    return (
        dynamics_low,
        dynamics_fine,
        combined_t,
        combined_y,
        splines,
        dynamics,
        idx_restart,
    )
