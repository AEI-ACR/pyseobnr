from typing import Callable

import numpy as np
import pygsl.errno as errno
import pygsl.odeiv2 as odeiv2
from numba import *
from numba import jit, types

from ..fits.fits_Hamiltonian import dSO as dSO_poly_fit
from ..hamiltonian import Hamiltonian
from ..utils.math_ops_opt import *
from ..utils.utils_precession_opt import project_spins_augment_dynamics_opt
from .initial_conditions_aligned_precessing import computeIC_augm
from .pn_evolution_opt import rhs_wrapper,compute_omega_orb,compute_quasiprecessing_PNdynamics_opt,build_splines_PN
from .integrate_ode import interpolate_dynamics
from ..utils.containers import EOBParams
# Test cythonization of PN equations

from scipy.interpolate import CubicSpline
from scipy.signal import argrelmin


step = odeiv2.pygsl_odeiv2_step
_control = odeiv2.pygsl_odeiv2_control
evolve = odeiv2.pygsl_odeiv2_evolve

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
    )->int :
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
        return 1

    if np.isnan(omega) or np.isnan(drdt) or np.isnan(dprdt) or np.isnan(omega_circ):
        return 2

    if omega < omega_previous:
        return 3

    if r < 3:

        if drdt >0:
            return 5

        if dprdt >0:
            return 6

        if omega_circ > 1:
            return 7

        if r > r_previous:
            return 8

    if omega > omegaPN_f:
        return 9

    return 0


def compute_dynamics_prec_opt(
    omega_start: float,
    omegaPN_f:float,
    H: Hamiltonian,
    RR: Callable,
    m_1: float,
    m_2: float,
    splines: dict,
    params: EOBParams,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    step_back: float = 250.,
    y_init=None,
    pa = False,
    ):
    """
    Function to perform a non-precessing EOB evolution with the spins modified
    at every timestep according to the values from the precessing-spin PN
    evolution.

    Args:
        omega_start (float): Starting orbital frequency
        omegaPN_f (float): Final orbital frequency from the precessing-spin PN evolution
        H (Hamiltonian): Hamiltonian class
        RR (Callable): RR force
        m_1 (float): Mass component of the primary
        m_2 (float): Mass component of the secondary
        splines (dict): Dictionary containing the splines in orbital frequency of the vector components of the spins, LN and L as
                        well as the spin projections onto LN and L
        params (EOBParams): Container of additional inputs
        rtol (float, optional): Relative tolerance for EOB integration. Defaults to 1e-12
        atol (float, optional): Absolute tolerance for EOB integration. Defaults to 1e-12
        step_back (float, optional): Amount of time to step back for fine interpolation. Defaults to 250.
        y_init (np.ndarray, optional): Initial condition vector (r,phi,pr,pphi)
        pa (bool,optional): If pa==False, not using PA, and thus using timestep roll-off in the dynamics, otherwise

    Returns:
        (tuple): Low and high sampling rate dynamics, unit Newtonian orbital angular momentum, assembled dynamics
                 and the index splitting the low and high sampling rate dynamics
    """

    # Step 3 : Evolve EOB dynamics

    # Step 3.1) Update spin parameters and calibration coeffs (only dSO) at the start of EOB integration
    tmp = splines["everything"](omega_start)
    chi1_LN_start = tmp[0]
    chi2_LN_start = tmp[1]
    chi1_L_start = tmp[2]
    chi2_L_start = tmp[3]
    chi1_v_start = tmp[4:7]
    chi2_v_start = tmp[7:10]

    params.p_params.chi_1x, params.p_params.chi_1y, params.p_params.chi_1z = chi1_v_start
    params.p_params.chi_2x, params.p_params.chi_2y, params.p_params.chi_2z = chi2_v_start

    params.p_params.chi_1, params.p_params.chi_2 = chi1_LN_start, chi2_LN_start
    params.p_params.chi1_L, params.p_params.chi2_L = chi1_L_start, chi2_L_start

    params.p_params.update_spins(chi1_LN_start, chi2_LN_start)

    X1 = params.p_params.X_1
    X2 = params.p_params.X_2

    ap_start = chi1_LN_start * X1 + chi2_LN_start * X2
    am_start = chi1_LN_start * X1 - chi2_LN_start * X2

    dSO_start = dSO_poly_fit(params.p_params.nu, ap_start, am_start)
    H.calibration_coeffs["dSO"] = dSO_start

    # Step 3.2: compute the initial conditions - uses the aligned-spin ID
    if y_init is None:
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
        h = 2 * np.pi / omega_start / 5
    else:
        h = 0.1

    res_gsl = []
    ts = []
    omegas = []
    ts.append(0.0)
    res_gsl.append(y)
    omegas.append(omega_start)

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

        # Append the last step
        res_gsl.append(y)
        ts.append(t)


        r = y[0]

        # Handle termination conditions
        if r <= 6:

            deriv = rhs_wrapper(t, y, [H, RR, m_1, m_2, params])
            drdt = deriv[0]
            omega = deriv[1]
            omegas.append(omega)
            dprdt = deriv[2]

            omega_circ = params.p_params.omega_circ

            # check termination conditions
            termination = check_terminal(r, omega, drdt, dprdt, omega_circ, omega_previous, r_previous, omegaPN_f)
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

        params.p_params.chi_1x, params.p_params.chi_1y, params.p_params.chi_1z = chi1_v
        params.p_params.chi_2x, params.p_params.chi_2y, params.p_params.chi_2z = chi2_v
        params.p_params.chi1_v[:] = chi1_v
        params.p_params.chi2_v[:] = chi2_v

        ap = chi1_LN * X1 + chi2_LN * X2
        am = chi1_LN * X1 - chi2_LN * X2

        # Check for possible extrapolation in the spin values
        if abs(ap)>1 or abs(am)>1:
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

    if pa==True:
        t_roll = ts
        dyn_roll = dyn
        omega_roll = omega_eob

        # Endpoint of adaptive integration
        t_stop = ts[-1]

        # Estimate of the starting point of fine integration
        t_desired = t_stop - step_back
        idx_restart = np.argmin(np.abs(ts - t_desired))

        # If the closest point within step back is actually the last point, step back more
        if idx_restart == len(ts) - 1:
            idx_restart -= 1
        if ts[idx_restart] > t_desired:
            idx_restart -= 1

    else:
        t_roll, dyn_roll, omega_roll, idx_restart = rolloff_timestep(ts,dyn,omega_eob,step_back)


    # Step 3.5: Iterative procedure to find the peak of omega or pr to have a more robust model
    #           under perturbations

    # Same refinement as in the aligned-spin model
    if peak_omega:
        t_desired = t_roll[-1] - step_back - 50
    else:
        t_desired = t_roll[-1] - step_back

    idx_close = np.argmin(np.abs(t_roll - t_desired))
    if t_roll[idx_close] > t_desired:
        idx_close -= 1

    # Guard against the case where when using PA dynamics,
    # there is less than step_back time between the start
    # of the ODE integration and the end of the dynamics
    # In that case make the fine dynamics be _all_ dynamics
    # except the 1st element
    if t_desired < t_roll[1]:
        idx_close = 1
        step_back = t_roll[-1]-t_roll[idx_close]
    dyn_coarse = np.c_[t_roll[:idx_close], dyn_roll[:idx_close]]
    dyn_fine = np.c_[t_roll[idx_close:], dyn_roll[idx_close:]]
    omega_low = omega_roll[:idx_close]
    omega_fine = omega_roll[idx_close:]

    t_peak = None
    if peak_omega:
        intrp = CubicSpline(dyn_fine[:, 0], omega_fine)
        left = dyn_fine[0, 0]
        right = dyn_fine[-1, 0]
        t_peak = iterative_refinement(intrp.derivative(), [left, right])

    if peak_pr:
        intrp = CubicSpline(dyn_fine[:, 0], dyn_fine[:, 3])
        left = dyn_fine[-1, 0] - 10
        right = dyn_fine[-1, 0]
        t_peak = iterative_refinement(intrp.derivative(), [left, right], pr = True)

    res = np.c_[t_roll, dyn_roll]


    # Step 3.6: Compute additional quantities needed to evaluate the waveform modes
    dyn_fine, dynamics_low, tmp_LN_low, tmp_LN_fine  = project_spins_augment_dynamics_opt(
        m_1, m_2, H, omega_low, omega_fine, res[:idx_close, :], res[idx_close:, :], splines
    )

    # Add LN to the array so that it is also interpolated onto the fine sampling rate dynamics
    dyn_fine = np.c_[dyn_fine,tmp_LN_fine]
    dyn_fine = interpolate_dynamics(
        dyn_fine, peak_omega=t_peak, step_back=step_back
    )

    # Take out tmp_LN_fine
    dynamics_fine = dyn_fine[:,:-3]
    tmp_LN_fine = dyn_fine[:,-3:]

    # Full dynamics array
    dynamics = np.vstack((dynamics_low, dynamics_fine))

    # Return EOB dynamics, LN vectors,  PN stuff and the splines
    return (
        dynamics_low,
        dynamics_fine,
        tmp_LN_low,
        tmp_LN_fine,
        dynamics,
        idx_restart
    )


def rolloff_timestep(ts: np.ndarray, dyn: np.ndarray, omega_eob: np.ndarray, step_back: float):
    """
       Function which smoothly transitions from large timesteps during the inspiral to
       fine timesteps at small separations

       Args:

            ts (np.ndarray): Time array
            dyn (np.ndarray): Dynamics array  (r,phi,pr,pphi)
            omega_eob (np.ndarray): Orbital frequency array
            step_back (float, optional): Amount of time to step back for fine interpolation. Defaults to 250

       Returns:
            (tuple): Time array, dynamics, orbital frequency and index splitting the low and high sampling rate dynamics
    """

    # Endpoint of adaptive integration
    t_stop = ts[-1]

    # Estimate of the starting point of fine integration
    t_desired = t_stop - step_back

    idx_restart = np.argmin(np.abs(ts - t_desired))

    # If the closest point within step back is actually the last point, step back more
    if idx_restart == len(ts) - 1:
        idx_restart -= 1
    if ts[idx_restart] > t_desired:
        idx_restart -= 1


    # Apply roll-off window to smooth the transition from large timesteps to small timesteps
    if idx_restart > 200 and len(dyn)>200:
        window_length = 200
    else:
        window_length = 100

    idx_0 = idx_restart +1  - window_length
    t_low = ts[:idx_0]
    t_middle = ts[idx_0:idx_restart+1]
    t_fine = ts[idx_restart:]

    dyn_low = dyn[:idx_0, :]
    dyn_middle = dyn[idx_0:idx_restart+1, :]
    dyn_fine = dyn[idx_restart:, :]

    t_low_last = t_middle[0]
    t_fine_init = t_fine[0]

    dt_low_last = t_middle[-1] - t_middle[-2]
    dt_fine_init = 0.1

    dt = dt_fine_init
    t = t_fine_init - dt
    t_new = []

    step_multiplier = 1.4#1.1
    while True:
        t_new.append(t)

        if step_multiplier * dt < dt_low_last:
            dt *= step_multiplier

        t -= dt
        if t < t_low_last:
            break

    t_new = t_new[::-1]
    #print(f"len(t_middle) = {len(t_middle)}, len(dyn_middle) = {len(dyn_middle[:,0])}")
    r_interp = CubicSpline(t_middle, dyn_middle[:, 0])
    phi_interp = CubicSpline(t_middle, dyn_middle[:, 1])
    pr_interp = CubicSpline(t_middle, dyn_middle[:, 2])
    pphi_interp = CubicSpline(t_middle, dyn_middle[:, 3])

    r_new = r_interp(t_new)
    phi_new = phi_interp(t_new)
    pr_new = pr_interp(t_new)
    pphi_new = pphi_interp(t_new)

    dyn_middle = np.vstack((r_new, phi_new, pr_new, pphi_new)).T

    omega_interp = CubicSpline(t_middle, omega_eob[idx_0:idx_restart+1])
    omega_middle = omega_interp(t_new)

    # Separate low and high SR omega
    #omega_low = omega_eob[:idx_restart]
    #omega_fine = omega_eob[idx_restart:]
    omega_low = omega_eob[:idx_0]
    omega_fine = omega_eob[idx_restart:]

    time = np.concatenate((t_low,t_new,t_fine))
    dynamics = np.vstack((dyn_low, dyn_middle, dyn_fine))
    omega = np.concatenate((omega_low,omega_middle,omega_fine))


    idx_restart_v1 =  len(omega_low)+len(omega_middle) #idx_restart -1  + window_length


    return time, dynamics, omega, idx_restart_v1



def iterative_refinement(f, interval, levels=2, dt_initial=0.1, pr = False):
    """Same function as in the non-precessing model (to be removed once merged with main)
    """
    left = interval[0]
    right = interval[1]
    for n in range(1, levels + 1):
        dt = dt_initial / (10 ** n)
        t_fine = np.arange(interval[0], interval[1], dt)
        deriv = np.abs(f(t_fine))

        mins = argrelmin(deriv, order=3)[0]
        if len(mins) > 0:
            result = t_fine[mins[0]]

            interval = max(result - 10 * dt, left), min(result + 10 * dt, right)

        else:
            if pr:
                return interval[-1]
            else:
                return (interval[0] + interval[-1]) / 2
    return result

def compute_dynamics_quasiprecessing(
    omega0: float,
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
):
    """
    Compute the dynamics starting from omega_start, with spins
    defined at omega0.

    First, PN evolution equations are integrated (including backwards in time)
    to get spin and orbital angular momentum. From that we construct splines
    either in time or orbital frequency for the PN quantities. Given the splines
    we now integrate aligned-spin EOB dynamics where at every step the projections
    of the spins onto orbital angular momentum is computed via the splines.

    Args:
        omega0 (float): Reference frequency
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
        y_init (np.ndarray, optional): Initial condition vector (r,phi,pr,pphi)

    Returns:
        tuple: Aligned-spin EOB dynamics, PN time, PN dynamics, PN splines
    """

    # Step 1: Compute PN dynamics
    combined_t, combined_y = compute_quasiprecessing_PNdynamics_opt(
        omega0, omega_start, m_1, m_2, chi_1, chi_2
    )

    # Compute last value of the orbital frequency from the PN evolution (to be used for
    # the termination conditions of the non-precessing evolution)
    omegaPN_f = combined_y[:, -1][-1]

    # Step 2: Interpolate PN dynamics
    splines = build_splines_PN(combined_t, combined_y, m_1, m_2, omega_start)

    # Step 3: Evolve the non-precessing EOB evolution equations
    (

        dynamics_low,
        dynamics_fine,
        tmp_LN_low,
        tmp_LN_fine,
        dynamics,
        idx_restart
    ) = compute_dynamics_prec_opt(
        omega_start,
        omegaPN_f,
        H,
        RR,
        m_1,
        m_2,
        splines,
        params,
        rtol=rtol,
        atol=atol,
        step_back=step_back,
    )

    return (
        dynamics_low,
        dynamics_fine,
        combined_t,
        combined_y,
        tmp_LN_low,
        tmp_LN_fine,
        splines,
        dynamics,
        idx_restart
    )
