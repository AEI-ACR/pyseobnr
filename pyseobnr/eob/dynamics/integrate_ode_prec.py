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
from .pn_evolution_opt import rhs_wrapper,compute_omega_orb,compute_quasiprecessing_PNdynamics_opt,build_splines_PN_opt
from .initial_conditions_aligned_precessing import computeIC_augm
from .integrate_ode import interpolate_dynamics
# Test cythonization of PN equations

step = odeiv2.pygsl_odeiv2_step
_control = odeiv2.pygsl_odeiv2_control
evolve = odeiv2.pygsl_odeiv2_evolve


def strictly_decreasing(L: list,) -> bool:
    return all(x > y for x, y in zip(L, L[1:]))

class control_y_new(_control):
    def __init__(self, eps_abs, eps_rel):
        a_y = 1
        a_dydt = 1
        _control.__init__(self, eps_abs, eps_rel, a_y, a_dydt, None)


def compute_dynamics_prec_opt(
    omega_start: float,
    omegaPN_f:float,
    H: Hamiltonian,
    RR: Callable,
    m_1: float,
    m_2: float,
    chi1L_spline,
    chi2L_spline,
    chi1LN_spline,
    chi2LN_spline,
    splines,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    params=None,
    step_back: float = 100,
    y_init=None,):
    # Step 3 : Evolve EOB  low SR dynamics
    # Update spin parameters and calibration coeffs (only dSO) at the start of EOB integration
    #print(f"omega_start={omega_start}")
    chi1_L = chi1L_spline(omega_start)
    chi2_L = chi2L_spline(omega_start)

    chi1_LN = chi1LN_spline(omega_start)
    chi2_LN = chi2LN_spline(omega_start)

    chi1_v = splines["chi1"](omega_start)
    chi2_v = splines["chi2"](omega_start)

    params.p_params.chi_1x, params.p_params.chi_1y, params.p_params.chi_1z = chi1_v
    params.p_params.chi_2x, params.p_params.chi_2y, params.p_params.chi_2z = chi2_v

    params.p_params.chi_1, params.p_params.chi_2 = chi1_LN, chi2_LN
    params.p_params.chi1_L, params.p_params.chi2_L = chi1_L, chi2_L

    params.p_params.update_spins(chi1_LN, chi2_LN)

    X1 = params.p_params.X_1
    X2 = params.p_params.X_2

    ap = chi1_LN * X1 + chi2_LN * X2
    am = chi1_LN * X1 - chi2_LN * X2

    dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)
    H.calibration_coeffs["dSO"] = dSO_new

    # Step 3.1: compute the initial conditions - uses the aligned-spin ID
    if y_init is None:

        r0, pphi0, pr0 = computeIC_augm(
            omega_start,
            H,
            RR,
            chi1_v,
            chi2_v,
            chi1_LN,
            chi2_LN,
            m_1,
            m_2,
            params=params,
        )
        y0 = np.array([r0, 0.0, pr0, pphi0])
    else:
        y0 = y_init.copy()
        r0 = y0[0]

    #print(f"y0={y0}")
    # Step 3.2: now integrate the dynamics

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
    omega_previous = omega_start
    res_gsl = []
    ts = []
    omegas = []
    ts.append(0.0)
    res_gsl.append(y)
    #ODE_system_RHS.omegas = [omega_start]
    #ODE_system_RHS.omegas_circ = [omega_start]
    #ODE_system_RHS.rs = [r0]
    omega_previous = omega_start
    r_previous = r0
    peak_omega = False
    peak_r = False
    X1 = params.p_params.X_1
    X2 = params.p_params.X_2
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
            dprdt = deriv[2]

            omega_circ = params.p_params.omega_circ

            if r <= 1.4:
                #print(f"r = {r}")
                break
            if np.isnan(omega) or np.isnan(drdt) or np.isnan(dprdt) or np.isnan(omega_circ):
                #print(f"r = {r}, omega = {omega}, dprdt = {dprdt}, drdt = {drdt}, omega_circ = {omega_circ}")
                break

            #def max_h(r):
            if r<2.5 and r>2 and h>0.08:
                h = 0.08
            else:
                if r<=2 and h>0.05:
                    h = 0.05

            if omega < omega_previous:
                peak_omega = True
                #print(f"r = {r}, omega = {omega}, omega_previous = {omega_previous}")
                break

            if r<3 and (drdt > 0 or dprdt > 0) :
                #print(f"r = {r}, drdt = {drdt} >0  , dprdt =  {dprdt} > 0")
                break

            if r < 3:
                if omega_circ > 1:
                    #print(f"r = {r}, omega_circ  = {omega_circ} > 1")
                    break
                if r > r_previous:
                    r_omega = True
                    print(f"r = {r}, omega = {omega}, omega_previous = {omega_previous}")
                    break
                    #else:
                #    print(f"No stopping condition: r = {r}, omega = {omega}, omega_previous = {omega_previous}, y = {y}")
                #if ODE_system_RHS.rs[-1]<r :
                #    print(f"r(i-1)<r(i) : r(i-1) = {ODE_system_RHS.rs[-1]}, r(i) = {r}")
                #    break


            omega_previous = omega
            omega_orb = omega
            #omega_orb = deriv[1]
            #if check_terminal(y, deriv):
            #    break

        else:
            omega_orb = compute_omega_orb(t, y, H, RR, m_1, m_2, params)

        if omega_orb > omegaPN_f:
            print(f"r = {r},   omega_orb = {omega_orb}, omegaPN_f = {omegaPN_f}")
            break

        # Update spin parameters and calibration coeffs (only dSO)
        params.p_params.omega = omega_orb

        omega_fs = params.p_params.omega
        omega_circ = params.p_params.omega_circ

        update = True
        if update:
            tmp = splines["everything"](omega_fs)
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

        if abs(ap)>1 or abs(am)>1:
            #print(f"r = {np.round(r,3)}, omega_fs = {np.round(omega_fs,4)}, chi1_LN = {np.round(chi1_LN,4)}, chi2_LN = {np.round(chi2_LN,4)}, ap = {np.round(ap,4)}, am = {np.round(am,4)}")
            break
        dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)
        H.calibration_coeffs["dSO"] = dSO_new
        #if r<2.:
        #    print(f"chi1_LN = {np.round(chi1_LN,6)}, chi2_LN = {np.round(chi2_LN,6)}, chi1_L = {np.round(chi1_L,6)}, chi2_L = {np.round(chi2_L,6)}")
        #    print(f"r = {np.round(r,6)}, ap ={np.round(ap,6)}, am = {np.round(am,6)}, dSO = {np.round(dSO_new,6)}, omega = {np.round(omega_orb,6)}, omega_c = {np.round(omega_c,6)}")

        #    print(f"======================================================================================================")
        #print(f"r = {r}, dSO = {dSO_new}, omega = {omega_orb}, omega_c = {omega_c}")

        params.p_params.chi_1 = chi1_LN
        params.p_params.chi_2 = chi2_LN

        params.p_params.chi1_L = chi1_L
        params.p_params.chi2_L = chi2_L

        #if r<1.42:
        #    print(f"r = {np.round(r,3)}, omega_fs = {np.round(omega_fs,4)}, pphi = {np.round(y[3],4)}")

        # Remember to update the derived spin quantities
        params.p_params.update_spins(params.p_params.chi_1, params.p_params.chi_2)
        #ODE_system_RHS.omegas.append(omega_fs)
        #ODE_system_RHS.omegas_circ.append(omega_c)
        #ODE_system_RHS.rs.append(r)

    ts = np.array(ts)
    dyn = np.array(res_gsl)

    res = np.c_[ts, dyn]


    # Now for the step back
    # We do this to ensure smooth variation in attachment as a function
    # of calibration parameters
    # Endpoint of adaptive integration
    t_stop = res[-1, 0]
    #print(f"t_stop={t_stop}")
    # The starting point of fine integration
    t_desired = t_stop - step_back
    #print(f"t_desired={t_desired}")
    idx_restart = np.argmin(np.abs(res[:, 0] - t_desired))
    # If the closest point within step back is actually the last point, step back more
    if idx_restart == len(res) - 1:
        idx_restart -= 1
    if res[idx_restart, 0] > t_desired:
        idx_restart -= 1

    #if np.isnan(res).any():
    #    print(f"res contains Nans!")
    #    print(f"res[-10:] = {res[-10:]}")

    dynamics_fine, dynamics_low  = project_spins_augment_dynamics_opt(
        m_1, m_2, H, res[:idx_restart, :], res[idx_restart:, :], splines, omegaPN_f
    )

    #if np.isnan(dynamics_fine).any():
    #    print(f"dynamics fine contains Nans! after projection")
    #if np.isnan(dynamics_low).any():
    #    print(f"dynamics low contains Nans! after projection")

    dynamics_fine = interpolate_dynamics(dynamics_fine)
    #print("After interpolation")
    #print(dynamics_fine[0],dynamics_fine[-1])
    dynamics = np.vstack((dynamics_low, dynamics_fine))


    # Return EOB dynamics, PN stuff and the splines
    return (
        dynamics_low,
        dynamics_fine,
        chi1L_spline,
        chi2L_spline,
        chi1LN_spline,
        chi2LN_spline,
        splines,
        dynamics,
    )


def compute_dynamics_quasiprecessing(
    omega0: float,
    omega_start: float,
    H: Hamiltonian,
    RR: Callable,
    m_1: float,
    m_2: float,
    chi_1: np.ndarray,
    chi_2: np.ndarray,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    backend: str = "dopri5",
    # params: Dict[Any, Any] = None,
    params=None,
    step_back: float = 100,
    max_step: float = 0.5,
    min_step: float = 1.0e-9,
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
        ODE_system_RHS (Callable): Which system of equations to use
        rtol (float, optional): Relative tolerance for EOB integration. Defaults to 1e-12.
        atol (float, optional): Absolute tolerance for EOB integration. Defaults to 1e-12.
        backend (str, optional): The backend to use for ODE integration. Defaults to "solve_ivp".
        params (Dict[Any,Any], optional): Dictionary of additional inputs. Defaults to None.
        step_back (float, optional): Amount of time to step back for fine integration. Defaults to 10.
        max_step (float, optional): Max step allowed for fine stepping. Defaults to 0.05.
        min_step (float, optional): Min step allowed. Defaults to 1.0e-9. Currently not used
        tp (str, optional): Whether to use time or orbital frequency splines. Defaults to "time".

    Raises:
        NotImplementedError: If a type of splines is not supported

    Returns:
        tuple: Aligned-spin EOB dynamics, PN time, PN dynamics, PN splines
    """

    # Step 1: Compute PN dynamics

    combined_t, combined_y = compute_quasiprecessing_PNdynamics_opt(
        omega0, omega_start, m_1, m_2, chi_1, chi_2
    )
    omegaPN_f = combined_y[:, -1][-1]
    # print(f"omegaPN_f = {omegaPN_f}")

    # a1 =params.p_params.a1
    # a2 =params.p_params.a2

    # Step 2: Interpolate PN dynamics
    (
        splines,
        chi_1,
        chi_2,
        chi1L_spline,
        chi2L_spline,
        chi1LN_spline,
        chi2LN_spline,
    ) = build_splines_PN_opt(combined_t, combined_y, m_1, m_2, omega_start)


    (
        dynamics_low,
        dynamics_fine,
        chi1L_spline,
        chi2L_spline,
        chi1LN_spline,
        chi2LN_spline,
        splines,
        dynamics,
    ) = compute_dynamics_prec_opt(
        omega_start,
        omegaPN_f,
        H,
        RR,
        m_1,
        m_2,
        chi1L_spline,
        chi2L_spline,
        chi1LN_spline,
        chi2LN_spline,
        splines,
        rtol=rtol,
        atol=atol,
        params=params,
        step_back=step_back,
        y_init=y_init,
    )

    return (
        dynamics_low,
        dynamics_fine,
        combined_t,
        combined_y,
        chi1L_spline,
        chi2L_spline,
        chi1LN_spline,
        chi2LN_spline,
        splines,
        dynamics,
    )


