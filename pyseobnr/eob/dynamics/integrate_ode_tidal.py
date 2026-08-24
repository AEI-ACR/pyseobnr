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
from .integrate_ode import ODE_system_RHS_opt

step = odeiv2.pygsl_lite_odeiv2_step
_control = odeiv2.pygsl_lite_odeiv2_control
evolve = odeiv2.pygsl_lite_odeiv2_evolve


class control_y_new(_control):
    def __init__(self, eps_abs, eps_rel):
        a_y = 1
        a_dydt = 1
        _control.__init__(self, eps_abs, eps_rel, a_y, a_dydt, None)


'''
def augment_dynamics(dynamics, chi_1, chi_2, m_1, m_2, H):
    """Compute dynamical quantities we need for the waveform

    Args:
        dynamics (np,ndarray): The dynamics array: t,r,phi,pr,pphi
    """
    result = []
    p_c = np.zeros(2)

    for i, row in enumerate(dynamics):
        q = row[1:3]
        p = row[3:5]
        p_c[1] = p[1]
        # Evaluate a few things: H, omega,omega_circ

        dyn = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
        omega = dyn[3]
        H_val = dyn[4]

        omega_c = H.omega(q, p_c, chi_1, chi_2, m_1, m_2)

        result.append([H_val, omega, omega_c])
    result = np.array(result)
    return np.c_[dynamics, result]
'''


@jit(nopython=True)
def h_max(r):
    return 1


def compute_dynamics_opt_tidal(
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
    omega_stop_NR=0.0,
    omega_stop_resonance=1.0,
):
    """
    Integrate the aligned-spin tidal (SEOBNRv5THM) equations of motion.

    The EOB right-hand sides have the same structure as the point-particle case
    (Eq. (2) of arXiv:2112.06952; see rhs_aligned.pyx), with tidal contributions
    entering both the Hamiltonian and the radiation-reaction force. The tidal model
    is described in arXiv:2503.18934 and arXiv:2606.02690. Uses the GSL
    Prince-Dormand 8(9) (rk8pd) integrator.

    Args:
        omega0 (float): initial orbital frequency (geometric units, M=1)
        H (Hamiltonian): The Hamiltonian object to use
        RR (function): The RR force to use. Must have same signature as Hamiltonian
        chi_1 (float): z-component of the primary spin
        chi_2 (float): z-component of the secondary spin
        m_1 (float): mass of the primary (M=1 units)
        m_2 (float): mass of the secondary (M=1 units)
        rtol (float): relative tolerance of the integrator
        atol (float): absolute tolerance of the integrator
        backend (str): accepted for interface compatibility but ignored
            (integration always uses the GSL rk8pd stepper)
        params (EOBParams): container of the EOB and tidal parameters
        step_back (float): time (in M) to step back for the start of the fine dynamics
        max_step (float): accepted for interface compatibility but ignored
        min_step (float): accepted for interface compatibility but ignored
        y_init (np.array, optional): initial state [r, phi, pr, pphi]. If None it is
            computed from omega0 via the post-circular initial conditions.
        r_stop (float): minimum final separation for the dynamics; if negative a
            default of 1.4 is used
        omega_stop_NR (float): NR-calibrated merger angular frequency. Reserved for
            the NR-merger stopping condition (currently disabled in the loop).
        omega_stop_resonance (float): f-mode resonance angular frequency; sets the
            radius r_test = max(5, omega_stop_resonance ** (-2/3) * 1.1) below which
            the termination conditions are evaluated.

    Returns:
        np.array, np.array: coarse and fine augmented dynamics arrays, with columns
        (t, r, phi, pr, pphi, H, omega, omega_circ)
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
    omegas = []
    ts.append(0.0)
    res_gsl.append(y)

    r_test = max(
        [
            5,
            omega_stop_resonance ** (-2 / 3) * 1.1,
        ]
    )  # 1.1
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

        # Comute the RHS after the step is done

        r = y[0]
        # print(r)

        # Check if the proposed step is larger than the maximum timestep
        # h_mx = h_max(r)
        # h = h / h_mx

        # Check for outspiral due to resonance, can happen for highly spinning BNS
        deriv = ODE_system_RHS_opt(t, y, [H, RR, chi_1, chi_2, m_1, m_2, params])
        drdt = deriv[0]
        omega = deriv[1]
        dprdt = deriv[2]
        if omega < omega_previous:
            # Reached peak in frequency
            peak_omega = True

            res_gsl.pop()
            ts.pop()
            if dprdt > -1e-3:
                res_gsl.pop()
                ts.pop()
            break
        omega_previous = omega

        # Handle termination conditions
        if r <= r_test:
            # deriv = ODE_system_RHS_opt(t, y, [H, RR, chi_1, chi_2, m_1, m_2, params])
            # drdt = deriv[0]
            omega = deriv[1]
            dprdt = deriv[2]
            """
            h_small = np.max((0.01,2*np.pi/(2.*omega) / (1 + np.exp(-(r - 4) / 0.13))))
            if h > h_small:
                h = h_small
            """

            if omega < omega_previous:
                # Reached peak in frequency
                peak_omega = True
                break
            # if omega > omega_stop_resonance:
            #     # Tidal disruption
            #     peak_omega = True
            #     break
            if peak_omega and ((t - ts[-2]) < 0.1):
                # got stuck
                break
            if drdt > 0:
                # Outspiral?!
                break
            if dprdt > 0:
                # Outspiral?!
                peak_pr = True
                break
            if r <= r_stop:
                # Reached ISCO
                break
            q_vec = y[:2]
            p_circ[1] = y[-1]
            # omega_circ = H.omega(q_vec, p_circ, chi_1, chi_2, m_1, m_2)
            # if omega_circ > omega_stop_NR:
            #     # Reached NR fit merger frequency
            #     print('Termination because of NR merger frequency reached')
            #     break
            # if omega_circ > omega_stop_resonance:
            #     # Reached resonance frequency
            #     print('Termination because of resonance excitation')
            #     break
            if r < 3:
                q_vec = y[:2]
                p_circ[1] = y[-1]
                omega_circ = H.omega(q_vec, p_circ, chi_1, chi_2, m_1, m_2)
                if omega_circ > 1:
                    break
            omega_previous = omega

    ts = np.array(ts)
    dyn = np.array(res_gsl)

    # plt.figure(dpi=300)
    # plt.plot(ts[-10:],dyn[-10:,2],'.')
    # print(f"{dyn[-1,2]/dyn[-2,2]=}, {dyn[-3:,2]=}")
    # # plt.axhline(omega_stop_resonance**(-2/3))
    # plt.xlabel('t/M')
    # plt.ylabel('r/M')
    # plt.show()
    # plt.savefig('funny_result.png')

    if peak_omega:
        t_desired = ts[-1] - step_back - 50
    else:
        if peak_pr:
            # For highly spinning BNS, the system might develop eccentricity,
            # so it's good to step back a bit further to avoid frequency
            # derivatives being wrong in the final waveform
            step_back += 100
        t_desired = ts[-1] - step_back

    idx_close = np.argmin(np.abs(ts - t_desired))
    if ts[idx_close] > t_desired:
        idx_close -= 1

    # Gaurd against the case where when using PA dynamics,
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

    # plt.plot(dyn_fine[:1000,0], dyn_fine[:1000,1],'.',label='fine',alpha=.6)
    # plt.plot(ts[idx_close-2:idx_close+2], dyn[idx_close-2:idx_close+2,0],'.',label='coarse',alpha=.6)
    # plt.legend()
    # # plt.xlim(12800,12900)
    # plt.show()

    return dyn_coarse, dyn_fine
