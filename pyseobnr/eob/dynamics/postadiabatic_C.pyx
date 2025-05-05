# cython: language_level=3

"""
Contains the functions needed for computing the post-adiabatic dynamics as well as the combined dynamics
(post-adiabatic + final part of the inspiral evolved using the usual EOB dynamics)
"""

cimport cython

import numpy as np
cimport numpy as cnp

from libc.math cimport fabs, ceil, sqrt

from pygsl_lite import errno, roots
from scipy import optimize
from scipy.interpolate import InterpolatedUnivariateSpline

from ..hamiltonian.Hamiltonian_C cimport (
    Hamiltonian_C,
    Hamiltonian_C_dynamics_return_t,
    Hamiltonian_C_grad_return_t,
)
from ..utils.containers cimport EOBParams, qp_param_t
from ..waveform.waveform cimport RadiationReactionForce
from ..utils.nr_utils import bbh_final_spin_non_precessing_HBR2016

from .initial_conditions_aligned_opt import computeIC_opt as computeIC
from .integrate_ode import augment_dynamics, compute_dynamics_opt as compute_dynamics


fin_diff_coeffs_order_9 = np.ascontiguousarray(
    [
        [
            -761.0 / 280.0,
            8.0,
            -14.0,
            56.0 / 3.0,
            -35.0 / 2.0,
            56.0 / 5.0,
            -14.0 / 3.0,
            8.0 / 7.0,
            -1.0 / 8.0,
        ],
        [
            -1.0 / 8.0,
            -223.0 / 140.0,
            7.0 / 2.0,
            -7.0 / 2.0,
            35.0 / 12.0,
            -7.0 / 4.0,
            7.0 / 10.0,
            -1.0 / 6.0,
            1.0 / 56.0,
        ],
        [
            1.0 / 56.0,
            -2.0 / 7.0,
            -19.0 / 20.0,
            2.0,
            -5.0 / 4.0,
            2.0 / 3.0,
            -1.0 / 4.0,
            2.0 / 35.0,
            -1.0 / 168.0,
        ],
        [
            -1.0 / 168.0,
            1.0 / 14.0,
            -1.0 / 2.0,
            -9.0 / 20.0,
            5.0 / 4.0,
            -1.0 / 2.0,
            1.0 / 6.0,
            -1.0 / 28.0,
            1.0 / 280.0,
        ],
        [
            1.0 / 280.0,
            -4.0 / 105.0,
            1.0 / 5.0,
            -4.0 / 5.0,
            0,
            4.0 / 5.0,
            -1.0 / 5.0,
            4.0 / 105.0,
            -1.0 / 280.0,
        ],
        [
            -1.0 / 280.0,
            1.0 / 28.0,
            -1.0 / 6.0,
            1.0 / 2.0,
            -5.0 / 4.0,
            9.0 / 20.0,
            1.0 / 2.0,
            -1.0 / 14.0,
            1.0 / 168.0,
        ],
        [
            1.0 / 168.0,
            -2.0 / 35.0,
            1.0 / 4.0,
            -2.0 / 3.0,
            5.0 / 4.0,
            -2.0,
            19.0 / 20.0,
            2.0 / 7.0,
            -1.0 / 56.0,
        ],
        [
            -1.0 / 56.0,
            1.0 / 6.0,
            -7.0 / 10.0,
            7.0 / 4.0,
            -35.0 / 12.0,
            7.0 / 2.0,
            -7.0 / 2.0,
            223.0 / 140.0,
            1.0 / 8.0,
        ],
        [
            1.0 / 8.0,
            -8.0 / 7.0,
            14.0 / 3.0,
            -56.0 / 5.0,
            35.0 / 2.0,
            -56.0 / 3.0,
            14.0,
            -8.0,
            761.0 / 280.0,
        ],
    ]
)

# for convolutions
fin_diff_coeffs_order_9_reverse = np.ascontiguousarray(fin_diff_coeffs_order_9[:, ::-1])


interpolated_integral_order_3 = np.ascontiguousarray([
    [3./8., 19./24., -5./24., 1./24.],
    [-1./24., 13./24., 13./24., -1./24.],
    [1./24., -5./24., 19./24., 3./8.],
])

interpolated_integral_order_5 = np.ascontiguousarray([
    [95./288., 1427./1440., -133./240., 241./720., -173./1440., 3./160.],
    [-3./160., 637./1440., 511./720., -43./240., 77./1440., -11./1440.],
    [11./1440., -31./480., 401./720., 401./720., -31./480., 11./1440.],
    [-11./1440., 77./1440., -43./240., 511./720., 637./1440., -3./160.],
    [3./160., -173./1440., 241./720., -133./240., 1427./1440., 95./288.],
])

interpolated_integral_order_7 = np.ascontiguousarray(
    [
        [
            5257.0 / 17280,
            139849.0 / 120960,
            -(4511.0 / 4480),
            123133.0 / 120960,
            -(88547.0 / 120960),
            1537.0 / 4480,
            -(11351.0 / 120960),
            275.0 / 24192,
        ],
        [
            -(275.0 / 24192),
            5311.0 / 13440,
            11261.0 / 13440,
            -(44797.0 / 120960),
            2987.0 / 13440,
            -(1283.0 / 13440),
            2999.0 / 120960,
            -(13.0 / 4480),
        ],
        [
            13.0 / 4480,
            -(4183.0 / 120960),
            6403.0 / 13440,
            9077.0 / 13440,
            -(20227.0 / 120960),
            803.0 / 13440,
            -(191.0 / 13440),
            191.0 / 120960,
        ],
        [
            -(191.0 / 120960),
            1879.0 / 120960,
            -(353.0 / 4480),
            68323.0 / 120960,
            68323.0 / 120960,
            -(353.0 / 4480),
            1879.0 / 120960,
            -(191.0 / 120960),
        ],
        [
            191.0 / 120960,
            -(191.0 / 13440),
            803.0 / 13440,
            -(20227.0 / 120960),
            9077.0 / 13440,
            6403.0 / 13440,
            -(4183.0 / 120960),
            13.0 / 4480,
        ],
        [
            -(13.0 / 4480),
            2999.0 / 120960,
            -(1283.0 / 13440),
            2987.0 / 13440,
            -(44797.0 / 120960),
            11261.0 / 13440,
            5311.0 / 13440,
            -(275.0 / 24192),
        ],
        [
            275.0 / 24192,
            -(11351.0 / 120960),
            1537.0 / 4480,
            -(88547.0 / 120960),
            123133.0 / 120960,
            -(4511.0 / 4480),
            139849.0 / 120960,
            5257.0 / 17280,
        ],
    ]
)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef cnp.ndarray[double, ndim=1, mode="c"] fin_diff_derivative(
    cnp.ndarray[double, ndim=1] x,
    cnp.ndarray[double, ndim=1] y,
):
    """
    Compute 8th order finite difference derivative,
    assuming an equally spaced grid. Correctly uses
    asymmetric stencils at the end points.
    """

    cdef int size = x.shape[0]
    cdef cnp.ndarray[double, ndim=1, mode="c"] dy_dx = np.empty(size)
    cdef double h = fabs(x[1] - x[0])
    cdef int i, j

    current_y = y[:9]
    for j in range(4):
        dy_dx[j] = np.convolve(fin_diff_coeffs_order_9_reverse[j], current_y, "valid")[0]

    dy_dx[4:-4] = np.convolve(fin_diff_coeffs_order_9_reverse[4], y, "valid")

    j = 5
    current_y = y[-9:]  # means we cannot use "wraparound(False)"
    for i in range(size - 4, size):
        dy_dx[i] = np.convolve(fin_diff_coeffs_order_9_reverse[j], current_y, "valid")[0]
        j += 1

    # division at the end, we can multiply the convolution kernel with it.
    return dy_dx / h


cpdef (double, double) Kerr_ISCO(
    double chi1,
    double chi2,
    double m1,
    double m2,
):
    """
    Compute the Kerr ISCO radius and angular momentum
    from the remnant spin predicted by NR fits
    """
    a = bbh_final_spin_non_precessing_HBR2016(
            m1, m2, chi1, chi2, version="M3J4"
    )
    # Compute the ISCO radius for this spin
    cdef double Z_1 = 1 + (1 - a**2) ** (1.0 / 3) * ((1 + a) ** (1.0 / 3) + (1 - a) ** (1.0 / 3))
    cdef double Z_2 = sqrt(3 * a**2 + Z_1**2)
    cdef double r_ISCO = 3 + Z_2 - np.sign(a) * sqrt((3 - Z_1) * (3 + Z_1 + 2 * Z_2))

    # Compute the ISCO L for this spin
    cdef double L_ISCO = 2 / (3 * sqrt(3)) * (1 + 2 * sqrt(3 * r_ISCO) - 2)
    return r_ISCO, L_ISCO


cpdef Newtonian_j0(cnp.ndarray[double, ndim=1] r):
    """
    Newtonian expression for orbital angular momentum using
    consistent normalization.
    """
    return np.sqrt(r)


cpdef double j0_eqn(
    double j0_sol,
    double r,
    Hamiltonian_C H,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    qp_param_t q,
    qp_param_t p
):
    q[0] = r
    p[1] = j0_sol

    cdef Hamiltonian_C_grad_return_t dH_dq = H.grad(q, p, chi_1, chi_2, m_1, m_2)
    cdef double dH_dr = dH_dq[0]
    return dH_dr


cpdef cnp.ndarray[double, ndim=1, mode="c"] compute_adiabatic_solution(
    cnp.ndarray[double, ndim=1, mode="c"] r,
    Hamiltonian_C H,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    qp_param_t q,
    qp_param_t p,
    double tol=1e-12,
):
    """
    Compute the adiabatic solution for the orbital angular momentum.
    Corresponds to Eq.(2.5) of arXiv:2105.06983
    """
    cdef int i
    cdef cnp.ndarray[double, ndim=1, mode="c"] j0 = Newtonian_j0(r)

    for i in range(r.shape[0]):
        j0_solution = optimize.root(
            j0_eqn,
            j0[i],
            args=(
                r[i],
                H,
                chi_1,
                chi_2,
                m_1,
                m_2,
                q,
                p),
            tol=tol,
        )
        j0[i] = j0_solution.x

    return j0


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef double pr_eqn(
    double pr_sol,
    double r,
    double pphi,
    double dpphi_dr,
    Hamiltonian_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    EOBParams params,
    qp_param_t q,
    qp_param_t p
):
    """
    Evaluate the equation for odd-PA orders, corresponding to corrections to pr.
    See Eq. (2.6) of arXiv:2105.06983
    """
    q[0] = r
    p[0] = pr_sol
    p[1] = pphi
    cdef:
        double dH_dpr
        double H_val
        double csi
        double omega
        double omega_circ
        double result
    cdef Hamiltonian_C_dynamics_return_t dynamics = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)

    dH_dpr = dynamics[2]
    H_val = dynamics[4]
    csi = dynamics[5]

    params.dynamics.p_circ[1] = pphi
    omega_circ = H.omega(q, params.dynamics.p_circ, chi_1, chi_2, m_1, m_2)

    omega = dynamics[3]

    cdef (double, double) flux = RR.RR(q, p, omega, omega_circ, H_val, params)

    result = dpphi_dr * dH_dpr * csi - flux[1]
    return result


cpdef double pr_eqn_wrapper(pr_sol, args):
    return pr_eqn(pr_sol, *args)


cpdef compute_pr(
    cnp.ndarray[double, ndim=1, mode="c"] r,
    cnp.ndarray[double, ndim=1, mode="c"] pr,
    cnp.ndarray[double, ndim=1, mode="c"] pphi,
    Hamiltonian_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    qp_param_t q,
    qp_param_t p,
    double tol=1e-12,
    EOBParams params=None,
):
    """
    Compute the value to pr at odd PA orders.
    This is done by numerically solving Eq(2.6) of
    arXiv:2105.06983 at every radial grid point
    """

    cdef cnp.ndarray[double, ndim=1, mode="c"] dpphi_dr = -fin_diff_derivative(r, pphi)

    cdef int i, iter
    for i in range(r.shape[0]):
        if np.abs(pr[i]) < 1e-14:
            x0 = 0.0
            x1 = -3e-2
        else:
            x0 = pr[i]*0.95
            x1 = pr[i]*1.05

        mysys = roots.gsl_function(
            pr_eqn_wrapper,
            (
                r[i],
                pphi[i],
                dpphi_dr[i],
                H,
                RR,
                chi_1,
                chi_2,
                m_1,
                m_2,
                params,
                q,
                p
            )
        )
        solver = roots.brent(mysys)
        solver.set(x1, x0)
        for iter in range(100):
            status = solver.iterate()
            x_lo = solver.x_lower()
            x_up = solver.x_upper()
            status = roots.test_interval(x_lo, x_up, tol, tol)
            result = solver.root()
            if status == errno.GSL_SUCCESS:
                break
        pr[i] = result
    return pr


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef double pphi_eqn(
    double pphi_sol,
    double r,
    double pr,
    double dpr_dr,
    Hamiltonian_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    EOBParams params,
    qp_param_t q,
    qp_param_t p
):
    """
    Evaluate the equation for even-PA orders, corresponding to corrections to pphi.
    See Eq. (2.7) of arXiv:2105.06983
    """
    cdef:
        double dH_dr
        double dH_dpr
        double H_val
        double csi
        double omega
        double omega_circ
        double result

    q[0] = r
    p[0] = pr
    p[1] = pphi_sol

    cdef Hamiltonian_C_dynamics_return_t dynamics = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
    dH_dr = dynamics[0]
    dH_dpr = dynamics[2]
    H_val = dynamics[4]
    csi = dynamics[5]

    params.dynamics.p_circ[1] = pphi_sol
    omega_circ = H.omega(q, params.dynamics.p_circ, chi_1, chi_2, m_1, m_2)

    omega = dynamics[3]

    cdef (double, double) flux = RR.RR(q, p, omega, omega_circ, H_val, params)

    result = dpr_dr * dH_dpr + dH_dr - flux[0] / csi

    return result


cpdef double pphi_eqn_wrapper(pphi_sol, args):
    return pphi_eqn(pphi_sol, *args)


cpdef compute_pphi(
    cnp.ndarray[double, ndim=1, mode="c"] r,
    cnp.ndarray[double, ndim=1, mode="c"] pr,
    cnp.ndarray[double, ndim=1, mode="c"] pphi,
    Hamiltonian_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    qp_param_t q,
    qp_param_t p,
    double tol=1e-12,
    EOBParams params=None
):
    """
    Compute the correction to pphi at even PA orders.
    This is done by numericall solving Eq(2.7) of
    arXiv:2105.06983 at every radial grid point
    """

    cdef double[:] dpr_dr = -fin_diff_derivative(r, pr)
    cdef int i, iter
    for i in range(r.shape[0]):

        x0 = 0.95*pphi[i]
        x1 = 1.05*pphi[i]
        mysys = roots.gsl_function(
            pphi_eqn_wrapper,
            (
                r[i],
                pr[i],
                dpr_dr[i],
                H,
                RR,
                chi_1,
                chi_2,
                m_1,
                m_2,
                params,
                q,
                p
            )
        )
        solver = roots.brent(mysys)
        solver.set(x0, x1)
        for iter in range(100):
            status = solver.iterate()
            x_lo = solver.x_lower()
            x_up = solver.x_upper()
            status = roots.test_interval(x_lo, x_up, tol, tol)
            result = solver.root()
            if status == errno.GSL_SUCCESS:
                break
        pphi[i] = result
    return pphi


cpdef compute_postadiabatic_solution(
    cnp.ndarray[double, ndim=1, mode="c"] r,
    cnp.ndarray[double, ndim=1, mode="c"] pphi,
    Hamiltonian_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    qp_param_t q,
    qp_param_t p,
    double tol=1e-12,
    int order=8,
    EOBParams params=None,
):
    """
    Compute the postadiabatic solution iteratively
    """
    cdef cnp.ndarray[double, ndim=1, mode="c"] pr = np.zeros(r.shape[0])
    cdef int n, parity
    cdef double tol_current
    for n in range(1, order+1):
        tol_current = 1.0e-2/10.0**n
        parity = n % 2

        if n >= 7:
            tol_current=tol

        if parity:
            pr = compute_pr(
                r,
                pr,
                pphi,
                H,
                RR,
                chi_1,
                chi_2,
                m_1,
                m_2,
                q,
                p,
                tol=tol_current,
                params=params,
            )
        else:
            pphi = compute_pphi(
                r,
                pr,
                pphi,
                H,
                RR,
                chi_1,
                chi_2,
                m_1,
                m_2,
                q,
                p,
                tol=tol,
                params=params,
            )

    return pr, pphi


cpdef compute_postadiabatic_dynamics(
    double omega0,
    Hamiltonian_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    double tol=1e-12,
    EOBParams params=None,
    order=8
):
    """Compute the dynamics using PA procedure starting from omega0

    Args:
        omega0 (float): The starting *orbital* frequency in geomtric units
        H (Hamiltonian): The Hamiltonian object to use
        RR (function): The RR function to use
        chi_1 (float): z-component of the primary spin
        chi_2 (float): z-component of the secondary spin
        m_1 (float): mass of the primary
        m_2 (float): mass of the secondary

    Returns:
        np.array: The dynamics results, as (t,q,p)
    """
    cdef double r0
    r0, _, _ = computeIC(
        omega0,
        H,
        RR,
        chi_1,
        chi_2,
        m_1,
        m_2,
        params=params,
    )
    cdef double chi_eff = m_1*chi_1+m_2*chi_2
    cdef double nu = m_1*m_2/(m_1+m_2)**2
    cdef double r_final_prefactor = 2.7+chi_eff*(1-4.*nu)
    cdef double r_ISCO = Kerr_ISCO(chi_1, chi_2, m_1, m_2)[0]
    cdef double r_final = max(10.0, r_final_prefactor * r_ISCO)

    cdef double dr0 = 0.3
    cdef int r_size = int(ceil((r0 - r_final) / dr0))

    if r_size <= 4 or r0 <= 11.5:
        raise ValueError
    elif r_size < 10:
        r_size = 10

    cdef:
        cnp.ndarray[double, ndim=1, mode="c"] r
        cnp.ndarray[double, ndim=1, mode="c"] pr
        cnp.ndarray[double, ndim=1, mode="c"] pphi
        cnp.ndarray[double, ndim=1, mode="c"] dt_dr
        cnp.ndarray[double, ndim=1, mode="c"] dphi_dr

    r = np.linspace(r0, r_final, num=r_size, endpoint=True, retstep=False)

    cdef:
        qp_param_t q = (0, 0)
        qp_param_t p = (0, 0)

    pphi = compute_adiabatic_solution(
        r,
        H,
        chi_1,
        chi_2,
        m_1,
        m_2,
        q,
        p,
        tol=tol
    )

    pr, pphi = compute_postadiabatic_solution(
        r,
        pphi,
        H,
        RR,
        chi_1,
        chi_2,
        m_1,
        m_2,
        q,
        p,
        tol=tol,
        order=order,
        params=params,
    )

    dt_dr = np.empty(r.shape[0])
    dphi_dr = np.empty(r.shape[0])

    cdef:
        int i
        double dH_dpr_times_csi
        double dH_dpphi
        Hamiltonian_C_dynamics_return_t dyn

    q[1] = 0

    for i in range(r.shape[0]):
        q[0] = r[i]
        p[0] = pr[i]
        p[1] = pphi[i]

        dyn = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
        dH_dpr_times_csi = dyn[2] * dyn[5]
        dH_dpphi = dyn[3]

        dt_dr[i] = 1 / dH_dpr_times_csi  # (dH_dpr * csi)
        dphi_dr[i] = dH_dpphi / dH_dpr_times_csi  # (dH_dpr * csi)

    cdef cnp.ndarray[double, ndim=1, mode="c"] t = cumulative_integral(r, dt_dr)
    cdef cnp.ndarray[double, ndim=1, mode="c"] phi = cumulative_integral(r, dphi_dr)

    postadiabatic_dynamics = np.c_[t, r, phi, pr, pphi]

    return postadiabatic_dynamics


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef compute_combined_dynamics(
    double omega0,
    Hamiltonian_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    double tol=1e-12,
    double rtol_ode=1e-11,
    double atol_ode=1e-12,
    EOBParams params=None,
    double step_back=50,
    str backend="ode",
    int PA_order=8,
    double r_stop=-1
):
    """
    Compute the full inspiral dynamics by combining PA + ODE
    integration. If the PA procedure fails (e.g. because the
    inspiral is too short), the code falls back to ODE
    integration.
    """
    cdef:
        cnp.ndarray[double, ndim=2] postadiabatic_dynamics
        cnp.ndarray[double, ndim=2] ode_dynamics_low
        cnp.ndarray[double, ndim=2] ode_dynamics_high

    try:
        postadiabatic_dynamics = compute_postadiabatic_dynamics(
            omega0,
            H,
            RR,
            chi_1,
            chi_2,
            m_1,
            m_2,
            tol=tol,
            params=params,
            order=PA_order
        )
        PA_success = True
        ode_y_init = postadiabatic_dynamics[-1, 1:]
    except ValueError as e:
        PA_success = False
        ode_y_init = None

    ode_dynamics_low, ode_dynamics_high = compute_dynamics(
        omega0,
        H,
        RR,
        chi_1,
        chi_2,
        m_1,
        m_2,
        rtol=rtol_ode,
        atol=atol_ode,
        backend=backend,
        params=params,
        step_back=step_back,
        max_step=0.5,
        min_step=1.0e-9,
        y_init=ode_y_init,
        r_stop=r_stop
    )

    if PA_success is True:
        postadiabatic_dynamics = augment_dynamics(postadiabatic_dynamics, chi_1, chi_2, m_1, m_2, H)
        ode_dynamics_low[:, 0] += postadiabatic_dynamics[-1, 0]
        ode_dynamics_high[:, 0] += postadiabatic_dynamics[-1, 0]
        combined_dynamics = np.vstack(
            (postadiabatic_dynamics[:-1], ode_dynamics_low)
        )
    else:
        combined_dynamics = ode_dynamics_low

    return combined_dynamics, ode_dynamics_high


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef cnp.ndarray[double, ndim=1, mode="c"] cumulative_integral(
    cnp.ndarray[double, ndim=1] x,
    cnp.ndarray[double, ndim=1] y,
    int order=7,
):
    """
    Compute a cumulative integral numerically using sampled data
    to a given order in accuracy. Assumes an equally spaced grid
    """

    cdef int size = x.size
    cdef double h = x[1] - x[0]
    cdef cnp.ndarray[double, ndim=1, mode="c"] integral = np.empty(x.size)
    integral[0] = 0

    if order == 3:
        integral[1] = np.convolve(interpolated_integral_order_3[0][::-1], y[:4], "valid")[0]
        integral[2:-1] = np.convolve(interpolated_integral_order_3[1][::-1], y, "valid")
        integral[size - 1] = np.convolve(interpolated_integral_order_3[2][::-1], y[-4:], "valid")[0]

    elif order == 5:
        integral[1] = np.convolve(interpolated_integral_order_5[0][::-1], y[:6], "valid")[0]
        integral[2] = np.convolve(interpolated_integral_order_5[1][::-1], y[:6], "valid")[0]
        integral[3:-2] = np.convolve(interpolated_integral_order_5[2][::-1], y, "valid")
        integral[size-2] = np.convolve(interpolated_integral_order_5[3][::-1], y[-6:], "valid")[0]
        integral[size-1] = np.convolve(interpolated_integral_order_5[4][::-1], y[-6:], "valid")[0]

    elif order == 7:
        integral[1] = np.convolve(interpolated_integral_order_7[0][::-1], y[:8], "valid")[0]
        integral[2] = np.convolve(interpolated_integral_order_7[1][::-1], y[:8], "valid")[0]
        integral[3] = np.convolve(interpolated_integral_order_7[2][::-1], y[:8], "valid")[0]
        integral[4:-3] = np.convolve(interpolated_integral_order_7[3][::-1], y, "valid")
        integral[size-3] = np.convolve(interpolated_integral_order_7[4][::-1], y[-8:], "valid")[0]
        integral[size-2] = np.convolve(interpolated_integral_order_7[5][::-1], y[-8:], "valid")[0]
        integral[size-1] = np.convolve(interpolated_integral_order_7[6][::-1], y[-8:], "valid")[0]

    else:
        raise RuntimeError("Unsupported order")

    integral *= h
    return np.cumsum(integral)


cpdef univariate_spline_integral(
    x: np.array,
    y: np.array,
):
    y_x_interp = InterpolatedUnivariateSpline(x[::-1], y[::-1])
    y_x_integral = y_x_interp.antiderivative()(x[::-1])[::-1]
    integral = y_x_integral - y_x_integral[0]

    return integral


def compute_adiabatic_parameter(
    dynamics,
    H,
    chi_1,
    chi_2,
    m_1,
    m_2,
    params,
):
    """
    Compute the adiabatic parameter :math:`\\frac{\\dot{\\Omega}}{2 \\Omega^{2}}` .
    """
    omega = np.zeros(dynamics.shape[0])
    dr_dt = np.zeros(dynamics.shape[0])

    for i in range(dynamics.shape[0]):
        q = dynamics[i, 1:3]
        p = dynamics[i, 3:]

        dyn = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
        dH_dpr = dyn[2]

        csi = dyn[5]
        dr_dt[i] = dH_dpr * csi

        omega[i] = dyn[-1]

    cdef cnp.ndarray[double, ndim=1, mode="c"] domega_dr = fin_diff_derivative(dynamics[:, 1], omega)

    adiabatic_param = dr_dt * domega_dr / (2 * omega * omega)

    return adiabatic_param
