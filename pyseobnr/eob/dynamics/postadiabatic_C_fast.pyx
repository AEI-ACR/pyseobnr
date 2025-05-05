# cython: language_level=3, profile=False, linetrace=False, binding=True

"""
Contains the functions needed for computing the "analytic" post-adiabatic dynamics
"""

cimport cython
import numpy as np
cimport numpy as cnp

from libc.math cimport sqrt, fabs, ceil

from ..utils.containers cimport EOBParams, qp_param_t
from ..hamiltonian.Hamiltonian_C cimport (
    Hamiltonian_C,
    Hamiltonian_C_auxderivs_return_t,
    Hamiltonian_C_dynamics_return_t,
    Hamiltonian_C_call_return_t
)
from ..waveform.waveform cimport RadiationReactionForce

from .initial_conditions_aligned_opt import computeIC_opt as computeIC
from .integrate_ode import compute_dynamics_opt as compute_dynamics
from .rhs_aligned cimport augment_dynamics
from .postadiabatic_C cimport fin_diff_derivative, cumulative_integral, Kerr_ISCO, compute_adiabatic_solution


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
    Compute the value of pr at odd PA orders.
    This is done by evaluating Eq.(C7) of SEOBNRv5 theory doc
    at a given radial grid point. See  DCC:T2300060
    """
    q[0] = r
    p[0] = pr_sol
    p[1] = pphi
    cdef:
        double omega
        double omega_circ

    cdef double nu = params.p_params.nu
    cdef Hamiltonian_C_auxderivs_return_t aux_derivs = H.auxderivs(q, p, chi_1, chi_2, m_1, m_2)
    cdef double dQdprst = aux_derivs[5]

    cdef Hamiltonian_C_call_return_t ret = H._call(q, p, chi_1, chi_2, m_1, m_2)
    omega = H.omega(q, p, chi_1, chi_2, m_1, m_2)

    cdef double H_val = ret[0]
    cdef double xi = ret[1]
    cdef double A = ret[2]
    cdef double Bnp = ret[3]
    cdef double Heven = ret[6]

    params.dynamics.p_circ[1] = pphi
    omega_circ = H.omega(q, params.dynamics.p_circ, chi_1, chi_2, m_1, m_2)

    cdef (double, double) flux = RR.RR(q, p, omega, omega_circ, H_val, params)

    cdef double result = xi/(2*(1+Bnp))*(flux[1]/dpphi_dr*2*nu*H_val*Heven/A - xi*dQdprst)
    return result


@cython.wraparound(False)
@cython.boundscheck(False)
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
    EOBParams params=None,
):
    """
    Compute the value to pr at odd PA orders.
    This is done by evaluating Eq(C7) of
    SEOBNRv5 theory doc at every radial grid point.
    See [SEOBNRv5HM-theory]_ .
    """
    cdef cnp.ndarray[double, ndim=1, mode="c"] dpphi_dr = -fin_diff_derivative(r, pphi)
    cdef int i
    for i in range(r.shape[0]):
        pr[i] = pr_eqn(
            pr[i],
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
    Compute value of pphi at a given radial grid point.
    This is done by solving Eq.(C8) from SEOBNRv5
    theory doc. See DCC:T2300060.
    """
    q[0] = r
    p[0] = pr
    p[1] = pphi_sol
    cdef double nu = params.p_params.nu
    params.dynamics.p_circ[1] = pphi_sol

    cdef:
        double dAdr,
        double dBnpdr
        double dBnpadr
        double dxidr
        double dQdr
        double dQdprst
        double dHodddr
        double xi
        double omega
        double omega_circ
    cdef:
        double A
        double Bnp
        double Bnpa
        double Q
        double Heven
        double H_val

    cdef Hamiltonian_C_auxderivs_return_t aux_derivs = H.auxderivs(q, p, chi_1, chi_2, m_1, m_2)

    dAdr = aux_derivs[0]
    dBnpdr = aux_derivs[1]
    dBnpadr = aux_derivs[2]
    dxidr = aux_derivs[3]
    dQdr = aux_derivs[4]
    dQdprst = aux_derivs[5]
    dHodddr = aux_derivs[6]

    cdef Hamiltonian_C_call_return_t ret = H._call(q, p, chi_1, chi_2, m_1, m_2)
    omega = H.omega(q, p, chi_1, chi_2, m_1, m_2)

    H_val = ret[0]
    xi = ret[1]
    A = ret[2]
    Bnp = ret[3]
    Bnpa = ret[4]
    Q = ret[5]
    Heven = ret[6]

    params.dynamics.p_circ[1] = pphi_sol
    omega_circ = H.omega(q, params.dynamics.p_circ, chi_1, chi_2, m_1, m_2)

    cdef double drdt = A/(2*nu*H_val*Heven)*(2*pr/xi*(1+Bnp)+xi*dQdprst)

    cdef (double, double) flux = RR.RR(q, p, omega, omega_circ, H_val, params)

    cdef double pr2 = pr*pr
    cdef double xi2 = xi*xi
    cdef double xi3 = xi2*xi
    cdef double r2 = r*r
    cdef double r3 = r2*r
    cdef double ap = m_1*chi_1+m_2*chi_2
    cdef double ap2 = ap*ap

    cdef double C_2 = dAdr*(1/r2+ap2/r2*Bnpa)+A*(-2/r3-2*ap**2/r3*Bnpa+ap2/r2*dBnpadr)
    cdef double C_1 = dHodddr/pphi_sol*2*Heven
    cdef double C_0_pt1 = (
        dAdr*(1+pr2/xi2*Bnp+pr2/xi2+Q)
        + A*(-2*pr2/xi3*Bnp*dxidr+pr2/xi2*dBnpdr-2*pr2/xi3*dxidr+dQdr)
    )
    cdef double tmp = 2*Heven*nu*H_val/xi
    cdef double C_0_pt2 = dpr_dr*drdt*tmp - pr/pphi_sol*tmp*flux[1]
    cdef double C_0 = C_0_pt1+C_0_pt2

    cdef double D = C_1*C_1 - 4*C_2*C_0

    if D < 0:
        if fabs(D) < 1e-10:
            D = 0.0
        else:
            raise ValueError

    cdef double result = (-C_1-sqrt(D))/(2*C_2)
    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef cnp.ndarray[double, ndim=1, mode="c"] compute_pphi(
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
    EOBParams params=None
):
    """
    Compute value of pphi at even PA orders.
    This is done by solving Eq.(C8) from SEOBNRv5
    theory doc at every radial grid point.
    See [SEOBNRv5HM-theory]_ .
    """

    cdef cnp.ndarray[double, ndim=1, mode="c"] dpr_dr = -fin_diff_derivative(r, pr)
    cdef int i
    for i in range(r.shape[0]):
        pphi[i] = pphi_eqn(
            pphi[i],
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
            p)

    return pphi


# cython: ctuples cannot contain python objects
# should be (cnp.ndarray[double, ndim=1, mode="c"], cnp.ndarray[double, ndim=1, mode="c"])
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
    Compute the post-adiabatic values
    of pr and pphi iteratively up to the given PA order.
    """
    cdef cnp.ndarray[double, ndim=1, mode="c"] pr = np.zeros(r.size)

    cdef int n, parity
    for n in range(1, order+1):
        parity = n % 2
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
                params=params,
            )

    return pr, pphi


cpdef cnp.ndarray[double, ndim=2] compute_postadiabatic_dynamics(
    double omega0,
    Hamiltonian_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    double tol=1e-12,
    EOBParams params=None,
    int order=8
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
    cdef double r_final = max(10, r_final_prefactor * r_ISCO)

    cdef double dr0 = 0.2
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

    pphi = compute_adiabatic_solution(r, H, chi_1, chi_2, m_1, m_2, q, p, tol=tol)

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
    cdef int i
    cdef:
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
    double r_stop = -1
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
    except Exception as e:

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
