# cython: language_level=3, profile=True, linetrace=True, binding=True

"""
Contains the functions needed for computing the "analytic" post-adiabatic dynamics
"""

cimport cython
from typing import Dict
import numpy as np
#import findiff
from scipy.integrate import solve_ivp, ode
from .initial_conditions_aligned_opt import computeIC_opt as computeIC


from pygsl_lite import  roots, errno
from scipy import optimize
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from scipy import integrate

from lalinference.imrtgr import nrutils

from .integrate_ode import compute_dynamics_opt as compute_dynamics
from .integrate_ode import augment_dynamics
from pyseobnr.eob.utils.containers cimport EOBParams
from pyseobnr.eob.hamiltonian.Hamiltonian_C cimport Hamiltonian_C
from pyseobnr.eob.waveform.waveform cimport RadiationReactionForce
from libc.math cimport log, sqrt, exp, abs,fabs, tgamma,sin,cos, tanh, sinh, asinh


from pyseobnr.eob.dynamics.postadiabatic_C cimport fin_diff_derivative, cumulative_integral, Kerr_ISCO, compute_adiabatic_solution


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
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
    double[::1] q,
    double[::1] p
):
    """
    Compute the value of pr at odd PA orders.
    This is done by evaluating Eq.(C7) of SEOBNRv5 theory doc
    at a given radial grid point. See  DCC:T2300060
    """
    q[0] = r
    p[0] = pr_sol
    p[1] = pphi
    cdef double dAdr,dBnpdr,dBnpadr,dxidr,dQdr,dQdprst,dHodddr,xi,omega,omega_circ
    cdef double A,Bnp,Bnpa,Q,Heven,Hodd,H_val
    cdef double aux_derivs[7]
    cdef double nu = params.p_params.nu
    aux_derivs[:] = H.auxderivs(q, p, chi_1, chi_2, m_1, m_2)

    dAdr = aux_derivs[0]
    dBnpdr = aux_derivs[1]
    dBnpadr = aux_derivs[2]
    dxidr = aux_derivs[3]
    dQdr = aux_derivs[4]
    dQdprst = aux_derivs[5]
    dHodddr = aux_derivs[6]

    H_val,xi,A,Bnp,Bnpa,Q,Heven,Hodd = H(q,p,chi_1,chi_2,m_1,m_2,verbose=True)
    omega = H.omega(q,p,chi_1,chi_2,m_1,m_2)

    params.dynamics.p_circ[1] = pphi
    omega_circ = H.omega(q,params.dynamics.p_circ,chi_1,chi_2,m_1,m_2)

    cdef (double,double) flux = RR.RR(q, p, omega,omega_circ,H_val,params)


    cdef double result =  xi/(2*(1+Bnp))*(flux[1]/dpphi_dr*2*nu*H_val*Heven/A - xi*dQdprst)


    return result


@cython.profile(True)
@cython.linetrace(True)
cpdef compute_pr(
    double[:] r,
    double[:] pr,
    double[:] pphi,
    Hamiltonian_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    double[::1] q,
    double[::1] p,
    EOBParams params=None,
):
    """
    Compute the value to pr at odd PA orders.
    This is done by evaluating  Eq(C7) of
    SEOBNRv5 theory doc at every radial grid point.
    See DCC:T2300060
    """
    cdef double[:] dpphi_dr = - fin_diff_derivative(r, pphi)
    cdef int i
    for i in range(r.shape[0]):

        pr[i]= pr_eqn(pr[i],r[i], pphi[i], dpphi_dr[i],
                H, RR, chi_1, chi_2, m_1, m_2,
                params,q,p)

    return pr

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
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
    double[::1] q,
    double[::1] p
):
    """
    Compute value of pphi at a given radial grid point.
    This is done by solving Eq.(C8) from SEOBNRv5
    theory doc. See DCC:T2300060.
    """
    q[0] = r
    p[0] = pr
    p[1] =  pphi_sol
    cdef double nu = params.p_params.nu
    params.dynamics.p_circ[1] =  pphi_sol

    cdef double dAdr,dBnpdr,dBnpadr,dxidr,dQdr,dQdprst,dHOdddr,xi,omega,omega_circ,result
    cdef double A,Bnp,Bnpa,Q,Heven,Hodd,H_val
    cdef double aux_derivs[7]

    aux_derivs[:] = H.auxderivs(q, p, chi_1, chi_2, m_1, m_2)

    dAdr = aux_derivs[0]
    dBnpdr = aux_derivs[1]
    dBnpadr = aux_derivs[2]
    dxidr = aux_derivs[3]
    dQdr = aux_derivs[4]
    dQdprst = aux_derivs[5]
    dHodddr = aux_derivs[6]

    H_val,xi,A,Bnp,Bnpa,Q,Heven,Hodd = H(q,p,chi_1,chi_2,m_1,m_2,verbose=True)
    omega = H.omega(q,p,chi_1,chi_2,m_1,m_2)

    params.dynamics.p_circ[1] = pphi_sol
    omega_circ = H.omega(q,params.dynamics.p_circ,chi_1,chi_2,m_1,m_2)

    cdef double drdt = A/(2*nu*H_val*Heven)*(2*pr/xi*(1+Bnp)+xi*dQdprst)

    cdef (double,double) flux = RR.RR(q, p, omega,omega_circ,H_val,params)

    cdef double pr2 = pr*pr
    cdef double xi2 = xi*xi
    cdef double xi3 = xi2*xi
    cdef double r2 = r*r
    cdef double r3 = r2*r
    cdef double ap = m_1*chi_1+m_2*chi_2
    cdef double ap2 = ap*ap

    cdef double C_2 = dAdr*(1/r2+ap2/r2*Bnpa)+A*(-2/r3-2*ap**2/r3*Bnpa+ap2/r2*dBnpadr)
    cdef double C_1 = dHodddr/pphi_sol*2*Heven
    cdef double C_0_pt1 = dAdr*(1+pr2/xi2*Bnp+pr2/xi2+Q) + A*(-2*pr2/xi3*Bnp*dxidr+pr2/xi2*dBnpdr-2*pr2/xi3*dxidr+dQdr)
    cdef double tmp = 2*Heven*nu*H_val/xi
    cdef double C_0_pt2 = dpr_dr*drdt*tmp - pr/pphi_sol*tmp*flux[1]
    cdef double C_0 = C_0_pt1+C_0_pt2

    D = C_1*C_1 - 4*C_2*C_0

    if D<0 and fabs(D)<1e-10:
        D = 0.0
    if D<0:
        raise ValueError
    result = (-C_1-sqrt(D))/(2*C_2)
    return result



@cython.profile(True)
@cython.linetrace(True)
cpdef compute_pphi(
    double[:] r,
    double[:] pr,
    double[:] pphi,
    Hamiltonian_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    double[::1] q,
    double[::1] p,
    EOBParams params=None
):
    """
    Compute value of pphi at even PA orders.
    This is done by solving Eq.(C8) from SEOBNRv5
    theory doc at every radial grid point.
    See DCC:T2300060
    """

    cdef double[:] dpr_dr = - fin_diff_derivative(r, pr)
    cdef int i
    for i in range(r.shape[0]):

        pphi[i]  = pphi_eqn(pphi[i],r[i], pr[i], dpr_dr[i], H, RR, chi_1, chi_2, m_1, m_2, params,q,p)

    return pphi

@cython.profile(True)
@cython.linetrace(True)
cpdef compute_postadiabatic_solution(
    double[:] r,
    double[:] pphi,
    Hamiltonian_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    double[::1] q,
    double[::1] p,
    double tol=1e-12,
    int order=8,
    EOBParams params=None,
):
    """
    Compute the post-adiabatic values
    of pr and pphi iteratively up to the given PA order.
    """
    pr = np.zeros(r.size)
    cdef int n
    cdef double tol_current
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

@cython.profile(True)
@cython.linetrace(True)
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
    r_ISCO, _ = Kerr_ISCO(chi_1, chi_2, m_1, m_2)
    cdef double r_final = max(10,r_final_prefactor * r_ISCO)
    cdef double r_switch_prefactor = 1.6
    cdef double r_switch = r_switch_prefactor * r_ISCO

    cdef double dr0 = 0.2
    cdef int r_size = int(np.ceil((r0 - r_final) / dr0))

    if r_size <= 4 or r0<=11.5:
        raise ValueError
    elif r_size < 10:
        r_size = 10

    r, _ = np.linspace(r0, r_final, num=r_size, endpoint=True, retstep=True)
    cdef double[::1] q = np.zeros(2)
    cdef double[::1] p = np.zeros(2)
    pphi = compute_adiabatic_solution(r, H, chi_1, chi_2, m_1, m_2,q,p, tol=tol,)

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

    dt_dr = np.zeros(r.shape[0])
    dphi_dr = np.zeros(r.shape[0])
    cdef int i
    cdef double dH_dpr,dH_dpphi,csi
    cdef double dyn[6]
    for i in range(r.shape[0]):
        q = np.array([r[i], 0])
        p = np.array([pr[i], pphi[i]])

        dyn[:] = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
        dH_dpr = dyn[2]
        dH_dpphi = dyn[3]


        csi = dyn[5]
        dt_dr[i] = 1 / (dH_dpr * csi)
        dphi_dr[i] = dH_dpphi / (dH_dpr * csi)


    t = cumulative_integral(r, dt_dr)

    phi = cumulative_integral(r, dphi_dr)

    postadiabatic_dynamics = np.c_[t, r, phi, pr, pphi]




    return postadiabatic_dynamics


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef compute_combined_dynamics(
    double omega0,
    Hamiltonian_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    double tol=1e-12,
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


    #print("PA_success: ", PA_success)
    #print("ode_y_init: ", ode_y_init)
    ode_dynamics_low, ode_dynamics_high = compute_dynamics(
        omega0,
        H,
        RR,
        chi_1,
        chi_2,
        m_1,
        m_2,
        rtol=1e-11,
        atol=1e-12,
        backend=backend,
        params=params,
        step_back=step_back,
        max_step=0.5,
        min_step=1.0e-9,
        y_init=ode_y_init,
        r_stop=r_stop
    )

    if PA_success is True:
        postadiabatic_dynamics = augment_dynamics(postadiabatic_dynamics,chi_1,chi_2,m_1,m_2,H)
        ode_dynamics_low[:, 0] += postadiabatic_dynamics[-1, 0]
        ode_dynamics_high[:, 0] += postadiabatic_dynamics[-1, 0]
        combined_dynamics = np.vstack(
            (postadiabatic_dynamics[:-1], ode_dynamics_low)
        )
    else:
        combined_dynamics = ode_dynamics_low


    return combined_dynamics, ode_dynamics_high
