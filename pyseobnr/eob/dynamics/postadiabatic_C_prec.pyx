# cython: language_level=3
"""
Contains the functions needed for computing the precessing post-adiabatic dynamics
"""

cimport cython
from typing import Dict
import numpy as np
import  quaternion
from libc.math cimport log, sqrt, exp, abs,fabs, tgamma,sin,cos, tanh, sinh, asinh

from .initial_conditions_aligned_opt import computeIC_opt as computeIC
from ..fits.fits_Hamiltonian import dSO as dSO_poly_fit

from pygsl_lite import  roots, errno
from scipy import optimize
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from scipy import integrate

from lalinference.imrtgr import nrutils

from pyseobnr.eob.utils.containers cimport EOBParams
from pyseobnr.eob.hamiltonian.Hamiltonian_v5PHM_C cimport Hamiltonian_v5PHM_C
from pyseobnr.eob.waveform.waveform cimport RadiationReactionForce

from .initial_conditions_aligned_precessing import computeIC_augm
from .pn_evolution_opt import compute_quasiprecessing_PNdynamics_opt
from .pn_evolution_opt import build_splines_PN
from .integrate_ode_prec import compute_dynamics_prec_opt
from pyseobnr.eob.utils.math_ops_opt import my_dot, my_norm

from .postadiabatic_C import Kerr_ISCO, Newtonian_j0, univariate_spline_integral

@cython.profile(True)
@cython.linetrace(True)
cpdef precessing_final_spin(
    double chi1_LN,
    double chi2_LN,
    chi1_v,
    chi2_v,
    LNhat,
    double m1,
    double m2,
):
    """
    Magnitude of the final spin computed from NR fits of precessing NR simulations. The sign of the final spin is computed
    from the non-precessing NR fits.

    Args:
        chi1_LN (double): Projection of the primary dimensionless spin onto LNhat
        chi2_LN (double): Projection of the secondary dimensionless spin onto LNhat
        chi1_v (double[:]): Primary dimensionless spin vector
        chi2_v (double[:]): Secondary dimensionless spin vector
        LNhat (double[:]): Newtonian orbital angular momentum vector
        m1 (double): Mass of the primary
        m2 (double): Mass of the secondary

    Returns:
        (double) Final spin
    """
    # Spin magnitudes
    cdef double a1 = my_norm(chi1_v)
    cdef double a2 = my_norm(chi2_v)

    cdef double tilt1, tilt2, angle, phi12

    # Tilt angles w.r.t L_N
    if a1 != 0:
        angle = chi1_LN / a1
        if angle < -1:
            tilt1 = np.pi
        elif angle > 1:
            tilt1 = 0.0
        else:
            tilt1 = np.arccos(np.clip(angle,-1,1))
    else:
        tilt1 = 0.0

    if a2 != 0:
        angle = chi2_LN / a2
        if angle < -1:
            tilt2 = np.pi
        elif angle > 1:
            tilt2 = 0.0
        else:
            tilt2 = np.arccos(np.clip(angle,-1,1))
    else:
        tilt2 = 0.0

    # Angle between the inplane spin components
    chi1_perp = chi1_v - chi1_LN * LNhat
    chi2_perp = chi2_v - chi2_LN * LNhat

    if a1 == 0 or a2 == 0:
        phi12 = np.pi / 2.0

    else:
        angle = np.dot(chi1_perp / a1, chi2_perp / a2)
        phi12 = np.arccos(np.clip(angle,-1,1))

    # Call non-precessing HBR fit to get the *sign* of the final spin
    cdef double final_spin_nonprecessing = nrutils.bbh_final_spin_non_precessing_HBR2016(
        m1, m2, chi1_LN, chi2_LN, version="M3J4"
    )
    final_spin_noprec = final_spin_nonprecessing
    cdef double sign_final_spin = +1

    cdef double norm_final_spin_nonprecessing = np.linalg.norm( final_spin_nonprecessing)
    if norm_final_spin_nonprecessing !=0:
      sign_final_spin = final_spin_nonprecessing / norm_final_spin_nonprecessing

    # Compute the magnitude of the final spin using the precessing fit
    final_spin = nrutils.bbh_final_spin_precessing_HBR2016(
        m1, m2, a1, a2, tilt1, tilt2, phi12, version="M3J4"
    )

    # Flip sign if needed
    final_spin *= sign_final_spin

    return final_spin


@cython.profile(True)
@cython.linetrace(True)
cpdef double j0_eqn(
    double j0_sol,
    double r,
    Hamiltonian_v5PHM_C H,
    double[:] chi1_v,
    double[:] chi2_v,
    double m_1,
    double m_2,
    double chi1_LN,
    double chi2_LN,
    double chi1_L,
    double chi2_L,
    EOBParams params,
    double[:] q,
    double[:] p,
):
    """
    Equation dH_dr =0 evaluated at prstar=0 used to obtain the adiabatic solution
    for the orbital angular momentum (pphi)


    Args:
        j0_sol (double): Guess of the solution
        r (double): Orbital separation
        H (Hamiltonian_v5PHM_C): Hamiltonian
        chi1_v (double[:]): Primary dimensionless spin vector
        chi2_v (double[:]): Secondary dimensionless spin vector
        m_1 (double): Mass of the primary
        m_2 (double): Mass of the secondary
        chi1_LN (double): Projection of the primary dimensionless spin onto LNhat
        chi2_LN (double): Projection of the secondary dimensionless spin onto LNhat
        chi1_L (double): Projection of the primary dimensionless spin onto Lhat
        chi2_L (double): Projection of the secondary dimensionless spin onto Lhat
        params (EOBParams): Container with useful EOB parameters
        q (double[:]): Position q=[r,phi]
        p (double[:]): Canonical momentum p=[pr,pphi]

    Returns:
        (double) Value of the orbital angular momemtum (pphi)
    """

    q[0] = r
    p[1]= j0_sol
    cdef double X1 = params.p_params.X_1
    cdef double X2 = params.p_params.X_2

    cdef double ap = chi1_LN * X1 + chi2_LN * X2
    cdef double am = chi1_LN * X1 - chi2_LN * X2

    cdef double dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)

    H.calibration_coeffs['dSO'] = dSO_new
    cdef (double,double,double,double) dH_dq = H.grad(
        q, p,
        chi1_v, chi2_v,
        m_1, m_2,
        chi1_LN, chi2_LN,
        chi1_L, chi2_L,
    )
    cdef double dH_dr = dH_dq[0]
    return dH_dr

cpdef compute_adiabatic_solution(
    double[:] r,
    double[:] omega,
    Hamiltonian_v5PHM_C H,
    dict splines,
    double m_1,
    double m_2,
    double[::1] q,
    double[::1] p,
    EOBParams params,
    double tol=1e-12,
):
    """
    Compute adiabatic solution for pphi by solving dH_dr =0 with prstar=0

    Args:
        r (double[:]): Radial grid on which to compute the adiabatic solution
        omega (double[:]): Grid of orbital frequencies used to compute the adiabatic solution
        H (Hamiltonian_v5PHM_C): Hamiltonian
        splines (dict): Dictionary containing the interpolated spin-precessing evolution
        m_1 (double): Mass of the primary
        m_2 (double): Mass of the secondary
        q (double[::1]): Position q=[r,phi]= np.zeros(2). These values are updated in the root finding routine.
        p (double[::1]): Canonical momentum p=[pr,pphi]=np.zeros(2). These values are updated in the root finding routine.
        params (EOBParams): Container with useful EOB parameters
        tol (double): Tolerance for the find root

    Returns:
        (np.array) Values of adiabatic pphi along the radial grid
    """

    cdef int i
    cdef double[:] j0 = Newtonian_j0(r)

    cdef double omega_start = splines["everything"].x[0]

    tmp = splines["everything"](omega)

    cdef double[:] chi1_LN = tmp[:,0]
    cdef double[:] chi2_LN = tmp[:,1]

    cdef double[:] chi1_L = tmp[:,2]
    cdef double[:] chi2_L = tmp[:,3]

    chi1_v = tmp[:,4:7]
    chi2_v = tmp[:,7:10]


    for i in range(r.shape[0]):

        if omega[i] < 0.9*omega_start:
          print(f"problem PA dynamics is extrapolating: omega ={omega[i]}< 0.9*omega_start {omega_start}")
        j0_solution = optimize.root(
            j0_eqn,
            j0[i],
            args=(r[i], H, chi1_v[i], chi2_v[i],
            chi1_LN[i], chi2_LN[i],
            chi1_L[i], chi2_L[i],
            m_1, m_2,params,q,p),
            tol=tol,
        )
        j0[i] = j0_solution.x

    return j0

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef double pr_eqn(
    pr_sol,
    double r,
    double pphi,
    double dpphi_dr,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    chi1_v,
    chi2_v,
    double chi1_LN,
    double chi2_LN,
    double chi1_L,
    double chi2_L,
    double m_1,
    double m_2,
    double omega_start,
    EOBParams params,
    double[::1] q,
    double[::1] p
):
    """
    Equation dpphi_dr * dH_dpr * csi - flux[1] = 0, where flux[1] is the azimuthal
    component of the radiation reaction force and csi is the tortoise coordinate
    transformation factor, to solve for prstar in the numerical postadiabatic routine.

    Args:
        pr_sol: Solution for prstar from the previous post-adiabatic order
        r (double): Orbital separation
        omega (double): Orbital frequency
        pphi (double): Orbital angular momentum
        dpphi_dr (double): Radial derivative of the orbital angular momentum
        H (Hamiltonian_v5PHM_C): Hamiltonian
        RR (RadiationReactionForce): Radiation Reaction Force
        chi1_v : Primary dimensionless spin vector
        chi2_v : Secondary dimensionless spin vector
        chi1_LN (double): Projection of the primary dimensionless spin onto LNhat
        chi2_LN (double): Projection of the secondary dimensionless spin onto LNhat
        chi1_L (double): Projection of the primary dimensionless spin onto Lhat
        chi2_L (double): Projection of the secondary dimensionless spin onto Lhat
        m_1 (double): Mass of the primary
        m_2 (double): Mass of the secondary
        omega_start (double): Initial orbital frequency
        params (EOBParams): Container with useful EOB parameters
        q (double[::1]): Position q=[r,phi]= np.zeros(2). These values are updated in the root finding routine.
        p (double[::1]): Canonical momentum p=[pr,pphi]=np.zeros(2). These values are updated in the root finding routine.

    Returns:
        (double) Value of prstar
    """
    q[0] = r
    p[0] = pr_sol[0]
    p[1] = pphi
    cdef double dH_dpr,H_val,csi,omega,omega_circ,result
    cdef double dynamics[6]

    cdef double X1 = params.p_params.X_1
    cdef double X2 = params.p_params.X_2

    cdef double ap = chi1_LN * X1 + chi2_LN * X2
    cdef double am = chi1_LN * X1 - chi2_LN * X2

    cdef double dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)

    H.calibration_coeffs['dSO'] = dSO_new

    dynamics[:] = H.dynamics(q, p, chi1_v, chi2_v, m_1, m_2,
        chi1_LN, chi2_LN,
        chi1_L, chi2_L,
    )

    dH_dpr = dynamics[2]
    omega = dynamics[3]
    H_val = dynamics[4]
    csi = dynamics[5]

    params.dynamics.p_circ[1] = pphi
    omega_circ = H.omega(q,params.dynamics.p_circ,chi1_v,chi2_v,m_1,m_2,chi1_LN, chi2_LN,
        chi1_L, chi2_L
    )

    if omega < 0.9*omega_start:
      print(f"problem PA dynamics is extrapolating: omega ={omega}< 0.9*omega_start {omega_start}")

    params.p_params.update_spins(chi1_LN, chi2_LN)
    cdef (double,double) flux = RR.RR(q, p, omega, omega_circ, H_val, params)

    params.p_params.omega_circ = omega_circ
    params.p_params.omega = omega
    params.p_params.H_val = H_val

    result = dpphi_dr * dH_dpr * csi - flux[1]

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef double pr_eqn_analytic(
    pr_sol,
    double r,
    double pphi,
    double dpphi_dr,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    chi1_v,
    chi2_v,
    double chi1_LN,
    double chi2_LN,
    double chi1_L,
    double chi2_L,
    double m_1,
    double m_2,
    double omega_start,
    EOBParams params,
    double[::1] q,
    double[::1] p
):
    """
    Equation dpphi_dr * dH_dpr * csi - flux[1] = 0, where flux[1] is the azimuthal
    component of the radiation reaction force and csi is the tortoise coordinate
    transformation factor, to solve for prstar in the analytic postadiabatic routine.
    This function provides the same result as the function "pr_eqn", but instead of
    using a root finding routine it computes the roots analytically using the
    potentials of the Hamiltonian.

    Args:
        pr_sol: Solution for prstar from the previous post-adiabatic order
        r (double): Orbital separation
        pphi (double): Orbital angular momentum
        dpphi_dr (double): Radial derivative of the orbital angular momentum
        H (Hamiltonian_v5PHM_C): Hamiltonian
        RR (RadiationReactionForce): Radiation Reaction Force
        chi1_v : Primary dimensionless spin vector
        chi2_v : Secondary dimensionless spin vector
        chi1_LN (double): Projection of the primary dimensionless spin onto LNhat
        chi2_LN (double): Projection of the secondary dimensionless spin onto LNhat
        chi1_L (double): Projection of the primary dimensionless spin onto Lhat
        chi2_L (double): Projection of the secondary dimensionless spin onto Lhat
        m_1 (double): Mass of the primary
        m_2 (double): Mass of the secondary
        omega_start (double): Initial orbital frequency
        params (EOBParams): Container with useful EOB parameters
        q (double[::1]): Position q=[r,phi]= np.zeros(2). These values are updated in the root finding routine.
        p (double[::1]): Canonical momentum p=[pr,pphi]=np.zeros(2). These values are updated in the root finding routine.

    Returns:
        (double) Value of prstar
    """

    q[0] = r
    p[0] = pr_sol
    p[1] = pphi

    cdef double nu = params.p_params.nu

    cdef double X1 = params.p_params.X_1
    cdef double X2 = params.p_params.X_2

    cdef double ap = chi1_LN * X1 + chi2_LN * X2
    cdef double am = chi1_LN * X1 - chi2_LN * X2

    cdef double dSO_new = dSO_poly_fit(nu, ap, am)
    H.calibration_coeffs['dSO'] = dSO_new

    cdef double A,Bnp,Bnpa,Q,Heven,Hodd,H_val

    cdef double xi,omega,omega_circ

    cdef double dAdr,dBnpdr,dBnpadr,dxidr,dQdr,dQdprst,dHodddr,dBpdr
    cdef double aux_derivs[9]
    aux_derivs[:] = H.auxderivs(
        q, p,
        chi1_v, chi2_v,
        m_1, m_2,
        chi1_LN, chi2_LN,
        chi1_L, chi2_L,
    )

    dAdr = aux_derivs[0]
    dBnpdr = aux_derivs[1]
    dBnpadr = aux_derivs[2]
    dxidr = aux_derivs[3]
    dQdr = aux_derivs[4]
    dQdprst = aux_derivs[5]
    dHodddr = aux_derivs[6]
    dBpdr = aux_derivs[7]
    dHevendr = aux_derivs[8]

    H_val, xi, A, Bnp, Bnpa, Q, Heven, Hodd, Bp = H(
        q, p,
        chi1_v, chi2_v,
        m_1, m_2,
        chi1_LN, chi2_LN,
        chi1_L, chi2_L,
    )

    omega = H.omega(
        q, p,
        chi1_v, chi2_v,
        m_1, m_2,
        chi1_LN, chi2_LN,
        chi1_L, chi2_L,
    )

    if omega < 0.9*omega_start:
      print(f"problem PA dynamics is extrapolating: omega ={omega}< 0.9*omega_start {omega_start}")

    params.dynamics.p_circ[1] = pphi
    omega_circ = H.omega(
        q, params.dynamics.p_circ,
        chi1_v, chi2_v,
        m_1, m_2,
        chi1_LN, chi2_LN,
        chi1_L, chi2_L,
    )
    params.p_params.update_spins(chi1_LN, chi2_LN)

    params.p_params.omega_circ = omega_circ
    params.p_params.omega = omega
    params.p_params.H_val = H_val

    cdef (double,double) flux = RR.RR(q, p, omega, omega_circ, H_val, params)

    cdef double result = (
        xi / (2 * (1 + Bnp)) * (
            flux[1] / dpphi_dr * 2 * nu * H_val * Heven / A - xi * dQdprst
        )
    )

    return result


@cython.profile(True)
@cython.linetrace(True)
cpdef compute_pr(
    double[:] r,
    double[:] pr,
    double[:] pphi,
    double[:] omega,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    dict splines,
    double m_1,
    double m_2,
    double omega_start,
    double[::1] q,
    double[::1] p,
    double tol=1e-12,
    str postadiabatic_type="analytic",
    EOBParams params=None,
):
    """
    Compute prstar at a given postadiabatic order by solving the equation
    dpphi_dr * dH_dpr * csi - flux[1] = 0, where flux[1] is the azimuthal component
    of the radiation reaction force and csi is the tortoise coordinate transformation
    factor, for prstar. This can be done either by root finding (postadiabatic_type="numeric")
    or by finding an analytic expression for the root using the potentials of the Hamiltonian
    (postadiabatic_type="analytic").

    Args:
        r (double[:]): Radial grid on which to compute prstar.
        pr (double[:]): Prstar values of the previous postadiabatic order.
        pphi (double[:]): Pphi values of the previous postadiabatic order.
        omega (double[:]): Orbital frequency values of the previous postadiabatic order.
        H (Hamiltonian_v5PHM_C): Hamiltonian.
        RR (RadiationReactionForce): Radiation Reaction Force.
        splines (dict): Dictionary containing the interpolated spin-precessing evolution.
        m_1 (double): Mass of the primary.
        m_2 (double): Mass of the secondary.
        omega_start (double): Initial orbital frequency.
        q (double[::1]): Position q=[r,phi]= np.zeros(2). These values are updated in the root finding routine.
        p (double[::1]): Canonical momentum p=[pr,pphi]=np.zeros(2). These values are updated in the root finding routine.
        tol (double): Tolerance for the root finding routine in case postadiabatic_type="numeric" is used.
        postadiabatic_type (str): If "analytic" find root analytically, otherwise numericall by a root finding method.
        params (EOBParams): Container with useful EOB parameters.

    Returns:
        (np.array) Value of prstar at a given postadiabatic order for the whole radial grid.
    """

    cdef double[:] dpphi_dr = spline_derivative(r, pphi)


    cdef int i

    tmp = splines["everything"](omega)
    chi1_v = tmp[:,4:7]
    chi2_v = tmp[:,7:10]

    cdef double[:] chi1_LN = tmp[:,0]
    cdef double[:] chi2_LN = tmp[:,1]

    cdef double[:] chi1_L = tmp[:,2]
    cdef double[:] chi2_L = tmp[:,3]

    for i in range(r.shape[0]):
        if np.abs(pr[i])<1e-14:
            x0 = 0.0
            x1 = -3e-2
        else:
            x0 = pr[i]*0.7
            x1 = pr[i]*1.3

        if postadiabatic_type == "analytic":
            pr_val = pr_eqn_analytic(
                pr[i], r[i], pphi[i], dpphi_dr[i],
                H, RR,
                chi1_v[i], chi2_v[i],
                chi1_LN[i], chi2_LN[i],
                chi1_L[i], chi2_L[i],
                m_1, m_2,
                omega_start,
                params,
                q, p
            )
        else:
            pr_solution = optimize.root(
                pr_eqn,
                pr[i],
                args=(
                    r[i], pphi[i], dpphi_dr[i],
                    H, RR, chi1_v[i], chi2_v[i],
                    chi1_LN[i],chi2_LN[i],
                    chi1_L[i],chi2_L[i],
                    m_1, m_2,
                    omega_start,
                    params,q,p
                ),
                tol=tol,

            )
            pr_val = pr_solution.x
        pr[i] = pr_val

        omega[i] = params.p_params.omega

        if omega[i] < 0.9*omega_start:
          print(f"problem PA dynamics is extrapolating: omega ={omega[i]}< 0.9*omega_start {omega_start}")


    return pr, omega

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef double pphi_eqn(
    pphi_sol,
    double r,
    double pr,
    double dpr_dr,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    chi1_v,
    chi2_v,
    double chi1_LN,
    double chi2_LN,
    double chi1_L,
    double chi2_L,
    double m_1,
    double m_2,
    double omega_start,
    EOBParams params,
    double[::1] q,
    double[::1] p
):
    """
    Equation dpr_dr * dH_dpr + dH_dr - flux[0] / csi, where flux[0] is the radial
    component of the radiation reaction force and csi is the tortoise coordinate
    transformation factor, to solve for ppphi in the numerical postadiabatic routine.

    Args:
        pphi_sol: Solution for pphi from the previous post-adiabatic order
        r (double): Orbital separation
        pr (double): Tortoise radial component of the momentum (prstar)
        dpr_dr (double): Radial derivative of prstar
        H (Hamiltonian_v5PHM_C): Hamiltonian
        RR (RadiationReactionForce): Radiation Reaction Force
        chi1_v : Primary dimensionless spin vector
        chi2_v : Secondary dimensionless spin vector
        chi1_LN (double): Projection of the primary dimensionless spin onto LNhat
        chi2_LN (double): Projection of the secondary dimensionless spin onto LNhat
        chi1_L (double): Projection of the primary dimensionless spin onto Lhat
        chi2_L (double): Projection of the secondary dimensionless spin onto Lhat
        m_1 (double): Mass of the primary
        m_2 (double): Mass of the secondary
        omega_start (double): Initial orbital frequency
        params (EOBParams): Container with useful EOB parameters
        q (double[::1]): Position q=[r,phi]= np.zeros(2). These values are updated in the root finding routine.
        p (double[::1]): Canonical momentum p=[pr,pphi]=np.zeros(2). These values are updated in the root finding routine.

    Returns:
        (double) Value of pphi
    """
    cdef double dH_dr,dH_dpr,H_val,csi,omega,omega_circ,result
    q[0] = r
    p[0] = pr
    p[1] =  pphi_sol[0]

    cdef double X1 = params.p_params.X_1
    cdef double X2 = params.p_params.X_2

    cdef double ap = chi1_LN * X1 + chi2_LN * X2
    cdef double am = chi1_LN * X1 - chi2_LN * X2

    cdef double dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)

    H.calibration_coeffs['dSO'] = dSO_new
    cdef double dynamics[6]

    dynamics[:] = H.dynamics(q,p, chi1_v, chi2_v, m_1, m_2,chi1_LN,chi2_LN,chi1_L,chi2_L)
    dH_dr = dynamics[0]
    dH_dpr = dynamics[2]

    omega = dynamics[3]
    H_val = dynamics[4]
    csi = dynamics[5]

    params.dynamics.p_circ[1] =  pphi_sol[0]
    omega_circ = H.omega(q,params.dynamics.p_circ,chi1_v,chi2_v,m_1,m_2,chi1_LN, chi2_LN, chi1_L, chi2_L)

    if omega < 0.9*omega_start:
      print(f"problem PA dynamics is extrapolating: omega ={omega}< 0.9*omega_start {omega_start}")

    params.p_params.update_spins(chi1_LN, chi2_LN)

    cdef (double,double) flux = RR.RR(q, p, omega,omega_circ,H_val,params)

    params.p_params.omega_circ = omega_circ
    params.p_params.omega = omega
    params.p_params.H_val = H_val

    result = dpr_dr * dH_dpr + dH_dr - flux[0] / csi

    return result



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef double pphi_eqn_analytic(
    double pphi_sol,
    double r,
    double pr,
    double dpr_dr,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    chi1_v,
    chi2_v,
    double chi1_LN,
    double chi2_LN,
    double chi1_L,
    double chi2_L,
    double m_1,
    double m_2,
    double omega_start,
    EOBParams params,
    double[::1] q,
    double[::1] p
  ):
    """
    Equation dpr_dr * dH_dpr + dH_dr - flux[0] / csi, where flux[0] is the radial
    component of the radiation reaction force and csi is the tortoise coordinate
    transformation factor, to solve for pphi in the analytic postadiabatic routine.
    This function provides the same result as the function "pphi_eqn", but instead of
    using a root finding routine it computes the roots analytically using the
    potentials of the Hamiltonian.

    Args:
        pphi_sol: Solution for pphi from the previous post-adiabatic order
        r (double): Orbital separation
        pr (double): Tortoise radial component of the momentum (prstar)
        dpr_dr (double): Radial derivative of prstar
        H (Hamiltonian_v5PHM_C): Hamiltonian
        RR (RadiationReactionForce): Radiation Reaction Force
        chi1_v : Primary dimensionless spin vector
        chi2_v : Secondary dimensionless spin vector
        chi1_LN (double): Projection of the primary dimensionless spin onto LNhat
        chi2_LN (double): Projection of the secondary dimensionless spin onto LNhat
        chi1_L (double): Projection of the primary dimensionless spin onto Lhat
        chi2_L (double): Projection of the secondary dimensionless spin onto Lhat
        m_1 (double): Mass of the primary
        m_2 (double): Mass of the secondary
        omega_start (double): Initial orbital frequency
        params (EOBParams): Container with useful EOB parameters
        q (double[::1]): Position q=[r,phi]= np.zeros(2). These values are updated in the root finding routine.
        p (double[::1]): Canonical momentum p=[pr,pphi]=np.zeros(2). These values are updated in the root finding routine.

    Returns:
        (double) Value of pphi
    """
    q[0] = r
    p[0] = pr
    p[1] =  pphi_sol
    cdef double nu = params.p_params.nu
    params.dynamics.p_circ[1] = pphi_sol

    cdef double X1 = params.p_params.X_1
    cdef double X2 = params.p_params.X_2

    cdef double ap = chi1_LN * X1 + chi2_LN * X2
    cdef double am = chi1_LN * X1 - chi2_LN * X2

    cdef double dSO_new = dSO_poly_fit(nu, ap, am)
    H.calibration_coeffs['dSO'] = dSO_new

    cdef double dAdr,dBnpdr,dBnpadr,dxidr,dQdr,dQdprst,dHOdddr,dBpdr,xi,omega,omega_circ,result
    cdef double A,Bp,Bnp,Bnpa,Q,Heven,Hodd,H_val
    cdef double aux_derivs[9]

    aux_derivs[:] = H.auxderivs(
        q, p,
        chi1_v, chi2_v,
        m_1, m_2,
        chi1_LN, chi2_LN,
        chi1_L, chi2_L,
    )


    dAdr = aux_derivs[0]
    dBnpdr = aux_derivs[1]
    dBnpadr = aux_derivs[2]
    dxidr = aux_derivs[3]
    dQdr = aux_derivs[4]
    dQdprst = aux_derivs[5]
    dHodddr = aux_derivs[6]
    dBpdr = aux_derivs[7]
    dHevendr = aux_derivs[8]

    H_val, xi, A, Bnp, Bnpa, Q, Heven, Hodd, Bp = H(
        q, p,
        chi1_v, chi2_v,
        m_1, m_2,
        chi1_LN, chi2_LN,
        chi1_L, chi2_L,
    )

    omega = H.omega(
        q, p,
        chi1_v, chi2_v,
        m_1, m_2,
        chi1_LN, chi2_LN,
        chi1_L, chi2_L,
    )

    if omega < 0.9*omega_start:
      print(f"problem PA dynamics is extrapolating: omega ={omega}< 0.9*omega_start {omega_start}")

    params.dynamics.p_circ[1] = pphi_sol
    omega_circ = H.omega(
        q, params.dynamics.p_circ,
        chi1_v, chi2_v,
        m_1, m_2,
        chi1_LN, chi2_LN,
        chi1_L, chi2_L,
    )


    params.p_params.omega_circ = omega_circ
    params.p_params.omega = omega
    params.p_params.H_val = H_val

    cdef double drdt = A/(2*nu*H_val*Heven)*(2*pr/xi*(1+Bnp)+xi*dQdprst)

    params.p_params.update_spins(chi1_LN, chi2_LN)

    cdef (double,double) flux = RR.RR(
        q, p,
        omega, omega_circ,
        H_val,
        params,
    )

    cdef double pr2 = pr*pr
    cdef double xi2 = xi*xi
    cdef double xi3 = xi2*xi
    cdef double r2 = r*r
    cdef double r3 = r2*r

    cdef double lap = chi1_L * X1 + chi2_L * X2
    cdef double lap2 = lap*lap


    cdef double C_2 = dAdr*(Bp/r2+lap2/r2*Bnpa)+A*(-2*Bp/r3-2*lap2/r3*Bnpa+lap2/r2*dBnpadr+1/r2*dBpdr)
    cdef double C_1 = dHodddr/pphi_sol*2*Heven
    cdef double C_0_pt1 = dAdr*(1+pr2/xi2*(1+Bnp) +Q) + A*(dQdr-2*pr2/xi3*dxidr*(1+Bnp)+pr2/xi2*(dBnpdr))
    cdef double tmp = 2*Heven*nu*H_val/xi
    cdef double C_0_pt2 = dpr_dr*drdt*tmp - pr/pphi_sol*tmp*flux[1]
    cdef double C_0 = C_0_pt1+C_0_pt2

    D = C_1*C_1 - 4*C_2*C_0

    if D < 0 and fabs(D) < 1e-10:
        D = 0.0
    if D < 0:
        raise ValueError

    result = (-C_1 - sqrt(D)) / (2 * C_2)

    return result


@cython.profile(True)
@cython.linetrace(True)
cpdef compute_pphi(
    double[:] r,
    double[:] pr,
    double[:] pphi,
    double[:] omega,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    splines,
    double m_1,
    double m_2,
    double omega_start,
    double[::1] q,
    double[::1] p,
    double tol=1e-12,
    EOBParams params=None,
    str postadiabatic_type="analytic",
):
    """
    Compute pphi at a given postadiabatic order by solving the equation
    dpr_dr * dH_dpr + dH_dr - flux[0] / csi, where flux[0] is the radial
    component of the radiation reaction force and csi is the tortoise coordinate
    transformation factor, for pphi. This can be done either by root finding
    (postadiabatic_type="numeric") or by finding an analytic expression for the
    root using the potentials of the Hamiltonian (postadiabatic_type="analytic").

    Args:
        r (double[:]): Radial grid on which to compute prstar.
        pr (double[:]): Prstar values of the previous postadiabatic order.
        pphi (double[:]): Pphi values of the previous postadiabatic order.
        omega (double[:]): Orbital frequency values of the previous postadiabatic order.
        H (Hamiltonian_v5PHM_C): Hamiltonian.
        RR (RadiationReactionForce): Radiation Reaction Force.
        splines (dict): Dictionary containing the interpolated spin-precessing evolution.
        m_1 (double): Mass of the primary.
        m_2 (double): Mass of the secondary.
        omega_start (double): Initial orbital frequency.
        q (double[::1]): Position q=[r,phi]= np.zeros(2). These values are updated in the root finding routine.
        p (double[::1]): Canonical momentum p=[pr,pphi]=np.zeros(2). These values are updated in the root finding routine.
        tol (double): Tolerance for the root finding routine in case postadiabatic_type="numeric" is used.
        params (EOBParams): Container with useful EOB parameters.
        postadiabatic_type (str): If "analytic" find root analytically, otherwise numericall by a root finding method.

    Returns:
        (np.array) Value of pphi at a given postadiabatic order for the whole radial grid.
    """

    cdef double[:] dpr_dr = spline_derivative(r, pr)

    cdef int i
    tmp = splines["everything"](omega)
    chi1_v = tmp[:,4:7]
    chi2_v = tmp[:,7:10]

    cdef double[:] chi1_LN = tmp[:,0]
    cdef double[:] chi2_LN = tmp[:,1]

    cdef double[:] chi1_L = tmp[:,2]
    cdef double[:] chi2_L = tmp[:,3]

    for i in range(r.shape[0]):
        if postadiabatic_type == "analytic":
            pphi_val = pphi_eqn_analytic(
                pphi[i], r[i], pr[i], dpr_dr[i],
                H, RR,
                chi1_v[i], chi2_v[i],
                chi1_LN[i], chi2_LN[i],
                chi1_L[i], chi2_L[i],
                m_1, m_2,
                omega_start,
                params,
                q, p,
            )
        else:
            pphi_solution = optimize.root(
                pphi_eqn,
                pphi[i],
                args=(
                    r[i], pr[i], dpr_dr[i],
                    H, RR,
                    chi1_v[i], chi2_v[i],
                    chi1_LN[i], chi2_LN[i],
                    chi1_L[i], chi2_L[i],
                    m_1, m_2,
                    omega_start,
                    params,
                    q, p,
                ),
                tol=tol,
            )
            pphi_val = pphi_solution.x

        omega[i] = params.p_params.omega
        pphi[i] = pphi_val

    return pphi, omega

@cython.profile(True)
@cython.linetrace(True)
cpdef compute_postadiabatic_solution(
    double[:] r,
    double[:] pphi,
    double[:] omega,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    splines,
    double m_1,
    double m_2,
    double[::1] q,
    double[::1] p,
    double tol=1e-12,
    int order=8,
    str postadiabatic_type="analytic",
    EOBParams params=None,
):
    """
    Compute postadiabatic solution for prstar, pphi and omega up to a certain
    postadiabatic order. This can be done either by root finding
    (postadiabatic_type="numeric") or by finding an analytic expression for the
    root using the potentials of the Hamiltonian (postadiabatic_type="analytic").

    Args:
        r (double[:]): Radial grid on which to compute postadiabatic solution.
        pphi (double[:]): Pphi values computed from the adiabatic solution.
        omega (double[:]): Orbital frequency values of the adiabatic solution.
        H (Hamiltonian_v5PHM_C): Hamiltonian.
        RR (RadiationReactionForce): Radiation Reaction Force.
        splines (dict): Dictionary containing the interpolated spin-precessing evolution.
        m_1 (double): Mass of the primary.
        m_2 (double): Mass of the secondary.
        q (double[::1]): Position q=[r,phi]= np.zeros(2). These values are updated in the root finding routine.
        p (double[::1]): Canonical momentum p=[pr,pphi]=np.zeros(2). These values are updated in the root finding routine.
        tol (double): Tolerance for the root finding routine in case postadiabatic_type="numeric" is used.
        order (int): Postadiabatic order up to which to obtain the solution for prstar and pphi.
        postadiabatic_type (str): If "analytic" find root analytically, otherwise numericall by a root finding method.
        params (EOBParams): Container with useful EOB parameters.

    Returns:
        (prstar, pphi, omega) up to a given postadiabatic order for the whole radial grid.
    """
    pr = np.zeros(r.size)
    cdef int i,n,parity
    cdef double tol_current
    cdef double X1 = params.p_params.X_1
    cdef double X2 = params.p_params.X_2

    cdef double omega_start = splines["everything"].x[0]

    for n in range(1, order+1):
        tol_current = 1e-3 / 10**n
        parity = n % 2
        if n>=7:
            tol_current=tol
        if parity:
            pr, _ = compute_pr(
                r,
                pr,
                pphi,
                omega,
                H,
                RR,
                splines,
                m_1,
                m_2,
                omega_start,
                q,
                p,
                tol=tol_current,
                postadiabatic_type=postadiabatic_type,
                params=params,
            )
        else:
            pphi, _ = compute_pphi(
                r,
                pr,
                pphi,
                omega,
                H,
                RR,
                splines,
                m_1,
                m_2,
                omega_start,
                q,
                p,
                tol=tol_current,
                postadiabatic_type=postadiabatic_type,
                params=params,
            )

        tmp = splines["everything"](omega)
        chi1_v = tmp[:,4:7]
        chi2_v = tmp[:,7:10]

        chi1_LN = tmp[:,0]
        chi2_LN = tmp[:,1]

        chi1_L = tmp[:,2]
        chi2_L = tmp[:,3]

        ap = chi1_LN * X1 + chi2_LN * X2
        am = chi1_LN * X1 - chi2_LN * X2
        dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)

        for i in range(r.size):
            q[0] = r[i]
            p[0] = pr[i]
            p[1] = pphi[i]

            params.p_params.chi1_v = chi1_v[i]
            params.p_params.chi2_v = chi2_v[i]

            params.p_params.chi_1 = chi1_LN[i]
            params.p_params.chi_2 = chi2_LN[i]

            params.p_params.chi1_L = chi1_L[i]
            params.p_params.chi2_L = chi2_L[i]

            H.calibration_coeffs['dSO'] = dSO_new[i]

            om = H.omega(
                q, p,
                chi1_v[i], chi2_v[i],
                m_1, m_2,
                chi1_LN[i], chi2_LN[i],
                chi1_L[i], chi2_L[i],
            )
            omega[i] = om

    return pr, pphi, omega

@cython.profile(True)
@cython.linetrace(True)
cpdef compute_postadiabatic_dynamics(
    double omega_ref,
    double omega_start,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    chi_1,
    chi_2,
    double m_1,
    double m_2,
    dict splines,
    t_pn: np.array,
    dynamics_pn: np.array,
    double tol=1e-12,
    EOBParams params=None,
    int order=1,
    str postadiabatic_type="analytic",
    int window_length=10,
    only_first_n=None,
):
    """
    Compute postadiabatic dynamics from the spins specified at a certain reference frequency and
    with a certain starting frequency. The postadiabatic evolution can be done
    either by root finding (postadiabatic_type="numeric") or by finding an analytic expression for
    the roots of prstar and pphi using the potentials of the Hamiltonian (postadiabatic_type="analytic").

    Args:
        omega_ref (double): Reference orbital frequency at which the spins are specified.
        omega_start (double): Initial orbital frequency.
        H (Hamiltonian_v5PHM_C): Hamiltonian.
        RR (RadiationReactionForce): Radiation Reaction Force.
        chi_1 : Primary dimensionless spin vector.
        chi_2 : Secondary dimensionless spin vector.
        m_1 (double): Mass of the primary
        m_2 (double): Mass of the secondary
        splines (dict): Dictionary containing the interpolated spin-precessing evolution.
        params (EOBParams): Container with useful EOB parameters
        tol (double): Tolerance for the root finding routine in case postadiabatic_type="numeric" is used.
        order (int): Postadiabatic order up to which to obtain the solution for prstar and pphi.
        postadiabatic_type (str): If "analytic" find root analytically, otherwise numericall by a root finding method.
        window_length (int): Minimum number of points in the radial grid.
        only_first_n (int): Compute the postadiabatic only on a specified number of points.

    Returns:
        (postadiabatic_dynamics, omega, lN_dyn, splines) where postadiabatic_dynamics includes also the
        quantities needed to compute the waveform modes = [t,r,phi,pr,pphi,H,omega,omega_circ,chi1LN,chi2LN].

    """

    cdef double Mtot = m_1 + m_2
    cdef double nu = m_1*m_2/Mtot/Mtot

    tmp = splines["everything"](omega_start)

    cdef double chi1_LN = tmp[0]
    cdef double chi2_LN = tmp[1]

    cdef double chi1_L = tmp[2]
    cdef double chi2_L = tmp[3]

    chi1_v = tmp[4:7]
    chi2_v = tmp[7:10]
    LNhat = tmp[10:13]

    params.p_params.omega = omega_start

    params.p_params.chi1_v = chi1_v
    params.p_params.chi2_v = chi2_v

    params.p_params.chi_1, params.p_params.chi_2 = chi1_LN, chi2_LN
    params.p_params.chi1_L, params.p_params.chi2_L = chi1_L, chi2_L

    params.p_params.update_spins(chi1_LN, chi2_LN)

    cdef double X1 = params.p_params.X_1
    cdef double X2 = params.p_params.X_2

    cdef double ap = chi1_LN * X1 + chi2_LN * X2
    cdef double am = chi1_LN * X1 - chi2_LN * X2
    cdef double chi_perp_eff = np.linalg.norm(chi1_v*X1+chi2_v*X2-ap*LNhat)

    cdef double dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)
    H.calibration_coeffs['dSO'] = dSO_new

    cdef double r0
    r0, _, _ =computeIC_augm(omega_start, H, RR, chi1_v, chi2_v, m_1, m_2, params=params)
    cdef double chi_eff = ap

    r_ISCO, _ = Kerr_ISCO(chi1_LN, chi2_LN, X1, X2)

    # Phenomenological prefactor to r_ISCO to adjust r_final in the precessing case
    cdef double r_final_prefactor = 2.7 + (chi_eff - 0.5*chi_perp_eff)*(1-4.*nu)

    # Compute the final point of the radial grid
    cdef double r_final = max(11.0, r_final_prefactor * r_ISCO)

    cdef double dr0 = 0.1
    cdef int r_size = int(np.ceil((r0 - r_final) / dr0))
    cdef double r_range = r0 - r_final

    # Compute now the number of precessing cycles (only in the aligned-spin limit)
    cdef double chi_in_plane = chi1_v[0]**2.+chi1_v[1]**2. + chi2_v[0]**2.+chi2_v[1]**2.
    cdef int r_size_new = 0
    cdef double precessiong_cycles = 0.0
    cdef double dr0_new = 0.1

    omega_pn = dynamics_pn[:,-1]
    lN_pn = dynamics_pn[:,:3]
    if chi_in_plane > 0 :
      precession_cycles = compute_prec_cycles(r_final,t_pn, omega_pn, lN_pn)

      # Multiply the number of precession cycles by 20, so that the have 20 points per precession cycle,
      # however as we are making a uniform radial grid below this condition will not be exactly fulfilled
      r_size_new = int(np.ceil(precession_cycles*20))


    # Some checks to decide whether the radial grid is large enough to perform the postadiabatic evolution
    if r_size <= 10 or r0<=11.5 or r_final >= r0:
        raise ValueError
    elif r_size < window_length + 2:
        r_size = window_length + 2

    # If the number of precessing cycles is zero, we use 0.1 as the default step size of the radial grid
    if r_size_new ==0:
      dr0_new = 0.1
      r_size_new = int(np.ceil(r_range/dr0_new))
    else:
      dr0_new = r_range/r_size_new

    # Put some upper and lower bounds to the step size of the radial grid
    if dr0_new < 0.05:
      dr0_new = 0.05
      r_size_new = int(np.ceil(r_range/dr0_new))

    if dr0_new > 0.1:
      dr0_new = 0.1
      r_size_new = int(np.ceil(r_range/dr0_new))

    # Compute the final grid
    r, _ = np.linspace(r0, r_final, num=r_size_new, endpoint=True, retstep=True)

    # Option only used in the PA initial conditions for the non-PA model
    if only_first_n is not None:
      r = r[:only_first_n]

    # First guess at circular omega, from Kepler's law
    omega = r**(-3./2)

    # Arrays for the position and canonical momentum
    cdef double[::1] q = np.zeros(2)
    cdef double[::1] p = np.zeros(2)

    # Compute the adiabatic solution
    pphi = compute_adiabatic_solution(r, omega, H, splines, m_1, m_2, q, p, params, tol=tol,)

    # Update the circular omega with the adiabatic solution
    for i in range(r.size):
        q[0] = r[i]
        p[1] = pphi[i]

        tmp = splines["everything"](omega[i])

        chi1_LN = tmp[0]
        chi2_LN = tmp[1]

        chi1_L = tmp[2]
        chi2_L = tmp[3]

        chi1_v = tmp[4:7]
        chi2_v = tmp[7:10]

        ap = chi1_LN * X1 + chi2_LN * X2
        am = chi1_LN * X1 - chi2_LN * X2
        dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)

        H.calibration_coeffs['dSO'] = dSO_new

        om = H.omega(
            q, p,
            chi1_v, chi2_v,
            m_1, m_2,
            chi1_LN, chi2_LN,
            chi1_L, chi2_L,
        )

        omega[i] = om
        if omega[i]< 0.9*omega_start :
          print(f"WARNING PA DYNAMICS IS EXTRAPOLATING!")


    # Compute the PA solution
    pr, pphi, omega = compute_postadiabatic_solution(
        r,
        pphi,
        omega,
        H,
        RR,
        splines,
        m_1,
        m_2,
        q,
        p,
        tol=tol,
        order=order,
        postadiabatic_type=postadiabatic_type,
        params=params,
    )

    dt_dr = np.zeros(r.shape[0])
    dphi_dr = np.zeros(r.shape[0])
    cdef double dH_dpr,dH_dpphi,csi
    cdef double dyn[6]
    cdef double[::1] p_circ = np.zeros(2)

    dyn_augm = []

    for i in range(r.shape[0]):
        q = np.array([r[i], 0])
        p = np.array([pr[i], pphi[i]])

        tmp = splines["everything"](omega[i])

        chi1_LN = tmp[0]
        chi2_LN = tmp[1]

        chi1_L = tmp[2]
        chi2_L = tmp[3]

        chi1_v = tmp[4:7]
        chi2_v = tmp[7:10]

        ap = chi1_LN * X1 + chi2_LN * X2
        am = chi1_LN * X1 - chi2_LN * X2

        dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)
        H.calibration_coeffs['dSO'] = dSO_new

        dyn[:] = H.dynamics(q, p, chi1_v, chi2_v, m_1, m_2,chi1_LN, chi2_LN, chi1_L, chi2_L)
        dH_dpr = dyn[2]
        dH_dpphi = dyn[3]
        H_val = dyn[4]
        csi = dyn[5]

        dt_dr[i] = 1 / (dH_dpr * csi)
        dphi_dr[i] = dH_dpphi / (dH_dpr * csi)
        omega[i] = dH_dpphi

        p_circ[1] = p[1]
        omega_circ = H.omega(q, p_circ, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L)
        dyn_augm.append([H_val,omega[i],omega_circ,chi1_LN,chi2_LN])

    # Compute integrals to obtain the time array and the orbital phase
    t = univariate_spline_integral(r,dt_dr)
    phi = univariate_spline_integral(r,dphi_dr)

    postadiabatic_dynamics = np.c_[t, r, phi, pr, pphi, dyn_augm]

    return postadiabatic_dynamics, omega



@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef compute_combined_dynamics_exp_v1(
    double omega_ref,
    double omega_start,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    double m_1,
    double m_2,
    chi_1,
    chi_2,
    double tol=1e-12,
    EOBParams params=None,
    double step_back=50,
    str backend="ode",
    int PA_order=8,
    str postadiabatic_type="analytic",
):

    """
    Compute the dynamics by using a postadiabatic scheme up to a certain separation from which an
    ODE integration is started. The postadiabatic evolution can be done either by root finding
    (postadiabatic_type="numeric") or by finding an analytic expression for the roots of prstar
    and pphi using the potentials of the Hamiltonian (postadiabatic_type="analytic").

    Args:
        omega_ref (double): Reference orbital frequency at which the spins are specified.
        omega_start (double): Initial orbital frequency.
        H (Hamiltonian_v5PHM_C): Hamiltonian.
        RR (RadiationReactionForce): Radiation Reaction Force.
        m_1 (double): Mass of the primary
        m_2 (double): Mass of the secondary
        chi_1 : Primary dimensionless spin vector.
        chi_2 : Secondary dimensionless spin vector.
        tol (double): Tolerance for the root finding routine in case postadiabatic_type="numeric" is used.
        params (EOBParams): Container with useful EOB parameters
        step_back (float, optional): Amount of time to step back for fine interpolation. Defaults to 250.
        PA_order (int): Postadiabatic order up to which to obtain the solution for prstar and pphi.
        postadiabatic_type (str): If "analytic" find root analytically, otherwise numericall by a root finding method.

        Returns:
            (tuple): Low and high sampling rate dynamics, unit Newtonian orbital angular momentum, assembled dynamics
                     and the index splitting the low and high sampling rate dynamics.

    """
    # Save initial frequency to use it afterwards for the roll-off
    cdef double omega_start_0 = omega_start

    combined_t, combined_y = compute_quasiprecessing_PNdynamics_opt(
        omega_ref,
        0.9*omega_start,
        m_1, m_2,
        chi_1, chi_2,
    )

    omegaPN_f = combined_y[:, -1][-1]

    splines = build_splines_PN(
        combined_t,
        combined_y,
        m_1, m_2,
        omega_start,
    )

    try:

        postadiabatic_dynamics, omega_pa = compute_postadiabatic_dynamics(
            omega_ref,
            omega_start,
            H,
            RR,
            chi_1,
            chi_2,
            m_1,
            m_2,
            splines,
            combined_t,
            combined_y,
            tol=tol,
            params=params,
            order=PA_order,
            postadiabatic_type=postadiabatic_type,
            window_length=10,
        )

        PA_success = True
        e_id = -1
        ode_y_init = postadiabatic_dynamics[e_id, 1:5]

    except ValueError as e:
        PA_success = False
        ode_y_init = None
        omega_pa = [omega_start]

    # Set omega_start to the last frequency of the PA evolution
    omega_start = omega_pa[-1]

    (
        ode_dynamics_low,
        ode_dynamics_high,
        dynamics,
        idx_restart
    ) = compute_dynamics_prec_opt(
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
        rtol=1e-8,
        atol=1e-8,
        step_back=step_back,
        y_init=ode_y_init,
    )


    if PA_success is True:

        # Interpolate the PA dynamics in its full range. Note that for the AS model this is only
        # done in a window before the ODE integration to smooth the timestep transition. For the
        # precessing model this is done on the full range of the post-adiabatic dynamics to ensure
        # enough resolution to accurately describe the rotations from the co-precessing frame to the
        # inertial frame

        t_pa = postadiabatic_dynamics[:, 0]

        ode_dynamics_low[:, 0] += t_pa[e_id]
        ode_dynamics_high[:, 0] += t_pa[e_id]

        # This is the part of the PA dynamics which is smoothed in
        # timestep until it reaches the one obtained solving ODEs
        t_ode_low = ode_dynamics_low[:, 0]

        dt_pa_first = t_pa[1] - t_pa[0]
        dt_ode_init = t_ode_low[1] - t_ode_low[0]

        # Estimate initial delta_t from the starting orbital frequency (this should correspond to the maximum timestep in the PA dynamics)
        dt0 = 2.*np.pi/omega_start_0

        while dt_pa_first>500:
            if  dt_pa_first > dt0:
              dt_pa_first = dt0/5.
            else:
              dt_pa_first = dt0/10.
            dt0 = dt_pa_first

        t_pa_first = t_pa[0]
        t_ode_init = t_ode_low[0]

        dt = dt_ode_init
        t = t_ode_init - dt
        t_new = []
        step_multiplier = 1.3

        while True:
            t_new.append(t)

            if step_multiplier * dt < dt_pa_first:
                dt *= step_multiplier

            t -= dt

            if t < t_pa_first:
                break

        # Add the initial time
        t_new.append(t_pa[0])

        # Reverse the timeseries
        t_new = t_new[::-1]

        # Interpolate window dynamics except the spins projections (otherwise bad things may happen due to the large timesteps of the PA dynamics)
        window_dynamics_interp = CubicSpline(t_pa, np.c_[postadiabatic_dynamics[:,:-2], omega_pa])
        tmp_window = window_dynamics_interp(t_new)

        window_dynamics = tmp_window[:,:-1]
        omega_new = tmp_window[:,-1]

        # Compute the spin projections onto LNhat from the new interpolated orbital frequency
        tmp = splines["everything"](omega_new)
        chi1_LN_window = tmp[:,0]
        chi2_LN_window = tmp[:,1]
        postadiabatic_dynamics_v1 = np.c_[window_dynamics, chi1_LN_window,chi2_LN_window]

        combined_dynamics = np.vstack((postadiabatic_dynamics_v1[:e_id], ode_dynamics_low))

    else:
        combined_dynamics = ode_dynamics_low

    dynamics = np.vstack((combined_dynamics, ode_dynamics_high))

    return combined_dynamics, ode_dynamics_high,combined_t,combined_y,splines,dynamics



@cython.boundscheck(True)
@cython.profile(True)
@cython.linetrace(True)
@cython.cdivision(True)
cpdef double compute_prec_cycles(r_final:double,t_pn: np.array, omega_pn: np.array, lN_pn:np.array):
    """
    Estimate the number of precession cycles from LNhat, computed as the phase of the precession
    frequency at the final point of the radial grid divided by 2 pi.

    Args:
        r_final (double): Last point of the radial grid.
        t_pn (np.array): Time array of the PN evolution.
        omega_pn (np.array): Orbital frequency from the PN evolution.
        lN_pn (np.array): LNhat obtained from the PN evolution.

    Returns:
        (double) Number of precession cycles.
    """

    # Compute the LNhat quaternion
    lN_quat = quaternion.as_quat_array(np.c_[np.zeros(len(lN_pn)), lN_pn])


    # Compute the angular velocity
    omega_prec_arr = quaternion.angular_velocity(lN_quat,t_pn)/2.0
    om_prec_norm = np.sqrt(np.einsum("ij,ij->i", omega_prec_arr, omega_prec_arr))

    # Compute the value of the precession frequency at r_final
    r_pn = omega_pn**(-3./2.)
    iut = CubicSpline(1./r_pn,t_pn)
    iom_prec = CubicSpline(t_pn,om_prec_norm)

    # Compute the phase
    ph_om_prec = iom_prec.antiderivative()(t_pn)
    iph_prec = CubicSpline(t_pn,ph_om_prec)


    # Evaluate the phase at t_final
    t_final = iut(1./r_final)
    prec_cycles = iph_prec(t_final)/(2.*np.pi)


    return prec_cycles



@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
@cython.cdivision(True)
def spline_derivative(x: np.array, y: np.array)->np.array:
      """
      Compute derivative of y(x) using cubic splines.

      Args:
          x (np.array): Dependent variable.
          y (np.array): Independent variable.

      Returns:
          (np.array) derivaive dy/dx. Note that the order is reversed as x will be
                     the orbital separation which is monotonically decreasing, while
                     the interpolation requires x to be monotonically increasing.
      """
      intrp = CubicSpline(x[::-1], y[::-1])
      deriv = intrp.derivative()(x[::-1])[::-1]
      return deriv
