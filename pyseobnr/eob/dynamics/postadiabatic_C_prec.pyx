# cython: language_level=3
cimport cython
from typing import Dict
import numpy as np
from libc.math cimport log, sqrt, exp, abs,fabs, tgamma,sin,cos, tanh, sinh, asinh

from .initial_conditions_aligned_opt import computeIC_opt as computeIC
from ..fits.fits_Hamiltonian import dSO as dSO_poly_fit

from pygsl import  roots, errno
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

fin_diff_coeffs_order_9 = np.array([
    [-761./280., 8., -14., 56./3., -35./2., 56./5., -14./3., 8./7., -1./8.],
    [-1./8., -223./140., 7./2., -7./2., 35./12., -7./4., 7./10., -1./6., 1./56.],
    [1./56., -2./7., -19./20., 2., -5./4., 2./3., -1./4., 2./35., -1./168.],
    [-1./168., 1./14., -1./2., -9./20., 5./4., -1./2., 1./6., -1./28., 1./280.],
    [1./280., -4./105., 1./5., -4./5., 0, 4./5., -1./5., 4./105., -1./280.],
    [-1./280., 1./28., -1./6., 1./2., -5./4., 9./20., 1./2., -1./14., 1./168.],
    [1./168., -2./35., 1./4., -2./3., 5./4., -2., 19./20., 2./7., -1./56.],
    [-1./56., 1./6., -7./10., 7./4., -35./12., 7./2., -7./2., 223./140., 1./8.],
    [1./8., -8./7., 14./3., -56./5., 35./2., -56./3., 14., -8., 761./280.],
])

interpolated_integral_order_3 = [
    [3./8., 19./24., -5./24., 1./24.],
    [-1./24., 13./24., 13./24., -1./24.],
    [1./24., -5./24., 19./24., 3./8.],
]

interpolated_integral_order_5 = [
    [95./288., 1427./1440., -133./240., 241./720., -173./1440., 3./160.],
    [-3./160., 637./1440., 511./720., -43./240., 77./1440., -11./1440.],
    [11./1440., -31./480., 401./720., 401./720., -31./480., 11./1440.],
    [-11./1440., 77./1440., -43./240., 511./720., 637./1440., -3./160.],
    [3./160., -173./1440., 241./720., -133./240., 1427./1440., 95./288.],
]

interpolated_integral_order_7 = [
    [5257./17280, 139849./120960, -(4511./4480), 123133./120960, -(88547./120960), 1537./4480, -(11351./120960), 275./24192],
    [-(275./24192), 5311./13440, 11261./13440, -(44797./120960), 2987./13440, -(1283./13440), 2999./120960, -(13./4480)],
    [13./4480, -(4183./120960), 6403./13440, 9077./13440, -(20227./120960), 803./13440, -(191./13440), 191./120960],
    [-(191./120960), 1879./120960, -(353./4480), 68323./120960, 68323./120960, -(353./4480), 1879./120960, -(191./120960)],
    [191./120960, -(191./13440), 803./13440, -(20227./120960), 9077./13440, 6403./13440, -(4183./120960), 13./4480],
    [-(13./4480), 2999./120960, -(1283./13440), 2987./13440, -(44797./120960), 11261./13440, 5311./13440, -(275./24192)],
    [275./24192, -(11351./120960), 1537./4480, -(88547./120960), 123133./120960, -(4511./4480), 139849./120960, 5257./17280],
]


@cython.profile(True)
@cython.linetrace(True)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef double single_deriv(double[:] y,double h,double[:] coeffs):
    """
    Compute the derivative.

    Args:
        y (double[:]): value of the function
        h (double): step
        coeffs (double[:]): values of the coefficients of the derivative

    Returns:
        (double) derivative
    """
    cdef int i
    cdef double total = 0.0
    for i in range(9):
        total+=coeffs[i]*y[i]
    total/=h
    return total

@cython.profile(True)
@cython.linetrace(True)
@cython.cdivision(True)
cpdef fin_diff_derivative(
    x,
    y,
    n: int = 8,
):
    """
    Compute the finite difference derivative of y(x) with n grid points.

    Args:
        x: Dependent variable
        y: Independent varible
        n: Number of grid points

    Returns:
        (np.array) derivative dy_dx
    """
    dy_dx = np.zeros(x.size)
    cdef double h = fabs(x[1] - x[0])
    cdef int size = x.shape[0]
    cdef int i
    for i in range(size):
        if i == 0:
            dy_dx[i] = single_deriv(y[0:9],h,fin_diff_coeffs_order_9[0])
        elif i == 1:
            dy_dx[i] = single_deriv(y[0:9],h,fin_diff_coeffs_order_9[1])
        elif i == 2:
            dy_dx[i] = single_deriv(y[0:9],h,fin_diff_coeffs_order_9[2])
        elif i == 3:
            dy_dx[i] = single_deriv(y[0:9],h,fin_diff_coeffs_order_9[3])
        elif i == size - 4:
            dy_dx[i] = single_deriv(y[-9:],h,fin_diff_coeffs_order_9[5])
        elif i == size - 3:
            dy_dx[i] = single_deriv(y[-9:],h,fin_diff_coeffs_order_9[6])
        elif i == size - 2:
            dy_dx[i] = single_deriv(y[-9:],h,fin_diff_coeffs_order_9[7])
        elif i == size - 1:
            dy_dx[i] = single_deriv(y[-9:],h,fin_diff_coeffs_order_9[8])
        else:
            dy_dx[i] = single_deriv(y[i-4:i+5],h,fin_diff_coeffs_order_9[4])

    return dy_dx

@cython.profile(True)
@cython.linetrace(True)
cpdef Kerr_ISCO(
    double chi1,
    double chi2,
    double m1,
    double m2,
):
    """
    Orbital separation and angular momentum of the Innermost Stable Circular Orbit (ISCO)
    for a Kerr black hole computed from the final spin estimated from NR fits

    Args:
        chi1 (double): Projection of the primary dimensionless spin onto LNhat
        chi2 (double): Projection of the secondary dimensionless spin onto LNhat
        m1 (double): Mass of the primary
        m2 (double): Mass of the secondary

    Returns:
        (np.array) rISCO and LISCO
    """
    # Final spin computed from NR fits
    a = nrutils.bbh_final_spin_non_precessing_HBR2016(
            m1, m2, chi1, chi2, version="M3J4"
    )
    # Compute the ISCO radius for this spin
    Z_1 = 1+(1-a**2)**(1./3)*((1+a)**(1./3)+(1-a)**(1./3))
    Z_2 = np.sqrt(3*a**2+Z_1**2)
    r_ISCO = 3+Z_2-np.sign(a)*np.sqrt((3-Z_1)*(3+Z_1+2*Z_2))

    # Compute the ISCO L for this spin
    L_ISCO = 2/(3*np.sqrt(3))*(1+2*np.sqrt(3*r_ISCO)-2)
    return np.array([r_ISCO,L_ISCO])

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
        chi1_LN (double): Projection of the primary dimensionless pin onto LNhat
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
    cdef double sign_final_spin = final_spin_nonprecessing / np.linalg.norm( final_spin_nonprecessing)

    # Compute the magnitude of the final spin using the precessing fit
    final_spin = nrutils.bbh_final_spin_precessing_HBR2016(
        m1, m2, a1, a2, tilt1, tilt2, phi12, version="M3J4"
    )

    # Flip sign if needed
    final_spin *= sign_final_spin

    return final_spin



@cython.profile(True)
@cython.linetrace(True)
def Newtonian_j0(r):
    """
    Newtonian estimate of the orbital angular momentum

    Args:
        r(double): Orbital separation

    Returns:
        (double) Orbital angular momentum
    """
    return np.sqrt(r)

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
    #if omega[0] < omega_start:
    #    omega[0] = omega_start

    tmp = splines["everything"](omega)
    cdef double[:] chi1_LN = tmp[:,0]
    cdef double[:] chi2_LN = tmp[:,1]

    chi1_v = tmp[:,4:7]
    chi2_v = tmp[:,7:10]

    cdef double[:] chi1_L = tmp[:,2]
    cdef double[:] chi2_L = tmp[:,3]

    for i in range(r.shape[0]):

        if omega[i] < 0.9*omega_start:
          print(f"problem: omega ={omega[i]}< 0.9*omega_start {omega_start}")
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
    #cdef (double,double,double,double)  dH_dx = dynamics[:4]

    dH_dpr = dynamics[2]
    H_val = dynamics[4]
    csi = dynamics[5]


    params.dynamics.p_circ[1] = pphi
    omega_circ = H.omega(q,params.dynamics.p_circ,chi1_v,chi2_v,m_1,m_2,chi1_LN, chi2_LN,
        chi1_L, chi2_L
    )

    omega = dynamics[3]
    #if omega < omega_start:
    #  omega = omega_start

    if omega < 0.9*omega_start:
      print(f"problem: omega ={omega}< 0.9*omega_start {omega_start}")

    params.p_params.update_spins(chi1_LN, chi2_LN)
    cdef (double,double) flux = RR.RR(q, p, omega,omega_circ,H_val,params)

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
    #if omega < omega_start:
    #  omega = omega_start
    if omega < 0.9*omega_start:
      print(f"problem: omega ={omega}< 0.9*omega_start {omega_start}")

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
    #print(f"r={r}")
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

    #print(f"r= {np.array(r)}, pphi = {np.array(pphi)}")
    cdef double[:] dpphi_dr = - fin_diff_derivative(r, pphi)
    cdef int i

    #if omega[0] < omega_start:
    #    omega[0] = omega_start

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
        #if i == 1:
        #print(f"pr : i = {i}, pr = {pr[i]}, omega = {omega[i]} r = {r[i]}, pphi = {pphi[i]}, dpphi_dr = {dpphi_dr[i]}, params.omega = {params.p_params.omega}")
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
          print(f"problem: omega ={omega[i]}< 0.9*omega_start {omega_start}")


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
    #cdef (double,double,double,double) dH_dx = dynamics[:4]
    dH_dr = dynamics[0]
    dH_dpr = dynamics[2]
    H_val = dynamics[4]
    csi = dynamics[5]

    params.dynamics.p_circ[1] =  pphi_sol[0]
    omega_circ = H.omega(q,params.dynamics.p_circ,chi1_v,chi2_v,m_1,m_2,chi1_LN, chi2_LN, chi1_L, chi2_L)

    omega = dynamics[3]
    if omega < 0.9*omega_start:
      print(f"problem: omega ={omega}< 0.9*omega_start {omega_start}")
      #omega = omega_start

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

    #if omega < omega_start:
    #    omega = omega_start

    if omega < 0.9*omega_start:
      print(f"problem: omega ={omega}< 0.9*omega_start {omega_start}")



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
    cdef double[:] dpr_dr = - fin_diff_derivative(r, pr)
    cdef int i

    tmp = splines["everything"](omega)

    #if omega[0] < omega_start:
    #    omega[0] = omega_start

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
        #print(f"pphi : i = {i}, pr = {pr[i]}, omega = {omega[i]} r = {r[i]}, pphi = {pphi[i]}, dpr_dr = {dpr_dr[i]}, params.omega = {params.p_params.omega}")

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
    #omega[0] = omega_start


    #print(f"Adiabatic omega: {np.array(omega)}")

    for n in range(1, order+1):
        tol_current = 1e-3 / 10**n
        parity = n % 2
        if n>=7:
            tol_current=tol
        if parity:
            pr, omega = compute_pr(
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
            pphi, omega = compute_pphi(
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

        #if i ==0:
        #print(f"PA sol. : n = {n}, i = {0}, r[0] = {np.round(r[0],2)}, pphi[0] = {pphi[0]}, pr[0] = {pr[0]}, omega[0] = {omega[0]}")

        #print(f"n = {n}, i = {0}, r = {np.round(r[0],2)}, pphi = {np.round(pphi[0],6)}, pr = {np.round(pr[0],6)}, omega = {omega[0]}")
        tmp = splines["everything"](omega)
        chi1_v = tmp[:,4:7]
        chi2_v = tmp[:,7:10]
        lN_v = tmp[:,10:13]

        chi1_LN = tmp[:,0]
        chi2_LN = tmp[:,1]

        chi1_L = tmp[:,2]
        chi2_L = tmp[:,3]

        ap = chi1_LN * X1 + chi2_LN * X2
        am = chi1_LN * X1 - chi2_LN * X2
        dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)

        #print(f"Before update Omega : n = {n}, omega = {np.array(omega)}, chi1_LN = {chi1_LN}, chi2_LN = {chi2_LN}")

        for i in range(r.size):
            q[0] = r[i]
            p[0] = pr[i]
            p[1] = pphi[i]

            params.p_params.chi1_v = chi1_v[i]
            params.p_params.chi2_v = chi2_v[i]
            params.p_params.lN = lN_v[i]

            params.p_params.chi_1 = chi1_LN[i]
            params.p_params.chi_2 = chi2_LN[i]

            params.p_params.chi1_L = chi1_L[i]
            params.p_params.chi2_L = chi2_L[i]

            H.calibration_coeffs['dSO'] = dSO_new[i]
            #"""
            om = H.omega(
                q, p,
                chi1_v[i], chi2_v[i],
                m_1, m_2,
                chi1_LN[i], chi2_LN[i],
                chi1_L[i], chi2_L[i],
            )
            omega[i] = om
            #"""
            #if omega[i] < omega_start:
            #  omega[i] = omega_start


            params.p_params.chi_1 = chi1_LN[i]
            params.p_params.chi_2 = chi2_LN[i]

            #if n==order:
            #if i ==0:
            #if i<2:
            #  print(f"After PA iter. : i = {i}, n ={n}, r = {np.round(r[i],2)}, pphi = {np.round(pphi[i],5)}, pr = {np.round(pr[i],5)}, omega = {omega[i]}")
            #  #omega[i] = omega_start
            #print(f"PA sol: n = {n}, i = {i}, r[i] = {np.round(r[i],2)}, om = {om}, pr = {pr[i]}, pphi = {pphi[i]}")

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
    double tol=1e-12,
    EOBParams params=None,
    order=1,
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
    params.p_params.lN = LNhat

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

    cdef double r_final_prefactor = 2.7 + chi_eff*(1-4.*nu) + chi_perp_eff
    #print(f"chi1_v = {chi1_v}, chi2_v = {chi2_v}, LNhat = {LNhat}")

    #print(f"r_final_prefactor = {r_final_prefactor}, chi_eff = {chi_eff}, chi_perp_eff = {chi_perp_eff}")
    r_ISCO, _ = Kerr_ISCO(chi1_LN, chi2_LN, X1, X2)
    #print(f"r_ISCO = {r_ISCO}, chi1_LN = {chi1_LN}, chi2_LN = {chi2_LN}, chi_perp_eff = {chi_perp_eff}")


    cdef double a_f = precessing_final_spin(chi1_LN, chi2_LN, chi1_v, chi2_v, LNhat, X1,X2)
    cdef double sign_final_spin = a_f/np.linalg.norm(a_f)

    chi_total = chi1_v + chi2_v
    chi_total_norm = np.linalg.norm(chi_total)
    chi_parallel = np.dot(chi_total, LNhat)

    chi_perp = chi_total - chi_parallel*LNhat
    cdef double chi_perp_norm = np.linalg.norm(chi_perp)

    cdef double alpha=0.5
    cdef double chi_tilde = a_f*np.dot(chi_perp/chi_perp_norm,chi_total)/(chi_total_norm*(1-2*nu))
    #chi_perp_eff *=  1.5*sign_final_spin

    cdef double r_final_prefactor_test = 2.7 + (chi_eff - 0.5*chi_perp_eff)*(1-4.*nu) #+ chi_tilde

    #print(f"r_final_prefactor = {r_final_prefactor}, r_final_prefactor = {r_final_prefactor_test}, chi_eff = {chi_eff}, chi_perp_eff = {chi_perp_eff}")
    #print(f"r_final_old = {r_final_prefactor*r_ISCO}, r_final_new = {r_final_prefactor_test*r_ISCO}")
    cdef double r_final = max(12.0, r_final_prefactor_test * r_ISCO)
    #r_final = 14.0
    cdef double r_switch_prefactor = 1.6
    cdef double r_switch = r_switch_prefactor * r_ISCO

    cdef double dr0 = 0.1
    cdef int r_size = int(np.ceil((r0 - r_final) / dr0))
    r_range = r0 - r_final
    #print(f"r0 = {r0}, r_final = {r_final}, dr0 = {dr0}, r_switch = {r_switch}, r_size = {r_size}, r_range = {r_range}")

    #"""
    cdef double omega_start_1 = omega_start

    """
    if r_range<3:
      #r0 += 2
      omega_start = (r0 + 2)**(-3./2.)
      params.p_params.omega = omega_start

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

      tmp = splines["everything"](omega_start)

      chi1_LN = tmp[0]
      chi2_LN = tmp[1]

      chi1_L = tmp[2]
      chi2_L = tmp[3]

      chi1_v = tmp[4:7]
      chi2_v = tmp[7:10]
      LNhat = tmp[10:13]

      #print(f"chi1_v = {chi1_v}, chi2_v = {chi2_v}, LNhat = {LNhat}")


      params.p_params.omega = omega_start
      params.p_params.lN = LNhat

      params.p_params.chi1_v = chi1_v
      params.p_params.chi2_v = chi2_v
      #print(f"params.chi1_v = {params.p_params.chi1_v[0],params.p_params.chi1_v[1],params.p_params.chi1_v[2]}")
      #print(f"params.chi2_v = {params.p_params.chi2_v[0],params.p_params.chi2_v[1],params.p_params.chi2_v[2]}")
      #print(f"params.LNhat = {params.p_params.lN[0],params.p_params.lN[1],params.p_params.lN[2]}")

      params.p_params.chi_1, params.p_params.chi_2 = chi1_LN, chi2_LN
      params.p_params.chi1_L, params.p_params.chi2_L = chi1_L, chi2_L

      params.p_params.update_spins(chi1_LN, chi2_LN)

      ap = chi1_LN * X1 + chi2_LN * X2
      am = chi1_LN * X1 - chi2_LN * X2
      chi_perp_eff = np.linalg.norm(chi1_v*X1+chi2_v*X2-ap*LNhat)

      dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)
      H.calibration_coeffs['dSO'] = dSO_new

      r0, _, _ = computeIC_augm(omega_start, H, RR, chi1_v, chi2_v, m_1, m_2, params=params)

      #print(f"New r0 = {r0}, omega_start = {omega_start}, omega_ref = {omega_ref}")
      r_size = int(np.ceil((r0 - r_final) / dr0))


      r_ISCO, _ = Kerr_ISCO(chi1_LN, chi2_LN, X1, X2)
      r_final_prefactor_test = 2.7 + (ap - chi_perp_eff)*(1-4.*nu)
      r_final = max(10.0, r_final_prefactor_test * r_ISCO)
      #print(f"r_ISCO = {r_ISCO}, chi1_LN = {chi1_LN}, chi2_LN = {chi2_LN}, chi_perp_eff = {chi_perp_eff}")
      r_size = int(np.ceil((r0 - r_final) / dr0))
      #print(f"r0 = {r0}, r_final_prefactor_test = {r_final_prefactor_test}, r_final = {r_final}, r_size = {r_size}")
      omega_start_1 = omega_start
    """
    if r_size <= 4 or r0<=11.5:# or r_range < 3 :
        #print(f"r_size = {r_size} <= 4 or r0 = {r0} <= 11.5")
        raise ValueError
    elif r_size < window_length + 2:
        r_size = window_length + 2

    r, _ = np.linspace(r0, r_final, num=r_size, endpoint=True, retstep=True)
    #print(f"Radial grid : r = {r}")

    if only_first_n is not None:
      r = r[:only_first_n]

    # First guess at circular omega, from Kepler's law
    omega = r**(-3./2)

    #print(f"r0 = {r0}")
    #print(f"First guess omega = {np.array(omega)}, r = {r}")
    cdef double[::1] q = np.zeros(2)
    cdef double[::1] p = np.zeros(2)

    #print(f"0pa: omega[0] = {omega[0]}")

    # Compute the adiabatic solution
    pphi = compute_adiabatic_solution(r, omega, H, splines, m_1, m_2, q, p, params, tol=tol,)

    #print(f"Adiabatic pphi = {np.array(pphi)}")
    #print(f"omega_start = {omega_start}, omega[0] = {omega[0]}")

    # Update the circular omega with the adiabatic solution
    for i in range(r.size):
        q[0] = r[i]
        p[1] = pphi[i]


        # omega[0] = omega_start
        # print(f"Ad. sol: i = {i}, omega[i] = {omega[i]}, omega_start = {omega_start}, r[i] = {r[i]}")
        #if i == 0:
        #print(f"Ad. sol: i = {i}, omega[i] = {omega[i]}, r[i] = {r[i]}")

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
          print(f"PROBLEM WE ARE EXTRAPOLATING!")
        #  omega[i] = om

        #omega[i] = omega_start

        #if i ==0:
        #  omega[i] = omega_start
        #print(f"Ad. sol: om = {om}")
        #print("---------------------------")


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
    #print(f"PA sol. : omega[0]  = {omega[0]}, omega_start = {omega_start}")
    #for i in range(r.size):
    #if i == 0:
    #  print(f"PA sol. : i = {i}, r = {r[i]}, pphi = {pphi[i]}, pr = {pr[i]}, omega = {omega[i]}")

    #    print(f"PA sol. : i = {i}, r = {r[i]}, pphi = {pphi[i]}, pr = {pr[i]}, omega = {omega[i]}")
    dt_dr = np.zeros(r.shape[0])
    dphi_dr = np.zeros(r.shape[0])
    cdef double dH_dpr,dH_dpphi,csi
    cdef double dyn[6]
    cdef double[::1] p_circ = np.zeros(2)

    cdef double[::1] tmp_LN = np.zeros(3)

    dyn_augm = []
    lN_dyn = []


    for i in range(r.shape[0]):
        q = np.array([r[i], 0])
        p = np.array([pr[i], pphi[i]])


        #if i <3:
        #  omega[i] = omega_start
        #  print(f"i = {i}, omega[i] = {omega[i]}, omega_start = {omega_start}, r[i] = {np.round(r[i],2)}")
        tmp = splines["everything"](omega[i])

        chi1_LN = tmp[0]
        chi2_LN = tmp[1]

        chi1_L = tmp[2]
        chi2_L = tmp[3]

        chi1_v = tmp[4:7]
        chi2_v = tmp[7:10]
        tmp_LN = tmp[10:13]

        ap = chi1_LN * X1 + chi2_LN * X2
        am = chi1_LN * X1 - chi2_LN * X2

        dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)
        H.calibration_coeffs['dSO'] = dSO_new
        #print(f"i = {i}, dSO = {dSO_new}, chi1_v = {chi1_v}, chi2_v = {chi2_v}, m1 = {m_1}, m2 = {m_2}")
        #if i < 2:
        #  print(f"i = {i}, q[0] = {float(q[0])},  q[1] = {float(q[1])}, p[0] = {float(p[0])}, p[1] = {float(p[1])}")
        dyn[:] = H.dynamics(q, p, chi1_v, chi2_v, m_1, m_2,chi1_LN, chi2_LN, chi1_L, chi2_L)
        dH_dpr = dyn[2]
        dH_dpphi = dyn[3]
        csi = dyn[5]
        dt_dr[i] = 1 / (dH_dpr * csi)
        dphi_dr[i] = dH_dpphi / (dH_dpr * csi)



        omega[i] = dH_dpphi
        #if i ==0:
        #  print(f"After iteration. i = {i}, omega[i] = {omega[i]}, omega_start = {omega_start}, r[i] = {np.round(r[i],2)}")

        #print(f"i = {i}, chi1_LN = {chi1_LN}, chi2_LN = {chi2_LN}, chi1_L = {chi1_L}, chi2_L = {chi2_L}")
        #if i < 2:
        #print(f"After PA: i = {i}, omega[i] = {omega[i]}, r[i] = {r[i]}, dt_dr[i] = {dt_dr[i]}, dH_dpr = {dH_dpr}")
        #print(f"After PA: i = {i}, omega[i] = {omega[i]}, r[i] = {r[i]}, pphi[i] = {pphi[i]}, pr[i] = {pr[i]}")
        #print("====================================")
        H_val = dyn[4]
        p_circ[1] = p[1]
        omega_circ = H.omega(q, p_circ, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L)
        dyn_augm.append([H_val,omega[i],omega_circ,chi1_LN,chi2_LN])
        lN_dyn.append(tmp_LN)


    t = cumulative_integral(r, dt_dr)
    #print(f"t = {t}, omega = {np.array(omega)}, r = {r}, dt_dr = {dt_dr}")
    phi = cumulative_integral(r, dphi_dr)
    postadiabatic_dynamics = np.c_[t, r, phi, pr, pphi, dyn_augm]

    #print(f"pa_dyn =  {postadiabatic_dynamics}")
    lN_dyn = np.array(lN_dyn)

    return postadiabatic_dynamics, omega, lN_dyn, splines, omega_start_1


@cython.profile(True)
@cython.linetrace(True)
cpdef compute_postadiabatic_dynamics_stepback(
    double omega_ref,
    double omega_start,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    chi_1,
    chi_2,
    double m_1,
    double m_2,
    dict splines,
    double tol=1e-12,
    EOBParams params=None,
    order=1,
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
    params.p_params.lN = LNhat

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

    cdef double r_final_prefactor = 2.7 + chi_eff*(1-4.*nu) + chi_perp_eff
    #print(f"chi1_v = {chi1_v}, chi2_v = {chi2_v}, LNhat = {LNhat}")

    #print(f"r_final_prefactor = {r_final_prefactor}, chi_eff = {chi_eff}, chi_perp_eff = {chi_perp_eff}")
    r_ISCO, _ = Kerr_ISCO(chi1_LN, chi2_LN, X1, X2)
    #print(f"r_ISCO = {r_ISCO}, chi1_LN = {chi1_LN}, chi2_LN = {chi2_LN}, chi_perp_eff = {chi_perp_eff}")


    cdef double a_f = precessing_final_spin(chi1_LN, chi2_LN, chi1_v, chi2_v, LNhat, X1,X2)
    cdef double sign_final_spin = a_f/np.linalg.norm(a_f)

    chi_total = chi1_v + chi2_v
    chi_total_norm = np.linalg.norm(chi_total)
    chi_parallel = np.dot(chi_total, LNhat)

    chi_perp = chi_total - chi_parallel*LNhat
    cdef double chi_perp_norm = np.linalg.norm(chi_perp)

    cdef double alpha=0.5
    cdef double chi_tilde = a_f*np.dot(chi_perp/chi_perp_norm,chi_total)/(chi_total_norm*(1-2*nu))
    #chi_perp_eff *=  1.5*sign_final_spin

    cdef double r_final_prefactor_test = 2.7 + (chi_eff - chi_perp_eff)*(1-4.*nu) #+ chi_tilde

    #print(f"r_final_prefactor = {r_final_prefactor}, r_final_prefactor = {r_final_prefactor_test}, chi_eff = {chi_eff}, chi_perp_eff = {chi_perp_eff}")
    #print(f"r_final_old = {r_final_prefactor*r_ISCO}, r_final_new = {r_final_prefactor_test*r_ISCO}")
    cdef double r_final = max(10.0, r_final_prefactor_test * r_ISCO)

    cdef double r_switch_prefactor = 1.6
    cdef double r_switch = r_switch_prefactor * r_ISCO

    cdef double dr0 = 0.1
    cdef int r_size = int(np.ceil((r0 - r_final) / dr0))
    r_range = r0 - r_final
    #print(f"r0 = {r0}, r_final = {r_final}, dr0 = {dr0}, r_switch = {r_switch}, r_size = {r_size}, r_range = {r_range}")

    #"""
    cdef double omega_start_1 = omega_start

    if r_range<3:
      #r0 += 2
      omega_start = (r0 + 1)**(-3./2.)
      params.p_params.omega = omega_start
      #print(f"===> New omega_start = {omega_start}")
      return  [-1, -1, -1, -1, -1]
      """
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

      tmp = splines["everything"](omega_start)

      chi1_LN = tmp[0]
      chi2_LN = tmp[1]

      chi1_L = tmp[2]
      chi2_L = tmp[3]

      chi1_v = tmp[4:7]
      chi2_v = tmp[7:10]
      LNhat = tmp[10:13]

      #print(f"chi1_v = {chi1_v}, chi2_v = {chi2_v}, LNhat = {LNhat}")


      params.p_params.omega = omega_start
      params.p_params.lN = LNhat

      params.p_params.chi1_v = chi1_v
      params.p_params.chi2_v = chi2_v
      #print(f"params.chi1_v = {params.p_params.chi1_v[0],params.p_params.chi1_v[1],params.p_params.chi1_v[2]}")
      #print(f"params.chi2_v = {params.p_params.chi2_v[0],params.p_params.chi2_v[1],params.p_params.chi2_v[2]}")
      #print(f"params.LNhat = {params.p_params.lN[0],params.p_params.lN[1],params.p_params.lN[2]}")

      params.p_params.chi_1, params.p_params.chi_2 = chi1_LN, chi2_LN
      params.p_params.chi1_L, params.p_params.chi2_L = chi1_L, chi2_L

      params.p_params.update_spins(chi1_LN, chi2_LN)

      ap = chi1_LN * X1 + chi2_LN * X2
      am = chi1_LN * X1 - chi2_LN * X2
      chi_perp_eff = np.linalg.norm(chi1_v*X1+chi2_v*X2-ap*LNhat)

      dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)
      H.calibration_coeffs['dSO'] = dSO_new

      r0, _, _ = computeIC_augm(omega_start, H, RR, chi1_v, chi2_v, m_1, m_2, params=params)

      #print(f"New r0 = {r0}, omega_start = {omega_start}, omega_ref = {omega_ref}")
      r_size = int(np.ceil((r0 - r_final) / dr0))


      r_ISCO, _ = Kerr_ISCO(chi1_LN, chi2_LN, X1, X2)
      r_final_prefactor_test = 2.7 + (ap - chi_perp_eff)*(1-4.*nu)
      r_final = max(10.0, r_final_prefactor_test * r_ISCO)
      #print(f"r_ISCO = {r_ISCO}, chi1_LN = {chi1_LN}, chi2_LN = {chi2_LN}, chi_perp_eff = {chi_perp_eff}")
      r_size = int(np.ceil((r0 - r_final) / dr0))
      #print(f"r0 = {r0}, r_final_prefactor_test = {r_final_prefactor_test}, r_final = {r_final}, r_size = {r_size}")
      omega_start_1 = omega_start
      """

    if r_size <= 4 or r0<=11.5:# or r_range < 3 :
        #print(f"r_size = {r_size} <= 4 or r0 = {r0} <= 11.5")
        raise ValueError
    elif r_size < window_length + 2:
        r_size = window_length + 2

    r, _ = np.linspace(r0, r_final, num=r_size, endpoint=True, retstep=True)
    #print(f"Radial grid : r = {r}")

    if only_first_n is not None:
      r = r[:only_first_n]

    # First guess at circular omega, from Kepler's law
    omega = r**(-3./2)

    #print(f"r0 = {r0}")
    #print(f"First guess omega = {np.array(omega)}, r = {r}")
    cdef double[::1] q = np.zeros(2)
    cdef double[::1] p = np.zeros(2)

    #print(f"0: omega[0] = {omega[0]}")

    # Compute the adiabatic solution
    pphi = compute_adiabatic_solution(r, omega, H, splines, m_1, m_2, q, p, params, tol=tol,)

    #print(f"Adiabatic pphi = {np.array(pphi)}")
    #print(f"omega_start = {omega_start}, omega[0] = {omega[0]}")

    # Update the circular omega with the adiabatic solution
    for i in range(r.size):
        q[0] = r[i]
        p[1] = pphi[i]


        # omega[0] = omega_start
        # print(f"Ad. sol: i = {i}, omega[i] = {omega[i]}, omega_start = {omega_start}, r[i] = {r[i]}")
        #if i == 0:
        #print(f"Ad. sol: i = {i}, omega[i] = {omega[i]}, r[i] = {r[i]}")

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
          print(f"PROBLEM WE ARE EXTRAPOLATING!")
        #  omega[i] = om

        #omega[i] = omega_start

        #if i ==0:
        #  omega[i] = omega_start
        #print(f"Ad. sol: om = {om}")
        #print("---------------------------")


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
    #print(f"PA sol. : omega[0]  = {omega[0]}, omega_start = {omega_start}")
    #for i in range(r.size):
    #if i == 0:
    #  print(f"PA sol. : i = {i}, r = {r[i]}, pphi = {pphi[i]}, pr = {pr[i]}, omega = {omega[i]}")

    #    print(f"PA sol. : i = {i}, r = {r[i]}, pphi = {pphi[i]}, pr = {pr[i]}, omega = {omega[i]}")
    dt_dr = np.zeros(r.shape[0])
    dphi_dr = np.zeros(r.shape[0])
    cdef double dH_dpr,dH_dpphi,csi
    cdef double dyn[6]
    cdef double[::1] p_circ = np.zeros(2)

    cdef double[::1] tmp_LN = np.zeros(3)

    dyn_augm = []
    lN_dyn = []


    for i in range(r.shape[0]):
        q = np.array([r[i], 0])
        p = np.array([pr[i], pphi[i]])


        #if i <3:
        #  omega[i] = omega_start
        #  print(f"i = {i}, omega[i] = {omega[i]}, omega_start = {omega_start}, r[i] = {np.round(r[i],2)}")
        tmp = splines["everything"](omega[i])

        chi1_LN = tmp[0]
        chi2_LN = tmp[1]

        chi1_L = tmp[2]
        chi2_L = tmp[3]

        chi1_v = tmp[4:7]
        chi2_v = tmp[7:10]
        tmp_LN = tmp[10:13]

        ap = chi1_LN * X1 + chi2_LN * X2
        am = chi1_LN * X1 - chi2_LN * X2

        dSO_new = dSO_poly_fit(params.p_params.nu, ap, am)
        H.calibration_coeffs['dSO'] = dSO_new
        #print(f"i = {i}, dSO = {dSO_new}, chi1_v = {chi1_v}, chi2_v = {chi2_v}, m1 = {m_1}, m2 = {m_2}")
        #if i < 2:
        #  print(f"i = {i}, q[0] = {float(q[0])},  q[1] = {float(q[1])}, p[0] = {float(p[0])}, p[1] = {float(p[1])}")
        dyn[:] = H.dynamics(q, p, chi1_v, chi2_v, m_1, m_2,chi1_LN, chi2_LN, chi1_L, chi2_L)
        dH_dpr = dyn[2]
        dH_dpphi = dyn[3]
        csi = dyn[5]
        dt_dr[i] = 1 / (dH_dpr * csi)
        dphi_dr[i] = dH_dpphi / (dH_dpr * csi)



        omega[i] = dH_dpphi
        #if i ==0:
        #  print(f"After iteration. i = {i}, omega[i] = {omega[i]}, omega_start = {omega_start}, r[i] = {np.round(r[i],2)}")

        #print(f"i = {i}, chi1_LN = {chi1_LN}, chi2_LN = {chi2_LN}, chi1_L = {chi1_L}, chi2_L = {chi2_L}")
        #if i < 2:
        #print(f"After PA: i = {i}, omega[i] = {omega[i]}, r[i] = {r[i]}, dt_dr[i] = {dt_dr[i]}, dH_dpr = {dH_dpr}")
        #print(f"After PA: i = {i}, omega[i] = {omega[i]}, r[i] = {r[i]}, pphi[i] = {pphi[i]}, pr[i] = {pr[i]}")
        #print("====================================")
        H_val = dyn[4]
        p_circ[1] = p[1]
        omega_circ = H.omega(q, p_circ, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L)
        dyn_augm.append([H_val,omega[i],omega_circ,chi1_LN,chi2_LN])
        lN_dyn.append(tmp_LN)


    t = cumulative_integral(r, dt_dr)
    #print(f"t = {t}, omega = {np.array(omega)}, r = {r}, dt_dr = {dt_dr}")
    phi = cumulative_integral(r, dphi_dr)
    postadiabatic_dynamics = np.c_[t, r, phi, pr, pphi, dyn_augm]

    #print(f"pa_dyn =  {postadiabatic_dynamics}")
    lN_dyn = np.array(lN_dyn)

    return postadiabatic_dynamics, omega, lN_dyn, splines, omega_start_1



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

    cdef double omega_start_pa = omega_start

    try:

        postadiabatic_dynamics, omega_pa, tmp_LN_pa, splines, omega_start_pa = compute_postadiabatic_dynamics(#_stepback(
            omega_ref,
            omega_start,
            H,
            RR,
            chi_1,
            chi_2,
            m_1,
            m_2,
            splines,
            tol=tol,
            params=params,
            order=PA_order,
            postadiabatic_type=postadiabatic_type,
            window_length=10,
        )

        #print(f"postadiabatic_dynamics[:,0] = {postadiabatic_dynamics[:,0]}")

        #print(f"postadiabatic_dynamics[0,0] = {postadiabatic_dynamics[0,0]}, postadiabatic_dynamics[-1,0] = {postadiabatic_dynamics[-1,0]}, len(pa_dyn) = {len(postadiabatic_dynamics)}")
        #print(f"postadiabatic_dynamics[0] = {postadiabatic_dynamics[0]}")
        #print(f"postadiabatic_dynamics[-1] = {postadiabatic_dynamics[-1]}")
        PA_success = True
        e_id = -1
        ode_y_init = postadiabatic_dynamics[e_id, 1:5]
    except ValueError as e:
        PA_success = False
        ode_y_init = None
        omega_pa = [omega_start]
        omega_start_pa = omega_start

    omega_start = omega_pa[-1]
    tmp = splines["everything"](omega_start)

    #print(f"PA_success = {PA_success}, omega_start = {omega_start}, omega_pa[-1] = {omega_pa[-1]}, omegaPN_f = {omegaPN_f}")
    #print(f"ode_y_init = {ode_y_init}")
    cdef double X1 = params.p_params.X_1
    cdef double X2 = params.p_params.X_2
    cdef double chi1_LN_start = tmp[0]
    cdef double chi2_LN_start = tmp[1]
    cdef double chi1_L_start = tmp[2]
    cdef double chi2_L_start = tmp[3]
    chi1_v_start = tmp[4:7]
    chi2_v_start = tmp[7:10]
    lN_start = tmp[10:13]

    params.p_params.omega = omega_start
    params.p_params.chi1_v = chi1_v_start
    params.p_params.chi2_v = chi2_v_start

    params.p_params.chi_1, params.p_params.chi_2 = chi1_LN_start, chi2_LN_start
    params.p_params.chi1_L, params.p_params.chi2_L = chi1_L_start, chi2_L_start
    params.p_params.lN = lN_start

    params.p_params.update_spins(chi1_LN_start, chi2_LN_start)

    ap_start = chi1_LN_start * X1 + chi2_LN_start * X2
    am_start = chi1_LN_start * X1 - chi2_LN_start * X2

    dSO_start = dSO_poly_fit(params.p_params.nu, ap_start, am_start)
    H.calibration_coeffs["dSO"] = dSO_start
    #print(f"PA: chi1LN = {chi1_LN_start}, chi2LN = {chi2_LN_start}, chi1L = {chi1_L_start}, chi2L = {chi2_L_start}, chi1_v = {chi1_v_start}, chi2_v = {chi2_v_start}, dSO_start = {dSO_start}")


    #print(f"ode_y_init = {ode_y_init}")
    #print(f"PA_success = {PA_success}, omega_start = {omega_start}, omega_pa[-1] = {omega_pa[-1]}, omegaPN_f = {omegaPN_f}")
    (
        ode_dynamics_low,
        ode_dynamics_high,
        tmp_LN_low,
        tmp_LN_fine,
        dynamics,
        idx_restart,_,_
    ) = compute_dynamics_prec_opt(
        omega_ref,
        omega_pa[-1],
        omegaPN_f,
        H,
        RR,
        m_1,
        m_2,
        splines,
        params,
        rtol=1e-9,
        atol=1e-9,
        step_back=step_back,
        y_init=ode_y_init,
    )
    #print(f"t_low[-1] = {ode_dynamics_low[-1,0]}, r_low[-1] = {ode_dynamics_low[-1,1]}")
    if PA_success is True:

        # Interpolate the PA dynamics in its full range. Note that for the AS model this is only
        # done in a window before the ODE integration to smooth the timestep transition. For the
        # precessing model the twisting-up procedure, precisely the time-dependent rotation from the
        # co-precessing to the J-frame imposes a minimum timestep in LN such that the quaternion can be
        # accurately interpolated into the finer waveform grid

        t_pa = postadiabatic_dynamics[:, 0]
        #print(f"e_id = {e_id}, t_pa = {t_pa[e_id]}")

        ode_dynamics_low[:, 0] += t_pa[e_id]
        ode_dynamics_high[:, 0] += t_pa[e_id]

        #print(f"t_low = {ode_dynamics_low[:,0]}, r_low = {ode_dynamics_low[:,1]}")


        # This is the part of the PA dynamics which is smoothed in
        # timestep until it reaches the one obtained solving ODEs
        t_ode_low = ode_dynamics_low[:, 0]

        #print(f"t_ode_low[0] = {t_ode_low[0]}, t_ode_low[-1] = {t_ode_low[-1]}, r_low[0] = {ode_dynamics_low[0,1]}, r_low[-1] = {ode_dynamics_low[-1,1]}")

        dt_pa_first = t_pa[1] - t_pa[0]
        dt_ode_init = t_ode_low[1] - t_ode_low[0]

        # Estimate initial delta_t from the starting orbital frequency (this should correspond to the maximum timestep in the PA dynamics)
        dt0 = 2.*np.pi/omega_start_0

        #print(f"dt_pa_first = {dt_pa_first}, t_pa[0] = {t_pa[0]}, t_pa[1] = {t_pa[1]}, dt0 = {dt0}")

        while dt_pa_first>500:
            if  dt_pa_first > dt0:
              dt_pa_first = dt0/5.
            else:
              dt_pa_first = dt0/10.

        #print(f"dt_pa_first = {dt_pa_first}")
        t_pa_first = t_pa[0]
        t_ode_init = t_ode_low[0]

        #print(f"dt_ode_init = {dt_ode_init}, t_pa_first = {t_pa_first}, t_ode_init = {t_ode_init}, dt_pa_first = {dt_pa_first}")

        dt = dt_ode_init
        t = t_ode_init - dt
        t_new = []

        step_multiplier = 1.3
        dt_pa_first = 50.

        while True:
            t_new.append(t)

            if step_multiplier * dt < dt_pa_first:
                dt *= step_multiplier

            t -= dt

            if t < t_pa_first:
                #print(f"t = {t}, t_pa_first = {t_pa_first}.")
                #print(f"t_new = {t_new[::-1]}. Breaking loop.")
                break

        # Add the initial time
        t_new.append(t_pa[0])

        t_new = t_new[::-1]
        #iom = CubicSpline(t_pa,omega_pa)
        #omega_new = iom(t_new)
        #print(f"t_pa = {t_pa}")
        #print(f"t_new = {t_new}")
        # Interpolate window dynamics except the spins projections (otherwise bad things may happen due to the large timesteps of the PA dynamics)
        window_dynamics_interp = CubicSpline(t_pa, np.c_[postadiabatic_dynamics[:,:-2], omega_pa])
        #window_dynamics_interp = CubicSpline(omega_pa, np.c_[postadiabatic_dynamics[:,:-2]])

        #print(f"t_pa = {t_pa}, r_pa = {postadiabatic_dynamics[:,1]}")
        #ir = CubicSpline(omega_pa, postadiabatic_dynamics[:,1])
        #print(f"r_new = {ir(t_new)}")
        tmp_window = window_dynamics_interp(t_new)
        #tmp_window = window_dynamics_interp(omega_new)

        window_dynamics = tmp_window[:,:-1]
        omega_new = tmp_window[:,-1]

        tmp = splines["everything"](omega_new)
        chi1_LN_window = tmp[:,0]
        chi2_LN_window = tmp[:,1]
        lN_window = tmp[:,10:13]

        #postadiabatic_dynamics_v1 = np.c_[window_dynamics, omega_new, chi1_LN_window,chi2_LN_window]
        postadiabatic_dynamics_v1 = np.c_[window_dynamics, chi1_LN_window,chi2_LN_window]

        #print(f"t_pa[0] = {t_pa[0]}, postadiabatic_dynamics[0,0] = {postadiabatic_dynamics[0,0]}")
        #print(f"postadiabatic_dynamics[:,1] = {postadiabatic_dynamics[:,1]}")
        #print(f"postadiabatic_dynamics_v1[:,1] = {postadiabatic_dynamics_v1[:,1]}")
        #print(f"ode_dynamics_low[:,1] = {ode_dynamics_low[:,1]}")
        #print(f"postadiabatic_dynamics[0,1] = {postadiabatic_dynamics[0,1]}, postadiabatic_dynamics_v1[0,1] = {postadiabatic_dynamics_v1[0,1]}, ode_dynamics_low[0,1] = {ode_dynamics_low[0,1]}")
        #print(f"postadiabatic_dynamics[-1,1] = {postadiabatic_dynamics[-1,1]}, postadiabatic_dynamics_v1[-1,1] = {postadiabatic_dynamics_v1[-1,1]}, ode_dynamics_low[-1,1] = {ode_dynamics_low[-1,1]}")
        combined_dynamics = np.vstack((postadiabatic_dynamics_v1[:e_id], ode_dynamics_low)  )
        tmp_LN = np.vstack((lN_window[:e_id],  tmp_LN_low, tmp_LN_fine))

        len_window = len(lN_window[:e_id])
        #combined_low = combined_dynamics[:idx_restart+len_window]
        #combined_high = combined_dynamics[idx_restart+len_window:]

        #print(f"combined_low[:,1] = {combined_low[:,1]}")
        #print(f"combined_high[:,1] = {combined_high[:,1]}")
        #print(f"ode_dynamics_high[:,1] = {ode_dynamics_high[:,1]}")

        tmp_LN_low = tmp_LN[:idx_restart+len_window]
        tmp_LN_fine = tmp_LN[idx_restart+len_window:]

        #window_dynamics_interp = CubicSpline(t_pa, postadiabatic_dynamics_v1)
        #postadiabatic_dynamics_v1 = window_dynamics_interp(t_new)
        #combined_dynamics = np.vstack((postadiabatic_dynamics_v1[:e_id], ode_dynamics_low))

    else:
        combined_dynamics = ode_dynamics_low

    dynamics = np.vstack((combined_dynamics, ode_dynamics_high))

    return combined_dynamics, ode_dynamics_high,combined_t,combined_y,splines,dynamics, tmp_LN_low, tmp_LN_fine, omega_start_pa

@cython.profile(True)
@cython.linetrace(True)
cpdef cumulative_integral(
    x: np.array,
    y: np.array,
    order: int = 7,
):
    """
    Compute cumulative integral of y(x)..

    Args:
        x (np.array): Dependent variable.
        y (np.array): Independent variable.
        order (np.array): Order of the finite difference (3,5 or 7).

    Returns:
        (np.array) cumulative integral
    """
    cdef double h = x[1] - x[0]

    integral = np.zeros(x.size)

    if order == 3:
        for i in range(x.size - 1):
            if i == 0:
                z = np.sum(interpolated_integral_order_3[0] * y[:4])
            elif i == x.size - 2:
                z = np.sum(interpolated_integral_order_3[2] * y[-4:])
            else:
                z = np.sum(interpolated_integral_order_3[1] * y[i-1:i+3])

            integral[i+1] = integral[i] + z * h
    elif order == 5:
        for i in range(x.size - 1):
            if i == 0:
                z = np.sum(interpolated_integral_order_5[0] * y[:6])
            elif i == 1:
                z = np.sum(interpolated_integral_order_5[1] * y[:6])
            elif i == x.size - 3:
                z = np.sum(interpolated_integral_order_5[3] * y[-6:])
            elif i == x.size - 2:
                z = np.sum(interpolated_integral_order_5[4] * y[-6:])
            else:
                z = np.sum(interpolated_integral_order_5[2] * y[i-2:i+4])

            integral[i+1] = integral[i] + z * h
    elif order == 7:
        for i in range(x.size - 1):
            if i == 0:
                z = np.sum(interpolated_integral_order_7[0] * y[:8])
            elif i == 1:
                z = np.sum(interpolated_integral_order_7[1] * y[:8])
            elif i == 2:
                z = np.sum(interpolated_integral_order_7[2] * y[:8])
            elif i == x.size - 4:
                z = np.sum(interpolated_integral_order_7[4] * y[-8:])
            elif i == x.size - 3:
                z = np.sum(interpolated_integral_order_7[5] * y[-8:])
            elif i == x.size - 2:
                z = np.sum(interpolated_integral_order_7[6] * y[-8:])
            else:
                z = np.sum(interpolated_integral_order_7[3] * y[i-3:i+5])

            integral[i+1] = integral[i] + z * h

    return integral

@cython.profile(True)
@cython.linetrace(True)
def univariate_spline_integral(
    x: np.array,
    y: np.array,
) -> np.array:
    """
    Compute integral of y(x) using univariate spline.

    Args:
        x (np.array): Dependent variable.
        y (np.array): Independent variable.

    Returns:
        (np.array) integral
    """
    y_x_interp = InterpolatedUnivariateSpline(x[::-1], y[::-1])
    y_x_integral = y_x_interp.antiderivative()(x[::-1])[::-1]
    integral = y_x_integral - y_x_integral[0]

    return integral

@cython.profile(True)
@cython.linetrace(True)
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
    Compute the adiabatic parameter \dot{\Omega}/2\Omega^{2}.

    Args:
        dynamics (tuple): Dynamics array [t,r,phi,pr,pphi].
        H (Hamiltonian_v5PHM_C): Hamiltonian.
        chi_1 : Primary dimensionless spin vector.
        chi_2 : Secondary dimensionless spin vector.
        m_1 (double): Mass of the primary
        m_2 (double): Mass of the secondary
        params (EOBParams): Container with useful EOB parameters

    Returns:
        (np.array): Adiabatic parameter

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

    domega_dr = fin_diff_derivative(dynamics[:, 1], omega)

    adiabatic_param = dr_dt * domega_dr / (2 * omega * omega)

    return adiabatic_param
