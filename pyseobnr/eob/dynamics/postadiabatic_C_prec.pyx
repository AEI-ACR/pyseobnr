# cython: language_level=3
cimport cython
from typing import Dict
import numpy as np
from libc.math cimport log, sqrt, exp, abs,fabs, tgamma,sin,cos, tanh, sinh, asinh

from .initial_conditions_aligned_opt import computeIC_opt as computeIC


from pygsl import  roots, errno
from scipy import optimize
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from scipy import integrate

from lalinference.imrtgr import nrutils

from .integrate_ode import compute_dynamics_opt as compute_dynamics
from .integrate_ode import augment_dynamics
from pyseobnr.eob.utils.containers cimport EOBParams
from pyseobnr.eob.hamiltonian.Hamiltonian_v5PHM_C cimport Hamiltonian_v5PHM_C
from pyseobnr.eob.waveform.waveform cimport RadiationReactionForce

from .initial_conditions_aligned_precessing import computeIC_augm
from .pn_evolution_opt import compute_quasiprecessing_PNdynamics_opt
from .pn_evolution_opt import build_splines_PN_opt
from .integrate_ode_prec import compute_dynamics_prec_opt
from pyseobnr.eob.utils.utils_precession_opt import augment_dynamics_precessing_opt,compute_spins_EOBdyn_opt
from pyseobnr.eob.fits.fits_Hamiltonian import dSO

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
def Newtonian_j0(r):
    return np.sqrt(r)

@cython.profile(True)
@cython.linetrace(True)
cpdef double j0_eqn(double j0_sol, double r, Hamiltonian_v5PHM_C H, double[:] chi1_v, double[:] chi2_v, double m_1, double m_2,
double chi1_LN,double chi2_LN,double chi1_L,double chi2_L, EOBParams params,double[:] q, double[:] p):
    q[0] = r
    p[1]= j0_sol
    cdef double X1 = params.p_params.X_1
    cdef double X2 = params.p_params.X_2

    cdef double ap = chi1_LN * X1 + chi2_LN * X2
    cdef double am = chi1_LN * X1 - chi2_LN * X2

    cdef double dSO_new = dSO(params.p_params.nu, ap, am)

    H.calibration_coeffs['dSO'] = dSO_new
    cdef (double,double,double,double) dH_dq = H.grad(q, p, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN,
        chi1_L, chi2_L)
    cdef double dH_dr = dH_dq[0]
    return dH_dr

cpdef compute_adiabatic_solution(
    double[:] r,
    double[:] omega,
    Hamiltonian_v5PHM_C H,
    splines,
    double m_1,
    double m_2,
    double[::1] q,
    double[::1] p,
    EOBParams params,
    double tol=1e-12,
):
    cdef int i
    cdef double[:] j0 = Newtonian_j0(r)
    chi1_v = splines["chi1"](omega)
    chi2_v = splines["chi2"](omega)

    cdef double[:] chi1_LN = splines["chi1_LN"](omega)
    cdef double[:] chi2_LN = splines["chi2_LN"](omega)

    cdef double[:] chi1_L = splines["chi1_L"](omega)
    cdef double[:] chi2_L = splines["chi2_L"](omega)

    for i in range(r.shape[0]):
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
    double chi1_LN,double chi2_LN,double chi1_L,double chi2_L,
    double m_1,
    double m_2,
    EOBParams params,
    double[::1] q,
    double[::1] p
):
    q[0] = r
    p[0] = pr_sol[0]
    p[1] = pphi
    cdef double dH_dpr,H_val,csi,omega,omega_circ,result
    cdef double dynamics[6]

    cdef double X1 = params.p_params.X_1
    cdef double X2 = params.p_params.X_2

    cdef double ap = chi1_LN * X1 + chi2_LN * X2
    cdef double am = chi1_LN * X1 - chi2_LN * X2

    cdef double dSO_new = dSO(params.p_params.nu, ap, am)

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
    params.p_params.update_spins(chi1_LN, chi2_LN)
    cdef (double,double) flux = RR.RR(q, p, omega,omega_circ,H_val,params)


    result = dpphi_dr * dH_dpr * csi - flux[1]


    return result

'''
@cython.profile(True)
@cython.linetrace(True)
cpdef compute_pr(
    double[:] r,
    double[:] pr,
    double[:] pphi,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    double[:] q,
    double[:] p,
    double tol=1e-12,
    EOBParams params=None,
):
    # pphi_interp = CubicSpline(r[::-1], pphi[::-1])
    # dpphi_dr = pphi_interp.derivative()(r)

    # dr = np.abs(r[1]-r[0])
    # d_dr = findiff.FinDiff(0, dr, 1, acc=8)
    # dpphi_dr = - d_dr(pphi)

    cdef double[:] dpphi_dr = - fin_diff_derivative(r, pphi)

    # print(dpphi_dr)
    # print(dpphi_dr_new)
    # raise KeyboardInterrupt
    cdef int i
    for i in range(r.shape[0]):
        if np.abs(pr[i])<1e-14:
            x0 = 0.0
            x1 = -3e-2
        else:
            x0 = pr[i]*0.98
            x1 = pr[i]
        pr_solution = optimize.root_scalar(
            pr_eqn,
            x0=x0,
            x1=x1,
            args=(
                r[i], pphi[i], dpphi_dr[i],
                H, RR, chi_1, chi_2, m_1, m_2,
                params,q,p
            ),
            rtol=tol,
            xtol=tol,
            method="secant"
        )
        pr[i] = pr_solution.root


    return pr
'''

cpdef double pr_eqn_wrapper(pr_sol,args):
    return pr_eqn(pr_sol,*args)

@cython.profile(True)
@cython.linetrace(True)
cpdef compute_pr(
    double[:] r,
    double[:] pr,
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
    EOBParams params=None,
):
    # pphi_interp = CubicSpline(r[::-1], pphi[::-1])
    # dpphi_dr = pphi_interp.derivative()(r)

    # dr = np.abs(r[1]-r[0])
    # d_dr = findiff.FinDiff(0, dr, 1, acc=8)
    # dpphi_dr = - d_dr(pphi)

    cdef double[:] dpphi_dr = - fin_diff_derivative(r, pphi)

    # print(dpphi_dr)
    # print(dpphi_dr_new)
    # raise KeyboardInterrupt
    cdef int i
    chi1_v = splines["chi1"](omega)
    chi2_v = splines["chi2"](omega)

    cdef double[:] chi1_LN = splines["chi1_LN"](omega)
    cdef double[:] chi2_LN = splines["chi2_LN"](omega)

    cdef double[:] chi1_L = splines["chi1_L"](omega)
    cdef double[:] chi2_L = splines["chi2_L"](omega)
    for i in range(r.shape[0]):
        if np.abs(pr[i])<1e-14:
            x0 = 0.0
            x1 = -3e-2
        else:
            x0 = pr[i]*0.7
            x1 = pr[i]*1.3

        pr_solution = optimize.root(
            pr_eqn,
            pr[i],
            args=(
                r[i], pphi[i], dpphi_dr[i],
                H, RR, chi1_v[i], chi2_v[i],
                chi1_LN[i],chi2_LN[i],
                chi1_L[i],chi2_L[i],
                m_1, m_2,
                params,q,p
            ),
            tol=tol,

        )
        pr[i] = pr_solution.x
        '''
        pr_solution = optimize.root_scalar(
            pr_eqn,
            x0=x0,
            x1=x1,
            args=(
                r[i], pphi[i], dpphi_dr[i],
                H, RR, chi1_v[i], chi2_v[i],
                chi1_LN[i],chi2_LN[i],
                chi1_L[i],chi2_L[i],
                m_1, m_2,
                params,q,p
            ),
            rtol=tol,
            xtol=tol,
            method="secant"
        )
        pr[i] = pr_solution.root

        mysys = roots.gsl_function(pr_eqn_wrapper, (
                r[i], pphi[i], dpphi_dr[i],
                H, RR, chi1_v[i], chi2_v[i],
                chi1_LN[i],chi2_LN[i],
                chi1_L[i],chi2_L[i],
                m_1, m_2,
                params,q,p
            ))
        solver = roots.brent(mysys)
        #print(f"x0={x0},x1={x1},chi1_v={chi1_v[i]},chi2_v={chi2_v[i]}")
        solver.set(x1,x0)
        for iter in range(100):
            status = solver.iterate()
            x_lo = solver.x_lower()
            x_up = solver.x_upper()
            status = roots.test_interval(x_lo, x_up, tol,tol)
            result = solver.root()
            if status == errno.GSL_SUCCESS:
                break
        pr[i] = result
        '''
    return pr

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
    EOBParams params,
    double[::1] q,
    double[::1] p
):
    cdef double dH_dr,dH_dpr,H_val,csi,omega,omega_circ,result
    q[0] = r
    p[0] = pr
    p[1] =  pphi_sol[0]

    cdef double X1 = params.p_params.X_1
    cdef double X2 = params.p_params.X_2

    cdef double ap = chi1_LN * X1 + chi2_LN * X2
    cdef double am = chi1_LN * X1 - chi2_LN * X2

    cdef double dSO_new = dSO(params.p_params.nu, ap, am)

    H.calibration_coeffs['dSO'] = dSO_new
    cdef double dynamics[6]

    dynamics[:] = H.dynamics(q,p, chi1_v, chi2_v, m_1, m_2,chi1_LN,chi2_LN,chi1_L,chi2_L)
    #cdef (double,double,double,double) dH_dx = dynamics[:4]
    dH_dr = dynamics[0]
    dH_dpr = dynamics[2]
    H_val = dynamics[4]
    csi = dynamics[5]

    params.dynamics.p_circ[1] =  pphi_sol[0]
    omega_circ = H.omega(q,params.dynamics.p_circ,chi1_v,chi2_v,m_1,m_2,chi1_LN, chi2_LN,
        chi1_L, chi2_L)

    omega = dynamics[3]
    params.p_params.update_spins(chi1_LN, chi2_LN)


    cdef (double,double) flux = RR.RR(q, p, omega,omega_circ,H_val,params)



    result = dpr_dr * dH_dpr + dH_dr - flux[0] / csi

    return result

cpdef double pphi_eqn_wrapper(pphi_sol,args):
    return pphi_eqn(pphi_sol,*args)

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
    double[::1] q,
    double[::1] p,
    double tol=1e-12,
    EOBParams params=None
):
    # pr_interp = CubicSpline(r[::-1], pr[::-1])
    # dpr_dr = pr_interp.derivative()(r)

    cdef double[:] dpr_dr = - fin_diff_derivative(r, pr)
    cdef int i
    chi1_v = splines["chi1"](omega)
    chi2_v = splines["chi2"](omega)

    cdef double[:] chi1_LN = splines["chi1_LN"](omega)
    cdef double[:] chi2_LN = splines["chi2_LN"](omega)

    cdef double[:] chi1_L = splines["chi1_L"](omega)
    cdef double[:] chi2_L = splines["chi2_L"](omega)
    for i in range(r.shape[0]):
        pphi_solution = optimize.root(
            pphi_eqn,
            pphi[i],
            args=(r[i], pr[i], dpr_dr[i], H, RR, chi1_v[i], chi2_v[i],chi1_LN[i], chi2_LN[i],
                chi1_L[i], chi2_L[i], m_1, m_2, params,q,p),
            tol=tol,
        )
        pphi[i] = pphi_solution.x
        '''
        pphi_solution = optimize.root_scalar(
            pphi_eqn,
            x0=pphi[i],
            x1=0.98*pphi[i],
            args=(r[i], pr[i], dpr_dr[i], H, RR, chi1_v[i], chi2_v[i],chi1_LN[i], chi2_LN[i],
                chi1_L[i], chi2_L[i], m_1, m_2, params,q,p),
            rtol=tol,
            xtol=tol,
            method="secant"
        )
        pphi[i] = pphi_solution.root

        x0 = 0.8*pphi[i]
        x1 = 1.2*pphi[i]
        mysys = roots.gsl_function(pphi_eqn_wrapper, (
                r[i], pr[i], dpr_dr[i], H, RR, chi1_v[i], chi2_v[i],chi1_LN[i], chi2_LN[i],
                chi1_L[i], chi2_L[i], m_1, m_2, params,q,p
            ))
        solver = roots.brent(mysys)
        solver.set(x0,x1)
        for iter in range(100):
            status = solver.iterate()
            x_lo = solver.x_lower()
            x_up = solver.x_upper()
            status = roots.test_interval(x_lo, x_up, tol,tol)
            result = solver.root()
            if status == errno.GSL_SUCCESS:
                break
        pphi[i] = result
        '''
    return pphi

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
    EOBParams params=None,
):
    pr = np.zeros(r.size)
    cdef int i,n,parity
    cdef double tol_current
    cdef double X1 = params.p_params.X_1
    cdef double X2 = params.p_params.X_2
    for n in range(1, order+1):
        tol_current = 1e-3/10**n
        #print(tol_current)
        parity = n % 2
        if n>=7:
            tol_current=tol
        if parity:
            pr = compute_pr(
                r,
                pr,
                pphi,
                omega,
                H,
                RR,
                splines,
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
                omega,
                H,
                RR,
                splines,
                m_1,
                m_2,
                q,
                p,
                tol=tol_current,
                params=params,
            )
        chi1_v = splines["chi1"](omega)
        chi2_v = splines["chi2"](omega)

        chi1_LN = splines["chi1_LN"](omega)
        chi2_LN = splines["chi2_LN"](omega)

        chi1_L = splines["chi1_L"](omega)
        chi2_L = splines["chi2_L"](omega)
        ap = chi1_LN * X1 + chi2_LN * X2
        am = chi1_LN * X1 - chi2_LN * X2
        dSO_new = dSO(params.p_params.nu, ap, am)
        for i in range(r.size):
            q[0] = r[i]
            p[0] = pr[i]
            p[1] = pphi[i]

            H.calibration_coeffs['dSO'] = dSO_new[i]
            om = H.omega(
                q, p,
                chi1_v[i], chi2_v[i],
                m_1, m_2,
                chi1_LN[i], chi2_LN[i],
                chi1_L[i], chi2_L[i],
            )
            omega[i] = om

    return pr, pphi,omega

@cython.profile(True)
@cython.linetrace(True)
cpdef compute_postadiabatic_dynamics(
    double omega0,
    double omega_start,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    chi_1,
    chi_2,
    double m_1,
    double m_2,
    double tol=1e-12,
    EOBParams params=None,
    order=1
):
    """Compute the dynamics starting from omega0

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
    cdef double Mtot = m_1 + m_2
    cdef double nu = m_1*m_2/Mtot/Mtot

    combined_t, combined_y = compute_quasiprecessing_PNdynamics_opt(
        omega0,
        omega_start,
        m_1, m_2,
        chi_1, chi_2,
    )

    splines, chi_1, chi_2, chi1L_spline, chi2L_spline, chi1LN_spline, chi2LN_spline = build_splines_PN_opt(
        combined_t,
        combined_y,
        m_1, m_2,
        omega_start,
    )

    cdef double chi1_L =  chi1L_spline(omega_start)
    cdef double chi2_L =  chi2L_spline(omega_start)

    cdef double chi1_LN =  chi1LN_spline(omega_start)
    cdef double chi2_LN =  chi2LN_spline(omega_start)

    chi1_v = splines["chi1"](omega_start)
    chi2_v = splines["chi2"](omega_start)

    params.p_params.chi_1x, params.p_params.chi_1y, params.p_params.chi_1z = chi1_v
    params.p_params.chi_2x, params.p_params.chi_2y, params.p_params.chi_2z = chi2_v

    params.p_params.chi_1, params.p_params.chi_2 = chi1_LN, chi2_LN
    params.p_params.chi1_L, params.p_params.chi2_L = chi1_L, chi2_L

    params.p_params.update_spins(chi1_LN, chi2_LN)

    LNhat = splines["L_N"](omega_start)

    cdef double X1 = params.p_params.X_1
    cdef double X2 = params.p_params.X_2

    cdef double ap = chi1_LN * X1 + chi2_LN * X2
    cdef double am = chi1_LN * X1 - chi2_LN * X2
    cdef double chi_perp_eff = np.linalg.norm(chi1_v*X1+chi2_v*X2-ap*LNhat)

    cdef double dSO_new = dSO(params.p_params.nu, ap, am)
    H.calibration_coeffs['dSO'] = dSO_new


    cdef double r0
    r0, _, _ =computeIC_augm(omega_start, H, RR, chi1_v, chi2_v, chi1_LN, chi2_LN, m_1, m_2, params=params)
    cdef double chi_eff = ap
    cdef double r_final_prefactor = 2.7+chi_eff*(1-4.*nu)+chi_perp_eff
    #print(f"r_final_prefactor={r_final_prefactor}")
    r_ISCO, _ = Kerr_ISCO(chi1_LN, chi2_LN, m_1, m_2)
    cdef double r_final = max(10.0,r_final_prefactor * r_ISCO)
    cdef double r_switch_prefactor = 1.6
    cdef double r_switch = r_switch_prefactor * r_ISCO

    cdef double dr0 = 0.2
    cdef int r_size = int(np.ceil((r0 - r_final) / dr0))

    '''
    print(f"r0={r0}")
    print(f"r_ISCO={r_ISCO}")
    print(f"r_final={r_final}")
    print(f"r_size={r_size}")
    '''
    if r_size <= 4 or r0<=11.5:
        raise ValueError
    elif r_size < 10:
        r_size = 10

    r, _ = np.linspace(r0, r_final, num=r_size, endpoint=True, retstep=True)

    # First guess at circular omega, from Kepler's law
    omega = r**(-3./2)
    cdef double[::1] q = np.zeros(2)
    cdef double[::1] p = np.zeros(2)

    # Compute the adiabatic solution
    pphi = compute_adiabatic_solution(r, omega,H, splines, m_1, m_2,q,p,params, tol=tol,)
    # Update the circular omega with the adiabatic solution
    for i in range(r.size):
        q[0] = r[i]
        p[1] = pphi[i]

        chi1_v = splines["chi1"](omega[i])
        chi2_v = splines["chi2"](omega[i])

        chi1_LN = splines["chi1_LN"](omega[i])
        chi2_LN = splines["chi2_LN"](omega[i])

        chi1_L = splines["chi1_L"](omega[i])
        chi2_L = splines["chi2_L"](omega[i])

        dH_dq = H.grad(
            q, p,
            chi1_v, chi2_v,
            m_1, m_2,
            chi1_LN, chi2_LN,
            chi1_L, chi2_L,
        )
        omega[i] = dH_dq[-1]


    # Compute the PA solution
    pr, pphi,omega = compute_postadiabatic_solution(
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
        params=params,
    )
    #print("Here4")
    dt_dr = np.zeros(r.shape[0])
    dphi_dr = np.zeros(r.shape[0])
    cdef double dH_dpr,dH_dpphi,csi
    cdef double dyn[6]
    for i in range(r.shape[0]):
        q = np.array([r[i], 0])
        p = np.array([pr[i], pphi[i]])
        chi1_LN = chi1LN_spline(omega[i])
        chi2_LN = chi2LN_spline(omega[i])

        chi_1L = chi1L_spline(omega[i])
        chi_2L = chi2L_spline(omega[i])
        chi1_v = splines["chi1"](omega[i])
        chi2_v = splines["chi2"](omega[i])
        X1 = params.p_params.X_1
        X2 = params.p_params.X_2

        ap = chi1_LN * X1 + chi2_LN * X2
        am = chi1_LN * X1 - chi2_LN * X2

        dSO_new = dSO(params.p_params.nu, ap, am)

        H.calibration_coeffs['dSO'] = dSO_new
        dyn[:] = H.dynamics(q, p, chi1_v, chi2_v, m_1, m_2,chi1_LN, chi2_LN, chi_1L, chi_2L)
        dH_dpr = dyn[2]
        dH_dpphi = dyn[3]


        csi = dyn[5]
        dt_dr[i] = 1 / (dH_dpr * csi)
        dphi_dr[i] = dH_dpphi / (dH_dpr * csi)


    t = cumulative_integral(r, dt_dr)
    # t = univariate_spline_integral(r, dt_dr)

    phi = cumulative_integral(r, dphi_dr)
    # phi = univariate_spline_integral(r, dphi_dr)

    postadiabatic_dynamics = np.c_[t, r, phi, pr, pphi]

    '''
    adiabatic_param = compute_adiabatic_parameter(
        postadiabatic_dynamics,
        H,
        chi_1,
        chi_2,
        m_1,
        m_2,
        params,
    )
    '''
    ap_threshold = 6e-3

    # postadiabatic_dynamics = postadiabatic_dynamics[
    #     np.where(np.abs(adiabatic_param) < ap_threshold)
    # ]
    # postadiabatic_dynamics = postadiabatic_dynamics[postadiabatic_dynamics[:, 1] >= r_switch]
    # postadiabatic_dynamics = postadiabatic_dynamics[:-2]
    return postadiabatic_dynamics,combined_t,combined_y,chi1L_spline,chi2L_spline,chi1LN_spline,chi2LN_spline,splines,omega


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef compute_combined_dynamics(
    double omega_ref,
    double omega0,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    double m_1,
    double m_2,
    chi_1,
    chi_2,
    ODE_system_RHS,
    double tol=1e-12,
    EOBParams params=None,
    double step_back=50,
    str backend="ode",
    int PA_order=8
):
    try:
        postadiabatic_dynamics,combined_t,combined_y,chi1L_spline,chi2L_spline,chi1LN_spline,chi2LN_spline,splines, omega= compute_postadiabatic_dynamics(
            omega_ref,
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
        e_id = -1
        ode_y_init = postadiabatic_dynamics[e_id, 1:]
        #dr = np.diff(postadiabatic_dynamics[:,1])
        #idx = np.where(dr>=0)
        #print(f"bad indices: {idx}")
        #print(f"PA dynamics at these indices: {postadiabatic_dynamics[idx]}")
        #print(f"omega[0] = {omega[0]},omega[-1]={omega[-1]}")
    except ValueError as e:
        #print(str(e))
        PA_success = False
        ode_y_init = None


    #print("PA_success: ", PA_success)
    #print("ode_y_init: ", ode_y_init)
    omegaPN_f = combined_y[:, -1][-1]
    (
        ode_dynamics_low,
        ode_dynamics_high,
        chi1L_spline,
        chi2L_spline,
        chi1LN_spline,
        chi2LN_spline,
        splines,
        dynamics,
    ) = compute_dynamics_prec_opt(
        omega[-1],
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
        rtol=1e-9,
        atol=1e-9,
        params=params,
        step_back=step_back,
        y_init=ode_y_init,
    )
    if PA_success is True:

        (
        chi1L_EOB_low,
        chi2L_EOB_low,
        chi1LN_EOB_low,
        chi2LN_EOB_low,
        chi1v_EOB_low,
        chi2v_EOB_low,
        ) = compute_spins_EOBdyn_opt(postadiabatic_dynamics, splines)


        postadiabatic_dynamics = augment_dynamics_precessing_opt(
            postadiabatic_dynamics,
            chi1L_EOB_low,
            chi2L_EOB_low,
            chi1LN_EOB_low,
            chi2LN_EOB_low,
            chi1v_EOB_low,
            chi2v_EOB_low,
            m_1,
            m_2,
            omegaPN_f,
            H,
        )
        #print(f"last point of PA is at t={postadiabatic_dynamics[-1,0]}")

        ode_dynamics_low[:, 0] += postadiabatic_dynamics[e_id, 0]
        ode_dynamics_high[:, 0] += postadiabatic_dynamics[e_id, 0]
        #print(f"postadiabatic_dynamics shape is {postadiabatic_dynamics.shape}")
        #print(f"ode_dynamics_low shape is {ode_dynamics_low.shape}")

        combined_dynamics = np.vstack(
            (postadiabatic_dynamics[:e_id], ode_dynamics_low)
        )
        #dr = np.diff(combined_dynamics[:,e_id])
        #idx = np.where(dr>=0)
        #print(f"bad indices: {idx}")
        #print(f"combined dynamics at these indices: {combined_dynamics[idx]}")
    else:
        combined_dynamics = ode_dynamics_low

    dynamics = np.vstack((combined_dynamics, ode_dynamics_high))
    return combined_dynamics, ode_dynamics_high,combined_t,combined_y,chi1L_spline,chi2L_spline,chi1LN_spline,chi2LN_spline,splines,dynamics

# def cumulative_integral_old(
#     x: np.array,
#     y: np.array,
# ) -> np.array:
#     oo12 = 1. / 12.

#     x_ext = np.r_[x[3], x, x[-4]]
#     y_ext = np.r_[y[3], y, y[-4]]

#     X0 = x_ext[0:]
#     X1 = x_ext[1:]
#     X2 = x_ext[2:]
#     X3 = x_ext[3:]

#     Y0 = y_ext[0:]
#     Y1 = y_ext[1:]
#     Y2 = y_ext[2:]
#     Y3 = y_ext[3:]

#     integral = np.zeros(x.size)

#     for i in range(x.size - 1):
#         a = X1[i] - X0[i]
#         b = X2[i] - X1[i]
#         c = X3[i] - X2[i]
#         d = Y1[i] - Y0[i]
#         e = Y2[i] - Y1[i]
#         h = Y3[i] - Y2[i]
#         g = 0.5 * (Y1[i] + Y2[i])

#         z = (
#             b * g
#             + oo12 * b * b * (
#                 c * b * (2 * c + b) * (c + b) * d
#                 - a * c * (c - a) * (2 * c + 2 * a + 3 * b) * e
#                 - a * b * (2 * a + b) * (a + b) * h
#             ) / (a * c * (a + b) * (c + b) * (c + a + b))
#         )

#         integral[i+1] = integral[i] + z

#     return integral

@cython.profile(True)
@cython.linetrace(True)
cpdef cumulative_integral(
    x: np.array,
    y: np.array,
    order: int = 7,
):
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
    Compute the adiabatic parameter \dot{\Omega}/2\Omega^{2}
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
