# cython: language_level=3, boundscheck=False, cdivision=True, profile=True, linetrace=True,initializedcheck=False
"""
Contains the actual RHS of the EOM, wrapped in cython. 
This allows some of the cython functions used in the RHS to be called more efficiently.
"""

cimport cython
from pyseobnr.eob.utils.containers cimport EOBParams,FluxParams
from pyseobnr.eob.waveform.waveform cimport RR_force,  RadiationReactionForce
from pyseobnr.eob.hamiltonian.Hamiltonian_v5PHM_C cimport Hamiltonian_v5PHM_C
import numpy as np
cimport numpy as np
from pyseobnr.eob.fits.fits_Hamiltonian import dSO as dSO_poly_fit


DTYPE = np.float64

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float64_t DTYPE_t

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)

cpdef get_rhs_prec(double t,double[::1] z,Hamiltonian_v5PHM_C H,RadiationReactionForce RR,
    double m_1,double m_2,EOBParams params):
    """
    Compute the RHS of the EOB evolution equations.
    In particular this function returns
    \dot{r},\dot{\phi},\dot{p}_{r},\dot{p}_{\phi}
    """
    cdef double[::1] q = z[:2]
    cdef double[::1] p = z[2:]

    cdef double chi1_LN = params.p_params.chi_1
    cdef double chi2_LN = params.p_params.chi_2

    cdef double chi1_L = params.p_params.chi1_L
    cdef double chi2_L = params.p_params.chi2_L

    cdef double H_val,omega,omega_circ,xi
    cdef double dynamics[6]
    dynamics[:] = H.dynamics(q,
        p,
        params.p_params.chi1_v,
        params.p_params.chi2_v,
        m_1,
        m_2,
        chi1_LN,
        chi2_LN,
        chi1_L,
        chi2_L,)
    H_val = dynamics[4]
    omega = dynamics[3]
    params.dynamics.p_circ[1] = p[1]

    omega_circ = H.omega(q,
        params.dynamics.p_circ,
        params.p_params.chi1_v,
        params.p_params.chi2_v,
        m_1,
        m_2,
        chi1_LN,
        chi2_LN,
        chi1_L,
        chi2_L,)

    xi = dynamics[5]

    params.p_params.omega_circ = omega_circ
    params.p_params.omega = omega
    params.p_params.H_val = H_val



    cdef (double, double) RR_f = RR.RR(q, p, omega, omega_circ, H_val, params)

    cdef double deriv[4]
    deriv[:] = [xi * dynamics[2], dynamics[3], -dynamics[0] * xi + RR_f[0], -dynamics[1] + RR_f[1]]
    return deriv
