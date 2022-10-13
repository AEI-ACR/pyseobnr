# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, profile=True, linetrace=True,initializedcheck=False

cimport cython
from pyseobnr.eob.utils.containers cimport EOBParams,FluxParams
from pyseobnr.eob.waveform.waveform cimport RR_force,  RadiationReactionForce
from pyseobnr.eob.hamiltonian.Hamiltonian_C cimport Hamiltonian_C


cpdef get_rhs(double t,double[::1] z,Hamiltonian_C H,RadiationReactionForce RR,
    double chi_1,double chi_2,double m_1,double m_2,EOBParams params):
    """
    Compute the RHS of the EOB evolution equations.
    In particular this function returns
    \dot{r},\dot{\phi},\dot{p}_{r},\dot{p}_{\phi}
    """
    cdef double[::1] q = z[:2]
    cdef double[::1] p = z[2:]

    cdef double H_val,omega,omega_circ,xi
    cdef double dynamics[6]
    dynamics[:] = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
    H_val = dynamics[4]
    omega = dynamics[3]
    params.dynamics.p_circ[1] = p[1]

    omega_circ = H.omega(q, params.dynamics.p_circ, chi_1, chi_2, m_1, m_2)

    xi = dynamics[5]

    cdef (double, double) RR_f = RR.RR(q, p, omega, omega_circ, H_val, params)

    cdef double deriv[4]
    deriv[:] = [xi * dynamics[2], dynamics[3], -dynamics[0] * xi + RR_f[0], -dynamics[1] + RR_f[1]]
    return deriv