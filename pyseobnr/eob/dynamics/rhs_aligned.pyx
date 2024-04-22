# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False,
# cython: profile=True, linetrace=True, binding=True, initializedcheck=False
"""
Contains the actual RHS of the EOM, wrapped in cython.
This allows some of the cython functions used in the RHS to be called more efficiently.
"""

import numpy as np
cimport numpy as np

from pyseobnr.eob.utils.containers cimport EOBParams
from pyseobnr.eob.waveform.waveform cimport RadiationReactionForce
from pyseobnr.eob.hamiltonian.Hamiltonian_C cimport Hamiltonian_C


cpdef get_rhs(
    double t,
    double[::1] z,
    Hamiltonian_C H,
    RadiationReactionForce RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    EOBParams params
):
    """
    Compute the RHS of the EOB evolution equations.
    In particular this function returns

    * :math:`\\dot{r}`,
    * :math:`\\dot{\\phi}`,
    * :math:`\\dot{p}_{r}`,
    * and :math:`\\dot{p}_{\\phi}`.

    See for example Eq(2) of [Buades2021]_ .

    The Hamiltonian is given by Eq(14) in [SEOBNRv5HM]_ doc
    and explicitly spelled out in Section I.C of [SEOBNRv5HM-theory]_
    and the RR force is described in Eq(43) of [SEOBNRv5HM]_ document, both
    contained in [DCC_T2300060]_ .

    """
    cdef double[::1] q = z[:2]
    cdef double[::1] p = z[2:]

    cdef double H_val, omega, omega_circ, xi
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


cpdef augment_dynamics(
    double[:, :] dynamics,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    Hamiltonian_C H
):
    """Compute dynamical quantities we need for the waveform

    Args:
        dynamics (np,ndarray): The dynamics array: t,r,phi,pr,pphi
    """
    result = []
    cdef double[:] p_c = np.zeros(2)
    cdef double H_val, omega, omega_c
    cdef double dyn[6]
    cdef double[:] q
    cdef double[:] p
    cdef double[:] row
    cdef int i, N
    N = dynamics.shape[0]

    for i in range(N):
        row = dynamics[i]
        q = row[1:3]
        p = row[3:5]
        p_c[1] = p[1]

        # Evaluate a few things: H, omega, omega_circ

        dyn[:] = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
        omega = dyn[3]
        H_val = dyn[4]

        omega_c = H.omega(q, p_c, chi_1, chi_2, m_1, m_2)

        result.append([H_val, omega, omega_c])
    result = np.array(result)
    return np.c_[dynamics, result]
