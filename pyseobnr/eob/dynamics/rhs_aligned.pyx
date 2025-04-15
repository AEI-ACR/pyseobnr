# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False,
# cython: profile=False, linetrace=False, binding=True, initializedcheck=False
"""
Contains the actual RHS of the EOM, wrapped in cython.
This allows some of the cython functions used in the RHS to be called more efficiently.
"""
import cython

import numpy as np
cimport numpy as cnp

from pyseobnr.eob.utils.containers cimport EOBParams, qp_param_t
from pyseobnr.eob.waveform.waveform cimport RadiationReactionForce
from pyseobnr.eob.hamiltonian.Hamiltonian_C cimport Hamiltonian_C, Hamiltonian_C_dynamics_return_t


cpdef (double, double, double, double) get_rhs(
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

    cdef qp_param_t q = (z[0], z[1])
    cdef qp_param_t p = (z[2], z[3])

    cdef Hamiltonian_C_dynamics_return_t dynamics = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
    cdef double H_val = dynamics[4]
    cdef double omega = dynamics[3]

    params.dynamics.p_circ[1] = p[1]
    cdef double omega_circ = H.omega(q, params.dynamics.p_circ, chi_1, chi_2, m_1, m_2)

    cdef double xi = dynamics[5]

    cdef (double, double) RR_f = RR.RR(q, p, omega, omega_circ, H_val, params)

    return xi * dynamics[2], dynamics[3], -dynamics[0] * xi + RR_f[0], -dynamics[1] + RR_f[1]


cpdef augment_dynamics(
    const double[:, :] dynamics,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    Hamiltonian_C H
):
    """Compute dynamical quantities we need for the waveform

    Args:
        dynamics (np.ndarray): The dynamics array: t,r,phi,pr,pphi
    """

    assert dynamics.shape[1] > 4

    cdef:
        double H_val
        double omega
        double omega_c

    cdef Hamiltonian_C_dynamics_return_t dyn

    cdef:
        int i
        int N = dynamics.shape[0]

    cdef:
        const double* current_row_input

    cdef:
        qp_param_t q = (0, 0)
        qp_param_t p = (0, 0)
        qp_param_t p_c = (0, 0)

    cdef cnp.ndarray[
        cnp.double_t,
        ndim=2,
        mode="c",
        negative_indices=False] result = np.empty((N, 3))

    if dynamics.strides[0] != 1:
        for i in range(N):
            q[0] = dynamics[i, 1]
            q[1] = dynamics[i, 2]

            p[0] = dynamics[i, 3]
            p[1] = dynamics[i, 4]
            p_c[1] = p[1]

            # Evaluate a few things: H, omega, omega_circ
            dyn = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
            omega = dyn[3]
            H_val = dyn[4]

            omega_c = H.omega(q, p_c, chi_1, chi_2, m_1, m_2)

            result[i, 0] = H_val
            result[i, 1] = omega
            result[i, 2] = omega_c
    else:
        for i in range(N):
            current_row_input = &dynamics[i][0]
            q[0] = current_row_input[1]
            q[1] = current_row_input[2]

            p[0] = current_row_input[3]
            p[1] = current_row_input[4]
            p_c[1] = current_row_input[4]

            # Evaluate a few things: H, omega, omega_circ
            dyn = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
            omega = dyn[3]
            H_val = dyn[4]

            omega_c = H.omega(q, p_c, chi_1, chi_2, m_1, m_2)

            result[i, 0] = H_val
            result[i, 1] = omega
            result[i, 2] = omega_c

    return np.c_[dynamics, result]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.double_t, ndim=2, mode="c", negative_indices=False] compute_H_and_omega(
    double[:, :] dynamics,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    Hamiltonian_C H
):
    """
    Compute dynamical quantities needed for the waveforms.

    Simplified version of :py:func:`.augment_dynamics`
    when only :math:`\\Omega` is needed.

    Args:
        dynamics (2d memory view of type double):  Dynamical variables (t, r, phi, pr, pphi)
        chi_1 (double): Z-component of the dimensionless spin vector of the primary black hole
        chi_2 (double): Z-component of the dimensionless spin vector of the secondary black hole
        m_1 (double): Mass of the primary black hole
        m_2 (double): Mass of the secondary black hole
        H (Hamiltonian_C): Hamiltonian

    Returns:
        np.array: ``dynamics`` with an additional column containing omega.
    """

    assert dynamics.shape[1] > 4

    cdef Hamiltonian_C_dynamics_return_t dyn
    cdef:
        int i
        int N = dynamics.shape[0]

    cdef cnp.ndarray[
        cnp.double_t,
        ndim=2,
        mode="c",
        negative_indices=False] result = np.empty((N, 2), np.float64)

    for i in range(N):
        # Only evaluate omega
        dyn = H.dynamics(
            (dynamics[i, 1], dynamics[i, 2]),
            (dynamics[i, 3], dynamics[i, 4]),
            chi_1, chi_2, m_1, m_2)
        result[i, 0] = dyn[4]  # H_val
        result[i, 1] = dyn[3]  # omega

    return result
