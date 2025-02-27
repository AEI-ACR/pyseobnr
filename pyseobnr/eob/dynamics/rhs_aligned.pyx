# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False,
# cython: profile=False, linetrace=False, binding=True, initializedcheck=False
"""
Contains the actual RHS of the EOM, wrapped in cython.
This allows some of the cython functions used in the RHS to be called more efficiently.
"""
import cython

import numpy as np
cimport numpy as cnp

from pyseobnr.eob.utils.containers cimport EOBParams
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

    # memview slicing is ok in cython
    cdef double[::1] q = z[0:2]
    cdef double[::1] p = z[2:4]

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
        # const double[:] row

        # those inputs should be compatible with the oe of the Hamiltonian calls
        # otherwise we get a conversion overhead
        double[2] p_c
        double[2] q
        double[2] p

        double[::1] q_view
        double[::1] p_view
        double[::1] pc_view

    cdef Hamiltonian_C_dynamics_return_t dyn

    cdef:
        int i
        int N = dynamics.shape[0]

    cdef:
        const double* current_row_input
        double* current_row_output

    p_c[0] = 0
    p_view = p
    q_view = q
    pc_view = p_c

    result = np.empty((N, 3))
    cdef double[:, ::1] result_view = result

    if dynamics.strides[0] != 1:
        for i in range(N):
            # row = dynamics[i]
            q[0] = dynamics[i, 1]
            q[1] = dynamics[i, 2]

            p[0] = dynamics[i, 3]
            p[1] = dynamics[i, 4]
            p_c[1] = p[1]

            # Evaluate a few things: H, omega, omega_circ
            dyn = H.dynamics(q_view, p_view, chi_1, chi_2, m_1, m_2)
            omega = dyn[3]
            H_val = dyn[4]

            omega_c = H.omega(q_view, pc_view, chi_1, chi_2, m_1, m_2)

            current_row_output = &result_view[i][0]
            current_row_output[0] = H_val
            current_row_output[1] = omega
            current_row_output[2] = omega_c
    else:
        for i in range(N):
            current_row_input = &dynamics[i][0]
            q[0] = current_row_input[1]
            q[1] = current_row_input[2]

            p[0] = current_row_input[3]
            p[1] = current_row_input[4]
            p_c[1] = current_row_input[4]

            # Evaluate a few things: H, omega, omega_circ
            dyn = H.dynamics(q_view, p_view, chi_1, chi_2, m_1, m_2)
            omega = dyn[3]
            H_val = dyn[4]

            omega_c = H.omega(q_view, pc_view, chi_1, chi_2, m_1, m_2)

            current_row_output = &result_view[i][0]
            current_row_output[0] = H_val
            current_row_output[1] = omega
            current_row_output[2] = omega_c

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
        dyn = H.dynamics(dynamics[i, 1:2], dynamics[i, 3:4], chi_1, chi_2, m_1, m_2)
        result[i, 0] = dyn[4]  # H_val
        result[i, 1] = dyn[3]  # omega

    return result
