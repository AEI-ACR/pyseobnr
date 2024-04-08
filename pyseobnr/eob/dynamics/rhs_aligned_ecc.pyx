# cython: language_level=3, boundscheck=False, cdivision=True
# cython: wraparound=False, profile=True, linetrace=True
# cython: binding=True, initializedcheck=False

"""
Contains the actual RHS of the EOM, wrapped in cython.
This allows some of the cython functions used in the RHS to be called more efficiently.
"""

import numpy as np
cimport numpy as np

from pyseobnr.eob.utils.containers cimport EOBParams
from pyseobnr.eob.waveform.waveform_ecc cimport RadiationReactionForceEcc
from pyseobnr.eob.dynamics.Keplerian_evolution_equations_flags cimport BaseCoupledExpressionsCalculation
from pyseobnr.eob.dynamics.secular_evolution_equations_flags cimport (
    BaseCoupledExpressionsCalculation as BaseCoupledExpressionsCalculation_secular
)
from pyseobnr.eob.hamiltonian.Hamiltonian_C cimport Hamiltonian_C


cpdef get_rhs_ecc(
    double t,
    double[::1] z,
    Hamiltonian_C H,
    RadiationReactionForceEcc RR,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    EOBParams params,
):
    """
    Compute the RHS of the EOB evolution equations including
    the evolution of the Keplerian elements for eccentric models as described
    in [Gamboa2024]_ .

    Args:
        t (double): Time rescaled by total mass
        z (double[::1]): Dynamical variables (r, phi, pr, pphi, e, z)
        H (Hamiltonian_C): Hamiltonian
        RR (RadiationReactionForceEcc): Radiation reaction force with
            eccentric corrections
        chi_1 (double): z-component of the dimensionless spin vector of
            the primary black hole
        chi_2 (double): z-component of the dimensionless spin vector of
            the secondary black hole
        m_1 (double): Mass of the primary black hole
        m_2 (double): Mass of the secondary black hole
        params (EOBParams): Container with useful variables

    Return:
        np.array: (dHdr, dHdphi, dHdpr, dHdpphi, edot, zdot)
    """

    cdef double[::1] q = z[:2]
    cdef double[::1] p = z[2:4]
    cdef double[::1] Kep = z[4:]

    cdef double H_val, omega, xi
    cdef double dynamics[6]
    cdef double deriv[6]

    dynamics[:] = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)

    omega = dynamics[3]
    H_val = dynamics[4]
    xi = dynamics[5]

    params.p_params.omega = omega
    params.p_params.H_val = H_val

    cdef BaseCoupledExpressionsCalculation evolution = RR.evolution_equations
    evolution.compute(z=Kep[1], e=Kep[0], omega=omega)
    cdef double x_avg = evolution.get("xavg_omegainst")

    Kep = np.append(Kep, x_avg)
    params.ecc_params.x_avg = x_avg

    cdef (double, double) RR_f = RR.RR(q, p, Kep, omega, omega, H_val, params)

    deriv[:] = [
        xi * dynamics[2],
        dynamics[3],
        -dynamics[0] * xi + RR_f[0],
        -dynamics[1] + RR_f[1],
        evolution.get("edot"),
        evolution.get("zdot")
    ]

    return deriv


cpdef get_rhs_ecc_secular(
    double t,
    double[::1] Kep,
    RadiationReactionForceEcc RR,
):
    """
    Compute the RHS of the secular evolution equations for eccentricity,
    relativistic anomaly and dimensionless orbit-averaged orbital
    frequency x = (M omega_avg)^{2/3}.

    Args:
        t (double): Time rescaled by total mass
        Kep (double[::1]): Dynamical variables (e, z, x)
        RR (RadiationReactionForceEcc): Instance of thea rdiation reaction
            force with eccentric corrections. Should already be initialized
            for the secular evolution equations

    Returns:
        List: [de/dt, dz/dt, dx/dt]
    """

    cdef double deriv[3]
    cdef BaseCoupledExpressionsCalculation_secular evolution = RR.secular_evolution_equations

    evolution.compute(e=Kep[0], z=Kep[1], x=Kep[2])

    deriv[:] = [
        evolution.get("edot"),
        evolution.get("zdot"),
        evolution.get("xdot"),
    ]

    return deriv


cpdef compute_x(
    double[:] e,
    double[:] z,
    double[:] omega,
    RadiationReactionForceEcc RR,
):
    """
    Compute the PN formula for the dimensionless orbit-averaged orbital frequency 'x'
    from the eccentricity 'e', relativistic anomaly 'z', and angular velocity 'omega'.

    Args:
        e (1d array of double): Eccentricity array
        z (1d array of double): Relativistic anomaly array
        omega (1d array of double): Orbital angular velocity array
        RR (RadiationReactionForceEcc): Radiation reaction force with eccentric corrections

    Return:
        np.array: x as a one dimensional contiguous array

    Note:
        The input arrays have to be of the same size and one dimensional,
        they are not required to be contiguous.
    """

    assert (e.shape[0] == z.shape[0]) and (e.shape[0] == omega.shape[0]) and (e.shape[0] > 0)
    # As this is a memory view, some dimensions may exist but set to 0
    assert len([_ for _ in e.shape if _]) == 1
    assert len([_ for _ in z.shape if _]) == 1
    assert len([_ for _ in omega.shape if _]) == 1

    cdef int i, N
    cdef BaseCoupledExpressionsCalculation evolution = RR.evolution_equations

    N = e.shape[0]
    result = np.empty(N, np.float64)
    cdef double[::1] result_mem_view = result

    for i in range(N):
        # Evaluate x
        evolution.compute(e=e[i], z=z[i], omega=omega[i])
        result_mem_view[i] = evolution.get("xavg_omegainst")

    return result
