
# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
# cython: profile=False, linetrace=False, cpow=True

"""
Additional utility functions to compute the dynamical tides due to f-mode resonance,
which directly influence the dynamics, as well as the PN contributions
"""

from libc.math cimport cos, sin, sqrt, M_PI

import numpy as np
from scipy.special.cython_special cimport fresnel as c_fresnel
from scipy.special import fresnel
from pyseobnr.eob.utils.containers cimport EOBParams


# Perturbative expansion of f(x) around the removable singularity at x = 1
f_keffresonant_pert = (
    np.poly1d(
        [
            -0.007120198902606311,
            0.01122256515775034,
            -0.006944444444444445,
            -0.1157407407407407,
            -0.0833333333333333
        ]
    )
)
# Perturbative expansion of f'(x) around the removable singularity at x = 1
df_keffresonant_pert = (
    np.poly1d(
        [
            0.01919764200071127,
            -0.02848079561042524,
            0.03366769547325102,
            -0.01388888888888889,
            -0.1157407407407407
        ]
    )
)
# Perturbative expansion of f''(x) around the removable singularity at x = 1
# ddf_keffresonant_pert = np.poly1d([0.01919764200071127*4, -0.02848079561042524*3,
#                                    0.03366769547325102*2, -0.01388888888888889])

cdef double pi = M_PI
cdef double _1_sqrt_2pi = 0.3989422804014326779399460599
cdef double _sqrt_pi_3 = sqrt(pi/(3))
cdef double four_sqrt_three_pi = 4./sqrt(3*pi)
cdef double prefactnine = 35840/(27.*sqrt(3*pi))
cdef double prefactfive = 64/(3.*sqrt(3*pi))
cdef double prefact = four_sqrt_three_pi

# =================================================================
#           FUNCTIONS TO INCLUDE DYNAMICAL TIDES AMPLIFICATION
# =================================================================

cpdef double tidal_contribution(EOBParams EOBpars,
                                double r,
                                int num):
    """
    Function to compute the quadrupolar tidal contribution to the radial potential of the Hamiltonian
    in an effective description due to the presence of f-mode resonance.
    Implements the upper line of eq. (2) of http://arxiv.org/abs/1812.08643
    Args:
        EOBpars (class): A PhysicalParams class as implemented in containers.pyx
        r (double) : the M-rescaled (dimensionless) EOB seperation, i.e. the seperation in dimensions of total
            mass r/M
        num (int in (1, 2)) : specification wether we are interested in the primary's or secondary's
            contribution

    Returns:
        k_eff:  enhancement of kltidal
    """
    # print(r)
    return quadru_tidal_contribution(EOBpars, r, num) + octu_tidal_contribution(EOBpars, r, num)

cpdef (double, double) tidal_and_d_tidal_contribution(EOBParams EOBpars,
                                                      double r,
                                                      int num):
    """
    Function to compute the r-derivative of the quadrupolar tidal contribution to the radial potential
    of the Hamiltonian in an effective description due to the presence of f-mode resonance AND the quadrupolar
    tidal contribution itself for efficiency
    Implements the upper line of eq. (2) of http://arxiv.org/abs/1812.08643
    Args:
        EOBpars (class): A PhysicalParams class as implemented in containers.pyx
        r (double) : the M-rescaled (dimensionless) EOB seperation, i.e. the seperation in dimensions of total
            mass r/M
        num (int in (1, 2)) : specification wether we are interested in the primary's or secondary's
            contribution

    Returns:
        [keff, d_k_eff]:  pure enhancement of and r-derivative of kltidal
    """
    cdef (double, double) quadrus, octus

    quadrus = quadru_and_d_quadru_tidal_contribution(EOBpars, r, num)
    octus = octu_and_d_octu_tidal_contribution(EOBpars, r, num)
    return (quadrus[0] + octus[0], quadrus[1] + octus[1])

cpdef (double, double, double) tidal_and_d_tidal_and_d2_tidal_contribution(EOBParams EOBpars,
                                                                           double r,
                                                                           int num):
    """
    Function to compute the rr-derivative of the quadrupolar tidal contribution to the radial potential
    of the Hamiltonian in an effective description due to the presence of f-mode resonance AND the
        r-derivative AND
    the quadrupolar tidal contribution itself for efficiency
    Implements the upper line of eq. (2) of http://arxiv.org/abs/1812.08643
    Args:
        EOBpars (class): A PhysicalParams class as implemented in containers.pyx
        r (double) : the M-rescaled (dimensionless) EOB seperation, i.e. the seperation in dimensions of total
            mass r/M
        num (int in (1, 2)) : specification wether we are interested in the primary's or secondary's
            contribution

    Returns:
        [dd_k_eff, d_k_eff, k_eff]:  rr-derivative of, r-derivative of, and pure enhancement of kltidal
    """
    cdef (double, double, double) quadrus, octus

    quadrus = quadru_and_d_quadru_and_d2_quadru_tidal_contribution(EOBpars, r, num)
    octus = octu_and_d_octu_and_d2_octu_tidal_contribution(EOBpars, r, num)
    return (quadrus[0] + octus[0], quadrus[1] + octus[1], quadrus[2] + octus[2])

cpdef double quadru_tidal_contribution(EOBParams EOBpars,
                                       double r,
                                       int num):
    """
    Function to compute the quadrupolar tidal contribution to the radial potential of the Hamiltonian
    in an effective description due to the presence of f-mode resonance.
    Implements the upper line of eq. (2) of http://arxiv.org/abs/1812.08643
    Args:
        EOBpars (class): A PhysicalParams class as implemented in containers.pyx
        r (double) : the M-rescaled (dimensionless) EOB seperation, i.e. the seperation in dimensions of total
            mass r/M
        num (int in (1, 2)) : specification wether we are interested in the primary's or secondary's
            contribution

    Returns:
        k_eff:  enhancement of kltidal
    """
    cdef double X_this, X_other, lambda2Tidal, omega02, sqrteps2, spinshift

    if num == 1:
        X_this = EOBpars.p_params.X_1
        X_other = EOBpars.p_params.X_2
        lambda2Tidal = EOBpars.tidal_params.lambda2Tidal1
        omega02 = EOBpars.tidal_params.omega02Tidal1
        sqrteps2 = EOBpars.tidal_params.sqrtepsilon2Tidal1
        spinshift = EOBpars.tidal_params.spinshiftomega02Tidal1
    elif num == 2:
        X_this = EOBpars.p_params.X_2
        X_other = EOBpars.p_params.X_1
        lambda2Tidal = EOBpars.tidal_params.lambda2Tidal2
        omega02 = EOBpars.tidal_params.omega02Tidal2
        sqrteps2 = EOBpars.tidal_params.sqrtepsilon2Tidal2
        spinshift = EOBpars.tidal_params.spinshiftomega02Tidal2
    else:
        raise ValueError("num must be either 1 or 2 to specify the object we are investigating!")
    if lambda2Tidal == 0.0:
        return 0.
    return (
        -3.*lambda2Tidal*k_eff_amplification(
            r, omega02, spinshift, sqrteps2, 2
        )*X_other/X_this/r**6*(
            1.+5./2.*X_this/r+(337./28.*X_this**2+X_this/8.+3.)/r**2
        )
    )

cpdef (double, double) quadru_and_d_quadru_tidal_contribution(EOBParams EOBpars,
                                                              double r,
                                                              int num):
    """
    Function to compute the r-derivative of the quadrupolar tidal contribution to the radial potential
    of the Hamiltonian in an effective description due to the presence of f-mode resonance AND the quadrupolar
    tidal contribution itself for efficiency
    Implements the upper line of eq. (2) of http://arxiv.org/abs/1812.08643
    Args:
        EOBpars (class): A PhysicalParams class as implemented in containers.pyx
        r (double) : the M-rescaled (dimensionless) EOB seperation, i.e. the seperation in dimensions of total
            mass r/M
        num (int in (1, 2)) : specification wether we are interested in the primary's or secondary's
            contribution

    Returns:
        [d_keff, k_eff]:  r-derivative of, and pure enhancement of kltidal
    """
    cdef double X_this, X_other, lambda2Tidal, omega02, sqrteps2, spinshift

    if num == 1:
        X_this = EOBpars.p_params.X_1
        X_other = EOBpars.p_params.X_2
        lambda2Tidal = EOBpars.tidal_params.lambda2Tidal1
        omega02 = EOBpars.tidal_params.omega02Tidal1
        sqrteps2 = EOBpars.tidal_params.sqrtepsilon2Tidal1
        spinshift = EOBpars.tidal_params.spinshiftomega02Tidal1
    else:
        X_this = EOBpars.p_params.X_2
        X_other = EOBpars.p_params.X_1
        lambda2Tidal = EOBpars.tidal_params.lambda2Tidal2
        omega02 = EOBpars.tidal_params.omega02Tidal2
        sqrteps2 = EOBpars.tidal_params.sqrtepsilon2Tidal2
        spinshift = EOBpars.tidal_params.spinshiftomega02Tidal2
    if lambda2Tidal == 0.0:
        return (0., 0.)

    cdef double k_eff, dk_eff, r2, NNLO_terms, PN_terms, A_dyn_tides, dA_dyn_tides_dr

    k_eff, dk_eff = k_eff_amplification_and_dk_eff_amplification(r, omega02, spinshift, sqrteps2, 2)
    r2 = r*r

    NNLO_terms = 337./28.*X_this**2 + X_this/8. + 3.
    PN_terms = (1.+5./2.*X_this/r+NNLO_terms/r2)
    A_dyn_tides = -3.*lambda2Tidal*k_eff*X_other/X_this/r**6 * PN_terms
    dA_dyn_tides_dr = (
        (dk_eff/k_eff - 6./r - (5./2.*X_this/r2 + 2.*NNLO_terms / r**3) / PN_terms) * A_dyn_tides
    )
    return (A_dyn_tides, dA_dyn_tides_dr)

cpdef (double, double, double) quadru_and_d_quadru_and_d2_quadru_tidal_contribution(EOBParams EOBpars,
                                                                                    double r,
                                                                                    int num):
    """
    Function to compute the rr-derivative of the quadrupolar tidal contribution to the radial potential
    of the Hamiltonian in an effective description due to the presence of f-mode resonance AND the
        r-derivative AND
    the quadrupolar tidal contribution itself for efficiency
    Implements the upper line of eq. (2) of http://arxiv.org/abs/1812.08643
    Args:
        EOBpars (class): A PhysicalParams class as implemented in containers.pyx
        r (double) : the M-rescaled (dimensionless) EOB seperation, i.e. the seperation in dimensions of total
            mass r/M
        num (int in (1, 2)) : specification wether we are interested in the primary's or secondary's
            contribution

    Returns:
        [dd_k_eff, d_k_eff, k_eff]:  rr-derivative of, r-derivative of, and pure enhancement of kltidal
    """
    cdef double X_this, X_other, lambda2Tidal, omega02, sqrteps2, spinshift

    if num == 1:
        X_this = EOBpars.p_params.X_1
        X_other = EOBpars.p_params.X_2
        lambda2Tidal = EOBpars.tidal_params.lambda2Tidal1
        omega02 = EOBpars.tidal_params.omega02Tidal1
        sqrteps2 = EOBpars.tidal_params.sqrtepsilon2Tidal1
        spinshift = EOBpars.tidal_params.spinshiftomega02Tidal1
    else:
        X_this = EOBpars.p_params.X_2
        X_other = EOBpars.p_params.X_1
        lambda2Tidal = EOBpars.tidal_params.lambda2Tidal2
        omega02 = EOBpars.tidal_params.omega02Tidal2
        sqrteps2 = EOBpars.tidal_params.sqrtepsilon2Tidal2
        spinshift = EOBpars.tidal_params.spinshiftomega02Tidal2
    if lambda2Tidal == 0.0:
        return (0., 0., 0.)

    cdef double k_eff, dk_eff, r2, NNLO_terms, PN_terms, A_dyn_tides, dA_dyn_tides_dr
    cdef double d2A_dyn_tides_dr2, deriv_terms

    k_eff, dk_eff, ddk_eff = (
        k_eff_amplification_and_dk_eff_amplification_and_ddk_eff_amplifcation(
            r, omega02, spinshift, sqrteps2, 2
        )
    )
    r2 = r*r

    NNLO_terms = 337./28.*X_this**2 + X_this/8. + 3.
    PN_terms = (1.+5./2.*X_this/r+NNLO_terms/r2)
    A_dyn_tides = -3.*lambda2Tidal*k_eff*X_other/X_this/r**6 * PN_terms
    deriv_terms = (dk_eff/k_eff - 6./r - (5./2.*X_this/r2 + 2.*NNLO_terms / r**3) / PN_terms)
    dA_dyn_tides_dr = deriv_terms * A_dyn_tides
    d2A_dyn_tides_dr2 = (
        (
            deriv_terms**2
            + ddk_eff/k_eff
            - (dk_eff/k_eff)**2
            + (6. + (5.*X_this/r + 6.*NNLO_terms/r2) / PN_terms) /r2
            - (5./2.*X_this/r2 + 2.*NNLO_terms / r**3)**2
            / PN_terms**2
        )
        * A_dyn_tides
    )
    return (A_dyn_tides, dA_dyn_tides_dr, d2A_dyn_tides_dr2)


cpdef double octu_tidal_contribution(EOBParams EOBpars,
                                     double r,
                                     int num):
    """
    Function to compute the octupolar tidal contribution to the radial potential of the Hamiltonian
    in an effective description due to the presence of f-mode resonance.
    Implements the lower line of eq. (2) of http://arxiv.org/abs/1812.08643
    Args:
        EOBpars (class): A PhysicalParams class as implemented in containers.pyx
        r (double) : the M-rescaled (dimensionless) EOB seperation, i.e. the seperation in dimensions of total
            mass r/M
        num (int in (1, 2)) : specification wether we are interested in the primary's or secondary's
            contribution

    Returns:
        k_eff:  enhancement of kltidal
    """
    cdef double X_this, X_other, lambda3Tidal, omega03, sqrteps3, spinshift

    if num == 1:
        X_this = EOBpars.p_params.X_1
        X_other = EOBpars.p_params.X_2
        lambda3Tidal = EOBpars.tidal_params.lambda3Tidal1
        omega03 = EOBpars.tidal_params.omega03Tidal1
        sqrteps3 = EOBpars.tidal_params.sqrtepsilon3Tidal1
        spinshift = EOBpars.tidal_params.spinshiftomega03Tidal1
    elif num == 2:
        X_this = EOBpars.p_params.X_2
        X_other = EOBpars.p_params.X_1
        lambda3Tidal = EOBpars.tidal_params.lambda3Tidal2
        omega03 = EOBpars.tidal_params.omega03Tidal2
        sqrteps3 = EOBpars.tidal_params.sqrtepsilon3Tidal2
        spinshift = EOBpars.tidal_params.spinshiftomega03Tidal2
    else:
        raise ValueError("num must be either 1 or 2 to specify the object we are investigating!")
    if lambda3Tidal == 0.0:
        return 0.
    return (
        -15.*lambda3Tidal*k_eff_amplification(
            r, omega03, spinshift, sqrteps3, 3
        )*X_other/X_this/r**8*(
            1.+(-2. + 15./2.*X_this)/r+ (8./3. - 311./24.*X_this + 110.*X_this**2/3.)/r**2
        )
    )

cpdef (double, double) octu_and_d_octu_tidal_contribution(EOBParams EOBpars,
                                                          double r,
                                                          int num):
    """
    Function to compute the r-derivative of the quadrupolar tidal contribution to the radial potential
    of the Hamiltonian in an effective description due to the presence of f-mode resonance AND the quadrupolar
    tidal contribution itself for efficiency
    Implements the upper line of eq. (2) of http://arxiv.org/abs/1812.08643
    Args:
        EOBpars (class): A PhysicalParams class as implemented in containers.pyx
        r (double) : the M-rescaled (dimensionless) EOB seperation, i.e. the seperation in dimensions of total
            mass r/M
        num (int in (1, 2)) : specification wether we are interested in the primary's or secondary's
            contribution

    Returns:
        [d_keff, k_eff]:  r-derivative of, and pure enhancement of kltidal
    """
    cdef double X_this, X_other, lambda3Tidal, omega03, sqrteps3

    if num == 1:
        X_this = EOBpars.p_params.X_1
        X_other = EOBpars.p_params.X_2
        lambda3Tidal = EOBpars.tidal_params.lambda3Tidal1
        omega03 = EOBpars.tidal_params.omega03Tidal1
        sqrteps3 = EOBpars.tidal_params.sqrtepsilon3Tidal1
        spinshift = EOBpars.tidal_params.spinshiftomega03Tidal1
    elif num == 2:
        X_this = EOBpars.p_params.X_2
        X_other = EOBpars.p_params.X_1
        lambda3Tidal = EOBpars.tidal_params.lambda3Tidal2
        omega03 = EOBpars.tidal_params.omega03Tidal2
        sqrteps3 = EOBpars.tidal_params.sqrtepsilon3Tidal2
        spinshift = EOBpars.tidal_params.spinshiftomega03Tidal2
    else:
        raise ValueError("num must be either 1 or 2 to specify the object we are investigating!")
    if lambda3Tidal == 0.0:
        return (0., 0.)

    cdef double k_eff, dk_eff, r2, NNLO_terms, PN_terms, A_dyn_tides, dA_dyn_tides_dr

    k_eff, dk_eff = k_eff_amplification_and_dk_eff_amplification(r, omega03, spinshift, sqrteps3, 3)
    r2 = r*r

    NNLO_terms = (8./3. - 311./24.*X_this + 110.*X_this**2/3.)
    PN_terms = (1. + (-2. + 15./2.*X_this)/r + NNLO_terms/r2)
    A_dyn_tides = -15.*lambda3Tidal*k_eff*X_other/X_this/r**8 * PN_terms
    dA_dyn_tides_dr = (
        (dk_eff/k_eff - 8./r - ((15./2.*X_this - 2.)/r2 + 2.*NNLO_terms / r**3) / PN_terms) * A_dyn_tides
    )
    return (A_dyn_tides, dA_dyn_tides_dr)

cpdef (double, double, double) octu_and_d_octu_and_d2_octu_tidal_contribution(EOBParams EOBpars,
                                                                              double r,
                                                                              int num):
    """
    Function to compute the rr-derivative of the quadrupolar tidal contribution to the radial potential
    of the Hamiltonian in an effective description due to the presence of f-mode resonance AND the
        r-derivative AND
    the quadrupolar tidal contribution itself for efficiency
    Implements the upper line of eq. (2) of http://arxiv.org/abs/1812.08643
    Args:
        EOBpars (class): A PhysicalParams class as implemented in containers.pyx
        r (double) : the M-rescaled (dimensionless) EOB seperation, i.e. the seperation in dimensions of total
            mass r/M
        num (int in (1, 2)) : specification wether we are interested in the primary's or secondary's
            contribution

    Returns:
        [dd_k_eff, d_k_eff, k_eff]:  rr-derivative of, r-derivative of, and pure enhancement of kltidal
    """
    cdef double X_this, X_other, lambda3Tidal, omega03, sqrteps3, spinshift

    if num == 1:
        X_this = EOBpars.p_params.X_1
        X_other = EOBpars.p_params.X_2
        lambda3Tidal = EOBpars.tidal_params.lambda3Tidal1
        omega03 = EOBpars.tidal_params.omega03Tidal1
        sqrteps3 = EOBpars.tidal_params.sqrtepsilon3Tidal1
        spinshift = EOBpars.tidal_params.spinshiftomega03Tidal1
    elif num == 2:
        X_this = EOBpars.p_params.X_2
        X_other = EOBpars.p_params.X_1
        lambda3Tidal = EOBpars.tidal_params.lambda3Tidal2
        omega03 = EOBpars.tidal_params.omega03Tidal2
        sqrteps3 = EOBpars.tidal_params.sqrtepsilon3Tidal2
        spinshift = EOBpars.tidal_params.spinshiftomega03Tidal2
    else:
        raise ValueError("num must be either 1 or 2 to specify the object we are investigating!")
    if lambda3Tidal == 0.0:
        return (0., 0., 0.)

    cdef double k_eff, dk_eff, r2, NNLO_terms, PN_terms, A_dyn_tides, dA_dyn_tides_dr
    cdef double d2A_dyn_tides_dr2, deriv_terms

    k_eff, dk_eff, ddk_eff = (
        k_eff_amplification_and_dk_eff_amplification_and_ddk_eff_amplifcation(
            r, omega03, spinshift, sqrteps3, 3
        )
    )
    r2 = r*r

    NNLO_terms = (8./3. - 311./24.*X_this + 110.*X_this**2/3)
    PN_terms = (1. + (-2. + 15./2.*X_this)/r + NNLO_terms/r2)
    A_dyn_tides = -15.*lambda3Tidal*k_eff*X_other/X_this/r**8 * PN_terms
    deriv_terms = (dk_eff/k_eff - 8./r - ((15./2.*X_this - 2.)/r2 + 2.*NNLO_terms / r**3) / PN_terms)
    dA_dyn_tides_dr = deriv_terms * A_dyn_tides

    d2A_dyn_tides_dr2 = (
        (
            deriv_terms**2
            + ddk_eff/k_eff
            - (dk_eff/k_eff)**2
            + (8. + ((15.*X_this - 4.)/r + 6.*NNLO_terms/r2) / PN_terms) /r2
            - ((15./2.*X_this - 2.)/r2 + 2.*NNLO_terms / r**3)**2
            / PN_terms**2
        )
        * A_dyn_tides
    )
    return (A_dyn_tides, dA_dyn_tides_dr, d2A_dyn_tides_dr2)

# =================================================================
#                        k_eff COMPUTATIONS
# =================================================================

cpdef double k_eff_amplification(double r, double omega_l, double spinshift, double sqrt_eps_l, int l):
    """
    Function to compute the enhancement of kltidal due to the presence of f-mode resonance
    Implements (11) of https://arxiv.org/pdf/1702.02053.pdf (note typo: mOmega^2->(mOmega)^2)
    f(x) with x=omega0l/(m*Omega) (with m=l here) is the sum of first two terms of the bracket there
    Near resonance (x-1<1e-2) f(x) and f'(x) are replaced by a perturbative expansion
    Expected numerical precision: |Deltaf/f| normal: 1e-10 | perturbative: 1e-10
    Expected numerical precision: |Deltaf'/f'| normal: 1e-8 | perturbative: 1e-10
    Args:
        r (double) : the M-rescaled (dimensionless) inverse of the EOB seperation: u = M/r
        omega_l (double) : The M-rescaled (dimensionless) f-mode resonance frequency: omega_l_phys*M
        sqrt_eps_l : The square-root of the l-mode perturbation parameter in the two-timescale expansion, see
        eq. (6.43) of http://arxiv.org/abs/1608.01907
        l: The mode of the tidal contribution under investigation

    Returns:
        k_eff:  enhancement of kltidal
    """
    cdef double a, b, x, x2, x53, that, shiftrel, k_eff

    if omega_l == 0.:
        return 1.
    if l==2:
        a = 1./4
        b = 3./4
    elif l==3:
        a = 3./8
        b = 5./8
    else:
        raise ValueError("Only quadru- and octoupolar dynamical tides implemented: Choose l=2 or l=3!")
    x = x_func_spin(r, omega_l, spinshift, l)
    x2 = x*x
    x53 = x**(5./3.)

    shiftrel = spinshift/omega_l

    that = that_func(sqrt_eps_l, x53)
    k_eff = a + b * (f_spin(x, x2, x53, omega_l, spinshift)
                     + (1. - shiftrel) * _sqrt_pi_3/sqrt_eps_l*x2*Q_mix_v2(that))
    return k_eff

# This function does exactly the same as the LAL counterpart
cpdef double dk_eff_amplification(double r, double omega_l, double spinshift, double sqrt_eps_l, int l):
    """
    Function to compute the enhancement of kltidal and its u-derivative due to the presence of f-mode
        resonance
    Implements (11) of https://arxiv.org/pdf/1702.02053.pdf (note typo: mOmega^2->(mOmega)^2)
    f(x) with x=omega0l/(m*Omega) (with m=l here) is the sum of first two terms of the bracket there
    Near resonance (x-1<1e-2) f(x) and f'(x) are replaced by a perturbative expansion
    Expected numerical precision: |Deltaf/f| normal: 1e-10 | perturbative: 1e-10
    Expected numerical precision: |Deltaf'/f'| normal: 1e-8 | perturbative: 1e-10
    Args:
        r (double) : the M-rescaled (dimensionless) inverse of the EOB seperation: u = M/r
        omega_l (double) : The M-rescaled (dimensionless) f-mode resonance frequency: omega_l_phys*M
        sqrt_eps_l : The square-root of the l-mode perturbation parameter in the two-timescale expansion, see
        eq. (6.43) of http://arxiv.org/abs/1608.01907
        l: The mode of the tidal contribution under investigation

    Returns:
        k_eff:  enhancement of kltidal
    """

    if omega_l == 0.:
        return 0.

    cdef double b, x, x2, x53, that
    cdef double x_r, that_x, resonant_term_x, shiftrel
    cdef double _cos_fact, _sin_fact, fresnel_term, fresnel_term_that, prefac, k_eff_r

    if l==2:
        b = 3./4
    elif l==3:
        b = 5./8
    else:
        raise ValueError("Only quadru- and octoupolar dynamical tides implemented: Choose l=2 or l=3!")
    x = x_func_spin(r, omega_l, spinshift, l)
    x_r = 3./2*x/r
    x2 = x*x
    x53 = x**(5./3.)

    shiftrel = spinshift/omega_l

    that = that_func(sqrt_eps_l, x53)
    that_x = -8./3./sqrt_eps_l * x53/x

    resonant_term_x = df_spin(x, x2, x53, omega_l, spinshift)

    fresnel_term, fresnel_term_that = dQ_dthat_and_Q_v2(that)

    prefac = (1.-shiftrel) * _sqrt_pi_3/sqrt_eps_l

    k_eff_r = b * (resonant_term_x + prefac*(x2*fresnel_term_that*that_x + 2*x*fresnel_term)) * x_r

    return k_eff_r

# This function does exactly the same as the LAL counterpart
cpdef (double, double) k_eff_amplification_and_dk_eff_amplification(
    double r,
    double omega_l,
    double spinshift,
    double sqrt_eps_l,
    int l
):
    """
    Function to compute the enhancement of kltidal and its u-derivative due to the presence of f-mode
        resonance
    Implements (11) of https://arxiv.org/pdf/1702.02053.pdf (note typo: mOmega^2->(mOmega)^2)
    f(x) with x=omega0l/(m*Omega) (with m=l here) is the sum of first two terms of the bracket there
    Near resonance (x-1<1e-2) f(x) and f'(x) are replaced by a perturbative expansion
    Expected numerical precision: |Deltaf/f| normal: 1e-10 | perturbative: 1e-10
    Expected numerical precision: |Deltaf'/f'| normal: 1e-8 | perturbative: 1e-10
    Args:
        r: the M-rescaled (dimensionless) inverse of the EOB seperation: u = M/r
        omega_l: The M-rescaled (dimensionless) f-mode resonance frequency: omega_l_phys*M
        eta: The symmetric mass ratio
        l: The mode of the tidal contribution under investigation

    Returns:
        d k_eff/d u:  u-derivative of k_eff
    """
    cdef double a, b, x, x2, x53, that
    cdef double x_r, that_x, resonant_term_x, shiftrel
    cdef double _cos_fact, _sin_fact, fresnel_term, fresnel_term_that, prefac, k_eff_r
    if omega_l == 0.:
        return (1., 0.)

    if l==2:
        a = 1./4
        b = 3./4
    elif l==3:
        a = 3./8
        b = 5./8
    else:
        raise ValueError("Only quadru- and octoupolar dynamical tides implemented: Choose l=2 or l=3!")
    x = x_func_spin(r, omega_l, spinshift, l)
    x_r = 3./2.*x/r
    x2 = x*x
    x53 = x**(5./3.)

    shiftrel = spinshift/omega_l

    that = that_func(sqrt_eps_l, x53)
    that_x = -8./3./sqrt_eps_l * x53/x

    resonant_term = f_spin(x, x2, x53, omega_l, spinshift)
    resonant_term_x = df_spin(x, x2, x53, omega_l, spinshift)  # df/dx

    fresnel_term, fresnel_term_that = dQ_dthat_and_Q_v2(that)

    prefac = (1.-shiftrel) * _sqrt_pi_3/sqrt_eps_l

    k_eff = a + b * (resonant_term + prefac*x2*fresnel_term)

    k_eff_r = b * (resonant_term_x + prefac*(x2*fresnel_term_that*that_x + 2*x*fresnel_term)) * x_r

    return (k_eff, k_eff_r)

# This function does exactly the same as the LAL counterpart
cpdef (double, double, double) k_eff_amplification_and_dk_eff_amplification_and_ddk_eff_amplifcation(
    double r,
    double omega_l,
    double spinshift,
    double sqrt_eps_l,
    int l
):
    """
    Function to compute the enhancement of kltidal and its u-derivative due to the presence of f-mode
        resonance
    Implements (11) of https://arxiv.org/pdf/1702.02053.pdf (note typo: mOmega^2->(mOmega)^2)
    f(x) with x=omega0l/(m*Omega) (with m=l here) is the sum of first two terms of the bracket there
    Near resonance (x-1<1e-2) f(x) and f'(x) are replaced by a perturbative expansion
    Expected numerical precision: |Deltaf/f| normal: 1e-10 | perturbative: 1e-10
    Expected numerical precision: |Deltaf'/f'| normal: 1e-8 | perturbative: 1e-10
    Args:
        r: the M-rescaled (dimensionless) inverse of the EOB seperation: u = M/r
        omega_l: The M-rescaled (dimensionless) f-mode resonance frequency: omega_l_phys*M
        eta: The symmetric mass ratio
        l: The mode of the tidal contribution under investigation

    Returns:
        d k_eff/d u:  u-derivative of k_eff
    """
    cdef double a, b, x, x2, x53, that
    cdef double x_r, that_x, resonant_term_x, shiftrel
    cdef double _cos_fact, _sin_fact, fresnel_term, fresnel_term_that, prefac, k_eff_r
    cdef double k_eff_diff

    if omega_l == 0.:
        return (1., 0., 0.)

    if l==2:
        a = 1./4
        b = 3./4
    elif l==3:
        a = 3./8
        b = 5./8
    else:
        raise ValueError("Only quadru- and octoupolar dynamical tides implemented: Choose l=2 or l=3!")
    x = x_func_spin(r, omega_l, spinshift, l)
    x_r = 3./2*x/r  # dx/dr
    x2 = x*x
    x53 = x**(5./3.)

    shiftrel = spinshift/omega_l

    that = that_func(sqrt_eps_l, x53)
    that_x = -8./3./sqrt_eps_l * x53/x  # dthat/dx

    resonant_term = f_spin(x, x2, x53, omega_l, spinshift)
    resonant_term_x = df_spin(x, x2, x53, omega_l, spinshift)  # df/dx

    fresnel_term, fresnel_term_that = dQ_dthat_and_Q_v2(that)

    prefac = (1.-shiftrel) * _sqrt_pi_3/sqrt_eps_l

    k_eff = a + b * (resonant_term + prefac*x2*fresnel_term)

    k_eff_r = b * (resonant_term_x + prefac*(x2*fresnel_term_that*that_x + 2.*x*fresnel_term)) * x_r

    k_eff_diff = dk_eff_amplification(r*(1.+1e-6), omega_l, spinshift, sqrt_eps_l, l)

    k_eff_rr = (k_eff_diff - k_eff_r)*1e6/r  # Numerical derivative is much faster and equivalent

    return (k_eff, k_eff_r, k_eff_rr)

cpdef double that_func(double sqrt_eps, double x53):
    return 8./(5*sqrt_eps)*(1.-x53)

cpdef double Q(double that):
    cdef double S, C, approximation, exact
    if that <= -26:
        return -four_sqrt_three_pi/that
    if that <= -25:
        approximation = -four_sqrt_three_pi/that
        S, C = fresnel(0.5*that/_sqrt_pi_3)
        exact = cos(3./8*that**2)*(1+2*S) - sin(3./8*that**2)*(1+2*C)
        return (-25 - that) * approximation + (that + 26) * exact
    else:
        S, C = fresnel(0.5*that/_sqrt_pi_3)
        return cos(3./8*that**2)*(1+2*S) - sin(3./8*that**2)*(1+2*C)

cpdef double Q_mix(double that):
    # We choose a cosine transition for smoothness of the function, and the
    # transition interval [-45, -44] is chosen such that:
    # Max. relative error of Q: <1.422923101003295e-06
    # Max. absolute error of Q: <4.2135928007142054e-08
    # Max. relative error of dQ/dthat: <1e-4
    # Max. absolute error of dQ/dthat: <1e-7
    cdef double S, C, approximation, exact
    if that <= -45:
        return -four_sqrt_three_pi/that
    if that <= -44:  # PLS CHECK
        approximation = -four_sqrt_three_pi/that
        S, C = fresnel(0.5*that/_sqrt_pi_3)
        exact = cos(3./8*that**2)*(1+2*S) - sin(3./8*that**2)*(1+2*C)
        return (.5 - .5*cos(pi*(-44 - that))) * approximation + (.5 - .5*cos(pi*(that + 45))) * exact
    else:
        S, C = fresnel(0.5*that/_sqrt_pi_3)
        return cos(3./8*that**2)*(1+2*S) - sin(3./8*that**2)*(1+2*C)

cpdef double Q_mix_v2(double that):
    # We choose a cosine transition for smoothness of the function, and the
    # transition interval [-45, -44] is chosen such that:
    # Max. relative error of Q: <1.422923101003295e-06
    # Max. absolute error of Q: <4.2135928007142054e-08
    # Max. relative error of dQ/dthat: <1e-4
    # Max. absolute error of dQ/dthat: <1e-7
    cdef double inv_t, inv_t2, inv_t4, inv_t5, inv_t9
    cdef double S, C, approximation, exact
    if that <= -10.0:
        # Here we use the Taylor expansion in the beginning
        inv_t = 1.0 / that
        inv_t2 = inv_t * inv_t
        inv_t4 = inv_t2 * inv_t2
        inv_t5 = inv_t4 * inv_t
        inv_t9 = inv_t5 * inv_t4
        return -prefact*inv_t + prefactfive*inv_t5 - prefactnine*inv_t9
    if that <= -9.0:
        # Here we transition to the full equation
        inv_t = 1.0 / that
        inv_t2 = inv_t * inv_t
        inv_t4 = inv_t2 * inv_t2
        inv_t5 = inv_t4 * inv_t
        inv_t9 = inv_t5 * inv_t4
        approximation = -prefact*inv_t + prefactfive*inv_t5 - prefactnine*inv_t9
        c_fresnel(0.5*that/_sqrt_pi_3, &S, &C)
        exact = cos(3./8*that**2)*(1.+2.*S) - sin(3./8*that**2)*(1.+2.*C)
        return (.5 + .5*cos(pi*(that + 10.))) * approximation + (.5 - .5*cos(pi*(that + 10.))) * exact
    if that <= 9.0:
        # Here we use the full equation
        c_fresnel(0.5*that/_sqrt_pi_3, &S, &C)
        return cos(3./8*that**2)*(1.+2.*S) - sin(3./8*that**2)*(1.+2.*C)
    if that <= 10.0:
        # Here we transition to the Taylor expansion in the end
        approximation = (-prefact/that + prefactfive/that**5 - prefactnine/that**9
                         + 2*sqrt(2)*cos(3*that**2/8 + pi/4))
        c_fresnel(0.5*that/_sqrt_pi_3, &S, &C)
        exact = cos(3./8*that**2)*(1.+2.*S) - sin(3./8*that**2)*(1.+2.*C)
        return (.5 + .5*cos(pi*(that - 9.))) * exact + (.5 - .5*cos(pi*(that - 9.))) * approximation
    else:
        # Here we use the Taylor expansion in the end
        return (-prefact/that + prefactfive/that**5 - prefactnine/that**9
                + 2*sqrt(2)*cos(3*that**2/8 + pi/4))

cpdef double dQ_dthat(double that):
    # REDUNDANT
    # This is a legacy function which is not being used
    # in the computation
    if that <= -45:
        return four_sqrt_three_pi/that**2

    cdef double S, C
    cdef double fresnel_arg, sincosarg
    cdef double cos_fact, sin_fact

    cdef double approximation, exact, d_approximation, d_exact

    cdef double fresnel_term_that

    if that <= -44:

        approximation = -four_sqrt_three_pi/that
        d_approximation = four_sqrt_three_pi/that**2

        fresnel_arg = 0.5*that/_sqrt_pi_3
        S, C = fresnel(fresnel_arg)
        sincosarg = 3./8*that**2
        cos_fact = cos(sincosarg)
        sin_fact = sin(sincosarg)
        exact = cos_fact*(1+2*S) - sin_fact*(1+2*C)  # Q(x)
        d_exact = -3./4. * that * (cos_fact*(1+2*C) + sin_fact*(1+2*S))  # dQ/dthat
        return pi/2*(
            sin(pi*(that + 45))*exact - sin(pi*(-that - 55))*approximation
            ) + (.5 - .5*cos(pi*(-44 - that))) * d_approximation + (.5 - .5*cos(pi*(that + 45))) * d_exact

    else:
        fresnel_arg = 0.5*that/_sqrt_pi_3
        S, C = fresnel(fresnel_arg)
        sincosarg = 3./8*that**2
        cos_fact = cos(sincosarg)
        sin_fact = sin(sincosarg)
        fresnel_term_that = -3/4. * that * (cos_fact*(1+2*C) + sin_fact*(1+2*S))  # dQ/dthat
        return fresnel_term_that

cpdef (double, double) dQ_dthat_and_Q(that):
    if that <= -45:
        return (-four_sqrt_three_pi/that, four_sqrt_three_pi/that**2)

    cdef double S, C
    cdef double fresnel_arg, sincosarg
    cdef double cos_fact, sin_fact

    cdef double approximation, exact, d_approximation, d_exact
    cdef double transition, d_transition

    cdef double fresnel_term, fresnel_term_that
    if that <= -44:
        approximation = -four_sqrt_three_pi/that
        d_approximation = four_sqrt_three_pi/that**2

        fresnel_arg = 0.5*that/_sqrt_pi_3
        S, C = fresnel(fresnel_arg)
        sincosarg = 3./8*that**2
        cos_fact = cos(sincosarg)
        sin_fact = sin(sincosarg)
        exact = cos_fact*(1+2*S) - sin_fact*(1+2*C)  # Q(x)
        d_exact = -3/4. * that * (cos_fact*(1+2*C) + sin_fact*(1+2*S))  # dQ/dthat
        d_transition = (
            pi/2*(sin(pi*(that + 45))*exact - sin(pi*(-that - 44))*approximation)
            + (.5 - .5*cos(pi*(-44 - that)))
            * d_approximation
            + (.5 - .5*cos(pi*(that + 45)))
            * d_exact
        )
        transition = (.5 - .5*cos(pi*(-44 - that))) * approximation + (.5 - .5*cos(pi*(that + 45))) * exact
        return (transition, d_transition)

    else:
        fresnel_arg = 0.5*that/_sqrt_pi_3
        S, C = fresnel(fresnel_arg)
        sincosarg = 3./8*that**2
        cos_fact = cos(sincosarg)
        sin_fact = sin(sincosarg)
        fresnel_term = cos_fact*(1+2*S) - sin_fact*(1+2*C)  # Q(x)
        fresnel_term_that = -3/4. * that * (cos_fact*(1+2*C) + sin_fact*(1+2*S))  # dQ/dthat
        return (fresnel_term, fresnel_term_that)

cpdef (double, double) dQ_dthat_and_Q_v2(that):

    cdef double Q, dQ
    cdef double inv_t, inv_t2, inv_t4, inv_t5, inv_t6, inv_t9, inv_t10

    cdef double S, C
    cdef double fresnel_arg, sincosarg
    cdef double cos_fact, sin_fact

    cdef double approximation, exact, d_approximation, d_exact
    cdef double transition, d_transition

    cdef double fresnel_term, fresnel_term_that

    if that <= -10.0:
        inv_t = 1.0 / that
        inv_t2 = inv_t * inv_t
        inv_t4 = inv_t2 * inv_t2
        inv_t5 = inv_t4 * inv_t
        inv_t6 = inv_t4 * inv_t2
        inv_t9 = inv_t5 * inv_t4
        inv_t10 = inv_t5 * inv_t5
        Q = -prefact*inv_t + prefactfive*inv_t5 - prefactnine*inv_t9
        dQ = +prefact*inv_t2 -5*prefactfive*inv_t6 + 9*prefactnine*inv_t10
        return (Q, dQ)

    if that <= -9.0:
        inv_t = 1.0 / that
        inv_t2 = inv_t * inv_t
        inv_t4 = inv_t2 * inv_t2
        inv_t5 = inv_t4 * inv_t
        inv_t6 = inv_t4 * inv_t2
        inv_t9 = inv_t5 * inv_t4
        inv_t10 = inv_t5 * inv_t5
        approximation = -prefact*inv_t + prefactfive*inv_t5 - prefactnine*inv_t9
        d_approximation = +prefact*inv_t2 -5*prefactfive*inv_t6 + 9*prefactnine*inv_t10

        fresnel_arg = 0.5*that/_sqrt_pi_3
        c_fresnel(fresnel_arg, &S, &C)
        sincosarg = 3./8*that**2
        cos_fact = cos(sincosarg)
        sin_fact = sin(sincosarg)
        exact = cos_fact*(1+2*S) - sin_fact*(1+2*C)  # Q(x)
        d_exact = -3/4. * that * (cos_fact*(1+2*C) + sin_fact*(1+2*S))  # dQ/dthat
        d_transition = (pi/2.*(-sin(pi*(that + 10))*approximation
                               + sin(pi*(that + 10))*exact)
                        + (.5 + .5*cos(pi*(that + 10))) * d_approximation
                        + (.5 - .5*cos(pi*(that + 10))) * d_exact)
        transition = (.5 + .5*cos(pi*(that + 10))) * approximation + (.5 - .5*cos(pi*(that + 10))) * exact
        return (transition, d_transition)
    if that <= 9.0:
        fresnel_arg = 0.5*that/_sqrt_pi_3
        c_fresnel(fresnel_arg, &S, &C)
        sincosarg = 3./8*that**2
        cos_fact = cos(sincosarg)
        sin_fact = sin(sincosarg)
        fresnel_term = cos_fact*(1+2*S) - sin_fact*(1+2*C)  # Q(x)
        fresnel_term_that = -3/4. * that * (cos_fact*(1+2*C) + sin_fact*(1+2*S))  # dQ/dthat
        return (fresnel_term, fresnel_term_that)
    if that <= 10.0:
        approximation = (-prefact/that + prefactfive/that**5 - prefactnine/that**9
                         + 2.*sqrt(2)*cos(3.*that**2./8. + pi/4))
        d_approximation = (prefact/that**2 - 5*prefactfive/that**6 + 9*prefactnine/that**10
                           - 3./sqrt(2)*sin(3*that**2/8 + pi/4)*that)

        fresnel_arg = 0.5*that/_sqrt_pi_3
        c_fresnel(fresnel_arg, &S, &C)
        sincosarg = 3./8*that**2
        cos_fact = cos(sincosarg)
        sin_fact = sin(sincosarg)
        exact = cos_fact*(1+2*S) - sin_fact*(1+2*C)  # Q(x)
        d_exact = -3./4. * that * (cos_fact*(1+2*C) + sin_fact*(1+2*S))  # dQ/dthat
        d_transition = (pi/2*(-sin(pi*(that - 9))*exact
                              + sin(pi*(that - 9))*approximation)
                        + (.5 + .5*cos(pi*(that - 9))) * d_exact
                        + (.5 - .5*cos(pi*(that - 9))) * d_approximation)
        transition = (.5 + .5*cos(pi*(that - 9))) * exact + (.5 - .5*cos(pi*(that - 9))) * approximation
        return (transition, d_transition)
    else:
        Q = (-prefact/that + prefactfive/that**5 - prefactnine/that**9
             + 2*sqrt(2)*cos(3*that**2/8 + pi/4))
        dQ = (prefact/that**2 - 5*prefactfive/that**6 + 9*prefactnine/that**10
              - 3/sqrt(2)*sin(3*that**2/8 + pi/4)*that)
        return (Q, dQ)

cpdef double x_func(double r, double omega_l, int l):
    return omega_l/l*r**(3./2)

cpdef double x_func_spin(double r, double omega_l, double spinshift, int l):
    return (omega_l + spinshift)/l*r**(3./2)

cpdef double f(double x, double x2, double x53):
    if abs(x-1.) < 1e-2:
        return f_keffresonant_pert(x-1.)
    else:
        return x2*(1./(x2-1.)+5./6/(1.-x53))

cpdef double df(double x, double x2, double x53):
    if abs(x-1.) < 1e-2:
        return df_keffresonant_pert(x-1.)
    else:
        return 2.*x*(-1./(x2 - 1.)/(x2 - 1.) + 5./6.*(1. - 1./6.*x53)/(1. - x53)/(1. - x53))

cpdef double f_spin_pert(double z, double omegal, double spinshift):
    cdef double[5] p
    p[0] = -0.08333333333333333*((omegal - 3*spinshift)*(omegal - spinshift))/omegal**2
    p[1] = (
        -0.004629629629629629*(
            (omegal - spinshift)*(25*omegal**2 - 54*omegal*spinshift - 27*spinshift**2)
        )/omegal**3
    )
    p[2] = (
        -0.006944444444444444*(
            omegal**4 + 8*omegal**3*spinshift - 18*omegal**2*spinshift**2 + 9*spinshift**4
        )/omegal**4
    )
    p[3] = (
        (
            (omegal - spinshift)*(1309*omegal**4 - 7290*omegal**2*spinshift**2 + 3645*spinshift**4)
        )/(
            116640.*omegal**5
        )
    )
    p[4] = ((omegal - spinshift)*(-1661*omegal**5 + 3645*omegal**4*spinshift + 7290*omegal**3*spinshift**2
                                  - 7290*omegal**2*spinshift**3 - 3645*omegal*spinshift**4
                                  + 3645*spinshift**5))/(233280.*omegal**6)
    return (((p[4]*z + p[3])*z + p[2])*z + p[1])*z + p[0]

cpdef double df_spin_pert(double z, double omegal, double spinshift):
    cdef double[4] p
    p[0] = (
        -0.004629629629629629*(
            (omegal - spinshift)*(25*omegal**2 - 54*omegal*spinshift - 27*spinshift**2)
        )/omegal**3
    )
    p[1] = (
        2*(
            -0.006944444444444444*(
                omegal**4 + 8*omegal**3*spinshift - 18*omegal**2*spinshift**2 + 9*spinshift**4
            )/omegal**4
        )
    )
    p[2] = (
        3*(
            (omegal - spinshift)*(1309*omegal**4 - 7290*omegal**2*spinshift**2 + 3645*spinshift**4)
        )/(
            116640.*omegal**5
        )
    )
    p[3] = 4*((omegal - spinshift)*(-1661*omegal**5 + 3645*omegal**4*spinshift
                                    + 7290*omegal**3*spinshift**2 - 7290*omegal**2*spinshift**3
                                    - 3645*omegal*spinshift**4 + 3645*spinshift**5))/(233280.*omegal**6)
    return ((p[3]*z + p[2])*z + p[1])*z + p[0]

cpdef double f_spin(double x, double x2, double x53, double omega_l, double spinshift):
    if abs(x-1.) < 1e-2:
        return f_spin_pert(x-1., omega_l, spinshift)
    else:
        return (
            (
                (
                    -omega_l + spinshift
                )*x2*(
                    5/(omega_l*(-1 + x53)) + 6./(omega_l + spinshift*(-1 + x)**2 - omega_l*x2)
                )
            )/6.
        )

cpdef double df_spin(double x, double x2, double x53, double omega_l, double spinshift):
    if abs(x-1.) < 1e-2:
        return df_spin_pert(x-1., omega_l, spinshift)
    else:
        return (
            (
                (
                    -omega_l + spinshift
                )*x*(
                    (-25*x53)/(3.*omega_l*(-1 + x53)**2)
                    + (
                        12*x*(spinshift + omega_l*x - spinshift*x)
                    )/(
                        omega_l + spinshift*(-1 + x)**2 - omega_l*x2
                    )**2
                    + 2*(5/(omega_l*(-1 + x53)) + 6/(omega_l + spinshift*(-1 + x)**2 - omega_l*x2))
                )
            )/6.
        )
