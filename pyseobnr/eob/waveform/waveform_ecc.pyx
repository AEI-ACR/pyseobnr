# cython: language_level=3
# cython: profile=False, linetrace=False, cpow=True, binding=True, initializedcheck=False


import cython
import numpy as np
cimport numpy as cnp

cimport libcpp.complex as ccomplex
from libc.math cimport log, sqrt, fabs, tgamma
from libc.math cimport M_PI as pi
from libc.math cimport M_E, M_PI_2

from scipy.special.cython_special cimport loggamma
from scipy.interpolate import CubicSpline

from ..utils.containers cimport EOBParams
from ..dynamics.Keplerian_evolution_equations_flags cimport edot_zdot_xavg_flags
from ..dynamics.secular_evolution_equations_flags cimport edot_zdot_xdot_flags

from .RRforce_NS_v5EHM_v1_flags cimport RRforce_ecc_corr_NS_v5EHM_v1_flags

from .modes_ecc_corr_NS_v5EHM_v1_flags._implementation cimport BaseModesCalculation
from .modes_ecc_corr_NS_v5EHM_v1_flags cimport hlm_ecc_corr_NS_v5EHM_v1_flags


from .waveform cimport (
    compute_mode,
    compute_rholm,
    compute_rholm_single,
    compute_rho_coeffs,
    compute_deltalm_single,
    compute_delta_coeffs,
    EOBFluxCalculateNewtonianMultipoleAbs,
    EOBFluxCalculateNewtonianMultipole,
    compute_tail,
)

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_T

cdef complex I
I.real = 0
I.imag = 1

cdef extern from "eob_parameters.h":
    const int PN_limit
    const int ell_max

cdef inline double cabs(complex z) noexcept nogil:
    cdef ccomplex.complex[double] c
    c.real(z.real)
    c.imag(z.imag)
    return ccomplex.abs(c)

cdef inline double creal(complex z) noexcept nogil:
    return z.real

cdef inline complex cexp(complex z) noexcept nogil:
    cdef ccomplex.complex[double] c
    c.real(z.real)
    c.imag(z.imag)
    c = ccomplex.exp(c)

    cdef complex ret
    ret.real = c.real()
    ret.imag = c.imag()
    return ret

cdef inline double carg(complex z) noexcept nogil:
    cdef ccomplex.complex[double] c
    c.real(z.real)
    c.imag(z.imag)
    return ccomplex.arg(c)


cdef class RadiationReactionForceEcc:
    """
    Class with the wrapper of the RR_force function to enable typed calls.

    Contains the evolution equations for the eccentricity, relativistic
    anomaly, and dimensionless orbit-averaged orbital frequency which
    are coupledto the EOB equations of motion.

    It also contains the secular evolution equations for the eccentricity,
    relativistic anomaly, and dimensionless orbit-averaged orbital
    frequency, to enable typed calls.
    """

    def __cinit__(self):
        self.evolution_equations = edot_zdot_xavg_flags()
        self.secular_evolution_equations = edot_zdot_xdot_flags()

    cpdef initialize(self, EOBParams eob_pars):
        cdef dict flags_ecc = eob_pars.ecc_params.flags_ecc
        cdef:
            int flagPN1
            int flagPN32
            int flagPN2
            int flagPN52
            int flagPN3

        # PN flags are 1 (0) depending whether the PN orders are enabled (disabled)
        # 1PN flag
        flagPN1 = flags_ecc["flagPN1"]
        # 1.5PN flag
        flagPN32 = flags_ecc["flagPN32"]
        # 2PN flag
        flagPN2 = flags_ecc["flagPN2"]
        # 2.5PN flag
        flagPN52 = flags_ecc["flagPN52"]
        # 3PN flag
        flagPN3 = flags_ecc["flagPN3"]

        cdef dict init_params = {
            "nu": eob_pars.p_params.nu,
            "delta": eob_pars.p_params.delta,
            "chiA": eob_pars.p_params.chi_A,
            "chiS": eob_pars.p_params.chi_S,
            "flagPN1": flagPN1,
            "flagPN32": flagPN32,
            "flagPN2": flagPN2,
            "flagPN52": flagPN52,
            "flagPN3": flagPN3,
        }

        self.evolution_equations.initialize(**init_params)

    cpdef initialize_secular_evolution_equations(self, EOBParams eob_pars):
        cdef dict flags_ecc = eob_pars.ecc_params.flags_ecc
        cdef:
            int flagPN1
            int flagPN32
            int flagPN2
            int flagPN52
            int flagPN3

        # PN flags are 1 (0) depending whether the PN orders are enabled (disabled)
        # 1PN flag
        flagPN1 = flags_ecc["flagPN1"]
        # 1.5PN flag
        flagPN32 = flags_ecc["flagPN32"]
        # 2PN flag
        flagPN2 = flags_ecc["flagPN2"]
        # 2.5PN flag
        flagPN52 = flags_ecc["flagPN52"]
        # 3PN flag
        flagPN3 = flags_ecc["flagPN3"]

        cdef dict init_params = {
            "nu": eob_pars.p_params.nu,
            "delta": eob_pars.p_params.delta,
            "chiA": eob_pars.p_params.chi_A,
            "chiS": eob_pars.p_params.chi_S,
            "flagPN1": flagPN1,
            "flagPN32": flagPN32,
            "flagPN2": flagPN2,
            "flagPN52": flagPN52,
            "flagPN3": flagPN3,
        }

        self.secular_evolution_equations.initialize(**init_params)

    cpdef (double, double) RR(
        self,
        qp_param_t q,
        qp_param_t p,
        (double, double, double) Kep,
        double omega,
        double omega_circ,
        double H,
        EOBParams eob_pars,
    ):
        pass


cdef class SEOBNRv5RRForceEcc(RadiationReactionForceEcc):
    """
    Convenience wrappers around the RR_force function to enable typed calls.
    """

    def __cinit__(self, str RRForce):
        self.RRForce = RRForce
        self.instance_hlm = hlm_ecc_corr_NS_v5EHM_v1_flags()
        self.instance_forces = RRforce_ecc_corr_NS_v5EHM_v1_flags()

    cpdef initialize(self, EOBParams eob_pars):
        """
        Initializes the instance with the settings of the binary.

        :param EOBParams eob_pars: Settings of the binary.

        .. note::

            Calling this function prior to :py:meth:`.SEOBNRv5RRForceEcc.RR`
            is mandatory.
        """

        RadiationReactionForceEcc.initialize(self, eob_pars)

        cdef dict flags_ecc = eob_pars.ecc_params.flags_ecc
        cdef:
            int flagPN12
            int flagPN1
            int flagPN32
            int flagPN2
            int flagPN52
            int flagPN3
            int flagPA_modes
            int flagTail
            int flagMemory

        # PN flags are 1 (0) depending whether the PN orders are enabled (disabled)

        # 0.5PN flag
        flagPN12 = flags_ecc["flagPN12"]

        # 1PN flag
        flagPN1 = flags_ecc["flagPN1"]

        # 1.5PN flag
        flagPN32 = flags_ecc["flagPN32"]

        # 2PN flag
        flagPN2 = flags_ecc["flagPN2"]

        # 2.5PN flag
        flagPN52 = flags_ecc["flagPN52"]

        # 3PN flag
        flagPN3 = flags_ecc["flagPN3"]

        # Post-adiabatic effects flag
        flagPA_modes = flags_ecc["flagPA_modes"]

        # Tail terms flag
        flagTail = flags_ecc["flagTail"]

        # Memory effects flag
        flagMemory = flags_ecc["flagMemory"]

        cdef dict init_params = {
            "nu": eob_pars.p_params.nu,
            "flagPN12": flagPN12,
            "flagPN1": flagPN1,
            "flagPN32": flagPN32,
            "flagPN2": flagPN2,
            "flagPN52": flagPN52,
            "flagPN3": flagPN3,
            "flagPA": flagPA_modes,
            "flagTail": flagTail,
            "flagMemory": flagMemory,
        }

        # Initialization of parameters
        self.instance_hlm.initialize(**init_params)

        # Remove flags that are not used as inputs
        init_params.pop("flagPN12")
        init_params.pop("flagPA")
        init_params.pop("flagTail")
        init_params.pop("flagMemory")

        # Initialize RR force parameters
        if self.RRForce == "Ecc":
            self.instance_forces.initialize(**init_params)
        else:
            raise ValueError(
                "Incorrect value for 'RRForce'. Supported value: 'Ecc'"
            )

    cpdef (double, double) RR(
        self,
        qp_param_t q,
        qp_param_t p,
        (double, double, double) Kep,
        double omega,
        double omega_circ,
        double H,
        EOBParams eob_pars,
    ):
        """
        Compute the radiation reaction (RR) force.

        Args:
            q (double[::1]): Polar coordinates (r, phi)
            p (double[::1]): Canonical momentum in polar coordinates (pr,pphi)
            Kep (double[::1]): Keplerian parameterization variables (eccentricity,
                relativistic anomaly and x parameter)
            omega (double): Instantaneuos orbital frequency
            omega_circ (double): Instantaneuos circular orbital frequency
            H (double): Hamiltonian value
            eob_pars (EOBParams): Container with useful variables

        Returns:
            Radial and azimuthal components (Fr, Fphi) of the RR force.
        """

        # "RRForce = Ecc" already enforced in the initialize
        return RR_force_ecc(
            q,
            p,
            Kep,
            omega,
            omega_circ,
            H,
            eob_pars,
            self.instance_hlm,
            self.instance_forces
        )


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef (double, double) RR_force_ecc(
    qp_param_t q,
    qp_param_t p,
    (double, double, double) Kep,
    double omega,
    double omega_circ,
    double H,
    EOBParams eob_pars,
    hlm_ecc_corr_NS_v5EHM_v1_flags instance_hlm,
    RRforce_ecc_corr_NS_v5EHM_v1_flags instance_forces,
):
    """
    Compute the RR force in polar coordinates.
    This corresponds to the 'F_modes' function multiplied by eccentricity
    corrections, as in Eqs. (68) of [Gamboa2024a]_ .

    Args:
        q (double[::1]): Polar coordinates (r, phi)
        p (double[::1]): Canonical momentum in polar coordinates (pr,pphi)
        Kep (double[::1]): Keplerian parameterization variables (eccentricity,
            relativistic anomaly and x parameter)
        omega (double): Instantaneuos orbital frequency
        omega_circ (double): Instantaneuos circular orbital frequency
        H (double): Hamiltonian value
        eob_pars (EOBParams): Container with useful variables
        instance_hlm (hlm_ecc_corr_NS_v5EHM_v1_flags): Flags enabling or disabling
            different PN orders in the eccentric corrections to the modes
        instance_forces (RRforce_ecc_corr_NS_v5EHM_v1): Flags enabling or disabling
            different PN orders in the eccentric corrections to the RR force

    Returns:
        (float, float): Radial and azimuthal components (Fr, Fphi) of the RR force.
    """

    # eccentricity
    cdef double e = Kep[0]
    # relativistic anomaly
    cdef double z = Kep[1]
    # x-parameter related to the orbit-average orbital frequency
    cdef double x = Kep[2]

    instance_forces.compute(e=e, z=z, x=x)

    cdef double r = q[0]
    cdef double phi = q[1]
    cdef double pr = p[0]
    cdef double pphi = p[1]
    cdef double nu = eob_pars.p_params.nu
    # Note the multiplication of H by nu!
    cdef double flux = compute_flux_ecc(
        r, phi, pr, pphi, omega, Kep, nu*H, eob_pars, instance_hlm
    )
    flux /= nu
    cdef double f_over_om = flux/omega
    cdef double Fr = -pr / pphi * f_over_om * instance_forces.get("radial")
    cdef double Fphi = -f_over_om * instance_forces.get("azimuthal")

    return Fr, Fphi


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef double compute_flux_ecc(
    double r,
    double phi,
    double pr,
    double pphi,
    double omega,
    (double, double, double) Kep,
    double H,
    EOBParams eob_pars,
    hlm_ecc_corr_NS_v5EHM_v1_flags instance_hlm,
):
    """
    Compute the function 'F_modes' divided by :math:``\\Omega``, as described in
    Eq. (68c) of [Gamboa2024a]_.

    Args:
        r (double): Radial separation
        phi (double): Azimuthal angle (in radians)
        pr (double): Radial component of the momentum vector
        pphi (double): Azimuthal component of the momentum vector
        omega (double): Instantaneous orbital frequency
        Kep (double[::1]): Keplerian variables (eccentricity, relativistic anomaly, x-parameter)
        H (double): Value of the Hamiltonian
        eob_pars (EOBParams): Container with useful variables

    Return:
        float: GW energy flux
    """

    # Initialization of variables

    # Miscellaneous
    cdef int l, m
    cdef bint extra_PN_terms = eob_pars.flux_params.extra_PN_terms

    # Physical parameters
    cdef double nu = eob_pars.p_params.nu
    cdef double delta = eob_pars.p_params.delta
    cdef double a = eob_pars.p_params.a
    cdef double chi_S = eob_pars.p_params.chi_S
    cdef double chi_A = eob_pars.p_params.chi_A
    cdef double[:, :, :] rho_coeffs = eob_pars.flux_params.rho_coeffs
    cdef double[:, :, :] rho_coeffs_log = eob_pars.flux_params.rho_coeffs_log
    cdef double[:, :, :] f_coeffs = eob_pars.flux_params.f_coeffs
    cdef double complex[:, :, :] f_coeffs_vh = eob_pars.flux_params.f_coeffs_vh

    # Dynamical variables
    cdef double e = Kep[0]
    cdef double z = Kep[1]
    cdef double x = Kep[2]
    if x < 0:
        raise ValueError("Domain error")
    cdef double v = x**0.5  # omega**(1./3.)
    cdef double vh3 = H * x**(1.5)  # H*omega
    cdef double vh = vh3**(1./3.)
    cdef double omega_avg = x**(1.5)
    cdef double omega2 = omega * omega
    cdef double v2 = v * v

    # Tail
    cdef double tail
    cdef double[:, :] Tlm = eob_pars.flux_params.Tlm
    compute_tail(omega_avg, H, Tlm)  # Precompute the tail

    # Source
    cdef double Slm = 0.0
    cdef double source1 = (H * H - 1.0) / (2.0 * nu) + 1.0  # H_eff
    cdef double source2 = v * pphi

    # Modes
    cdef double complex hlm_QC = 1.0
    cdef double complex hlm_EccCorr = 1.0
    cdef ccomplex.complex[double] hlm_EccCorr_1
    cdef double complex hlm_Fact = 1.0
    cdef double hNewton = 1.0
    cdef double complex rholmPwrl = 1.0

    # Flux
    cdef double flux = 0.0
    cdef double[:, :] prefixes = eob_pars.flux_params.prefixes_abs  # Note the "abs" in the prefixes.

    # Computation of rho_lm

    compute_rho_coeffs(
        nu,
        delta,
        a,
        chi_S,
        chi_A,
        rho_coeffs,
        rho_coeffs_log,
        f_coeffs,
        f_coeffs_vh,
        extra_PN_terms)
    compute_rholm(v, vh, nu, eob_pars)

    # Updates all the modes at once in an efficient way
    instance_hlm.compute(e=e, z=z, x=x)

    for l in range(2, ell_max+1):
        for m in range(1, l+1):
            # Assemble the waveform
            hNewton = EOBFluxCalculateNewtonianMultipoleAbs(v2, M_PI_2, l, m, prefixes)
            if ((l + m) % 2) == 0:
                Slm = source1
            else:
                Slm = source2
            tail = Tlm[l, m]
            rholmPwrl = eob_pars.flux_params.rholm[l, m]
            hlm_QC = hNewton * Slm * tail * rholmPwrl
            hlm_EccCorr_1 = instance_hlm.get(l, m)
            hlm_EccCorr.real = hlm_EccCorr_1.real()
            hlm_EccCorr.imag = hlm_EccCorr_1.imag()
            hlm_Fact = hlm_QC * hlm_EccCorr

            flux += m * m * omega2 * cabs(hlm_Fact)**2

    return flux/(8 * pi)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef compute_hlms_ecc(
    double[:, :] dynamics,
    BaseModesCalculation modes_calculation,
    EOBParams eob_pars
):
    """
    Compute the inspiral modes for aligned-spin eccentric binaries
    with the eccentric corrections computed in [Gamboa2024a]_ .

    Args:
        dynamics (2d memory view of type double): Dynamical variables (r, phi, pr, pphi,
            eccentricity, relativistic anomaly, x-parameter, Hamiltonian, omega)
        modes_calculation (BaseModesCalculation): Base class for the calculation of
            the modes. It precomputes quantities before the start of the dynamics.
        eob_pars (EOBParams): Container with useful variables

    Returns:
        Dictionary with modes
    """

    # Initialization of variables

    cdef int l, m, i, j
    cdef (int, int) ell_m
    cdef int N = dynamics.shape[0]
    cdef bint extra_PN_terms = eob_pars.flux_params.extra_PN_terms

    cdef double pphi, v, H, vh, phi
    cdef (double, double, double) Kep = (0, 0, 0)
    cdef double vs[PN_limit]
    cdef double vhs[PN_limit]

    # Source term
    cdef double Slm = 0.0
    cdef double source1
    cdef double source2

    # Physical parameters
    cdef double nu = eob_pars.p_params.nu
    cdef double delta = eob_pars.p_params.delta
    cdef double a = eob_pars.p_params.a
    cdef double chi_S = eob_pars.p_params.chi_S
    cdef double chi_A = eob_pars.p_params.chi_A
    cdef double[:, :, :] rho_coeffs = eob_pars.flux_params.rho_coeffs
    cdef double[:, :, :] rho_coeffs_log = eob_pars.flux_params.rho_coeffs_log
    cdef double[:, :, :] f_coeffs = eob_pars.flux_params.f_coeffs
    cdef double complex[:, :, :] f_coeffs_vh = eob_pars.flux_params.f_coeffs_vh
    cdef double complex[:, :, :] delta_coeffs = eob_pars.flux_params.delta_coeffs
    cdef double complex[:, :, :] delta_coeffs_vh = eob_pars.flux_params.delta_coeffs_vh

    # Initial computation of the coefficients appearing in rho and delta
    compute_rho_coeffs(
        nu,
        delta,
        a,
        chi_S,
        chi_A,
        rho_coeffs,
        rho_coeffs_log,
        f_coeffs,
        f_coeffs_vh,
        extra_PN_terms)
    compute_delta_coeffs(nu, delta, a, chi_S, chi_A, delta_coeffs, delta_coeffs_vh)

    # Modes
    cdef int nb_modes = len(eob_pars.mode_array)
    cdef int[:, ::1] l_modes = np.empty((nb_modes, 2), dtype=np.intc)
    for j in range(nb_modes):
        ell_m = eob_pars.mode_array[j]
        l_modes[j, 0] = ell_m[0]
        l_modes[j, 1] = ell_m[1]

    # this temporary array is compact in the sense that its shape
    # is the one needed for the modes
    cdef double complex[:, ::1] temp_modes = np.empty(
        (N, nb_modes),
        dtype=np.complex128
    )

    vs[0] = 1
    vhs[0] = 1

    # Computation of the factorized modes
    for i in range(N):
        # For each instant of time, extract the quantities needed for the computation of the modes
        # (r, phi, pr_star, pphi, e, z, x, H, omega)
        phi = dynamics[i, 1]
        pphi = dynamics[i, 3]
        H = nu*dynamics[i, 7]
        v = dynamics[i, 6]**(0.5)  # omega**(1./3.)
        vh = H**(1./3.) * v  # (H*omega)**(1./3.)
        if v != v or vh != vh:
            raise ValueError("Domain error")

        # Various powers of v that enter the computation of rholm and deltalm
        for j in range(1, PN_limit):
            vs[j] = v**j
            vhs[j] = vh**j

        source1 = (H * H - 1.0) / (2.0 * nu) + 1.0  # H_eff
        source2 = v * pphi

        # Compute the eccentric correction coefficients
        Kep[0] = dynamics[i, 4]
        Kep[1] = dynamics[i, 5]
        Kep[2] = dynamics[i, 6]
        modes_calculation.compute(e=Kep[0], z=Kep[1], x=Kep[2])

        # Compute all the modes for the current instant of time
        for j in range(nb_modes):
            l = l_modes[j, 0]
            m = l_modes[j, 1]
            if ((l + m) % 2) == 0:
                Slm = source1
            else:
                Slm = source2

            temp_modes[i, j] = compute_mode_ecc(
                l,
                m,
                phi,
                Slm,
                vs,
                vhs,
                modes_calculation,
                eob_pars)

    cdef dict modes = {}
    for j in range(nb_modes):
        ell_m = eob_pars.mode_array[j]
        l = ell_m[0]
        m = ell_m[1]
        modes[l, m] = temp_modes[:, j]

    return modes


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef double complex compute_mode_ecc(
    int l,
    int m,
    double phi,
    double Slm,
    double[] vs,
    double[] vhs,
    BaseModesCalculation instance_hlm,
    EOBParams eob_pars
):
    """
    Compute the given (l,m) mode at one instant in time including eccentric
    corrections in a factorized form. See Eq. (97) in [Gamboa2024a]_.

    Args:
        l (int): l-index of the spin-weighted -2 spherical harmonic basis
        m (int): m-index of the spin-weighted -2 spherical harmonic basis
        phi (double): Orbital phase
        Slm (double): Source function
        vs (double): Velocity parameter to a certain integer power s
        vhs (double): Velocity parameter times the Hamitonian to a certain integer power s
        modes_calculation (BaseModesCalculation): Base class for the calculation of the modes.
            It precomputes quantities before the start of the dynamics.
        eob_pars (EOBParams): Container with useful variables

    Returns:
        (l,m)-mode including eccentric corrections

    """

    # Initialization of variables
    cdef double complex Tlm, hNewton, rholm, deltalm, hlmQC
    cdef double complex hlm_Fact = 1.0
    cdef double complex hlm_EccCorr = 1.0
    cdef ccomplex.complex[double] hlm_EccCorr_1
    cdef double v = vs[1]
    cdef double v2 = vs[2]
    cdef double vh = vhs[1]

    # For the computation of the tail T factor
    cdef double k = m* v * v * v  # This is m*Omega
    cdef double hathatk = m* vh * vh * vh  # This is m*Omega*H
    cdef double z2 = tgamma(l + 1)  # This is equal to l!
    cdef double complex lnr1 = loggamma(l + 1.0 - 2.0 * hathatk * I)

    # Quasi-circular part

    # Calculate the newtonian multipole, 1st term in Eq. 17, given by Eq. A1
    hNewton = EOBFluxCalculateNewtonianMultipole(v2, phi, l, m, eob_pars.flux_params.prefixes)

    # Compute rho^l
    rholm = compute_rholm_single(vs, vh, l, m, eob_pars)

    # Compute delta
    deltalm = compute_deltalm_single(vs, vhs, l, m, eob_pars.flux_params)

    # Compute tail function T
    Tlm = cexp((pi * hathatk) + I * (2.0 * hathatk * log(4.0 * k / sqrt(M_E)))) * cexp(lnr1)
    Tlm /= z2

    # Compute the QC mode
    hlmQC = hNewton * Slm * Tlm * rholm * cexp (I * deltalm)

    # Factorization corresponding to:
    # h_lm = h_lm(x) h_lm^{ecc corr}(x, e, z)  with x = x_avg
    hlm_EccCorr_1 = instance_hlm.get(l, m)
    hlm_EccCorr.real = hlm_EccCorr_1.real()
    hlm_EccCorr.imag = hlm_EccCorr_1.imag()
    hlm_Fact = hlmQC * hlm_EccCorr

    return hlm_Fact


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef compute_special_coeffs_ecc(
    double[:, :] dynamics,
    double t_attach,
    EOBParams eob_pars,
    dict amp_fits,
    dict amp_thresholds,
    dict modes=None
):
    """
    Compute the "special" amplitude coefficients. See discussion after Eq. (33)
    in https://dcc.ligo.org/LIGO-T2300060 (SEOBNRv5HM.pdf).
    This function is specialized to the order of the variables in dynamics of
    the eccentric model. In particular, the order of variables in 'dynamics' is:
    (r, phi, pr_star, pphi, e, z, x, H, omega)
    Additionally, instead of using omega, we are employing the replacement:
    omega -> x^(3/2)
    where x = <omega>^2/3.
    """

    # Step 0: spline the dynamics to the attachment point

    if modes is None:
        modes = {
            (2, 1): 7,
            (4, 3): 7,
        }

    cdef int i, j, l, m, power
    cdef double phi, pphi, H, v, vh, vphi, vphi2, source1, source2, Slm
    cdef double rholm, hlm, K, amp, min_amp
    cdef double vs[PN_limit]
    cdef double vhs[PN_limit]

    cdef cnp.ndarray[DTYPE_T, ndim=1] dynamics_all = np.zeros(dynamics.shape[1]-1)

    for i in range(1, dynamics.shape[1]):
        spline = CubicSpline(dynamics[:, 0], dynamics[:, i])
        dynamics_all[i-1] = spline(t_attach)

    compute_rho_coeffs(
        eob_pars.p_params.nu,
        eob_pars.p_params.delta,
        eob_pars.p_params.a,
        eob_pars.p_params.chi_S,
        eob_pars.p_params.chi_A,
        eob_pars.flux_params.rho_coeffs,
        eob_pars.flux_params.rho_coeffs_log,
        eob_pars.flux_params.f_coeffs,
        eob_pars.flux_params.f_coeffs_vh,
        eob_pars.flux_params.extra_PN_terms
    )

    compute_delta_coeffs(
        eob_pars.p_params.nu,
        eob_pars.p_params.delta,
        eob_pars.p_params.a,
        eob_pars.p_params.chi_S,
        eob_pars.p_params.chi_A,
        eob_pars.flux_params.delta_coeffs,
        eob_pars.flux_params.delta_coeffs_vh
    )

    # Now, loop over every mode of interest
    for mode in modes.keys():
        l, m = mode
        dynamics_interp = 1*dynamics_all

        phi = dynamics_interp[1]
        pphi = dynamics_interp[3]
        H = eob_pars.p_params.nu * dynamics_interp[7]
        v = dynamics_interp[6]**(0.5)  # omega**(1./3)
        vh = H**(1./3.) * v  # (H*omega)**(1./3)
        if v != v or vh != vh:
            raise ValueError("Domain error")
        vphi = v  # omega/omega_circ**(2./3)
        vphi2 = vphi*vphi

        for j in range(PN_limit):
            vs[j] = v**j
            vhs[j] = vh**j
        source1 = (H * H - 1.0) / (2.0 * eob_pars.p_params.nu) + 1.0  # H_eff
        source2 = v*pphi

        if ((l + m) % 2) == 0:
            Slm = source1
        else:
            Slm = source2
        # Step 1: compute the mode with the calibration coeff set to 0
        power = modes[mode]

        eob_pars.flux_params.f_coeffs[l, m, power] = 0.0
        hlm = cabs(compute_mode(vphi2, phi, Slm, vs, vhs, l, m, eob_pars))
        # Step 2: compute rho_lm for this mode
        rholm = creal(compute_rholm_single(vs, vh, l, m, eob_pars))

        # Step 3: compute K = |h_{lm}|/|rho_{lm}|
        K = hlm/fabs(rholm)

        # Step 4: compute |h_{lm}^{NR}| from fit

        amp = amp_fits[(l, m)]
        amp22 = amp_fits[(2, 2)]

        # when the amplitude at merger is too small a positive sign is better
        if np.abs(amp)<1e-4:
            amp = np.abs(amp)

        min_amp = amp_thresholds[(l, m)]
        if np.abs(amp)<amp22/min_amp:
            amp = np.sign(amp)*amp22/min_amp

        amp*=eob_pars.p_params.nu

        # Step 5: compute c_lm = (|h_{lm}^{NR}|/K - rho_{lm}(clm=0))/v**power
        clm1 = (amp/K - rholm)*1/v**power

        # We always pick the positive solution
        eob_pars.flux_params.f_coeffs[l, m, power] = clm1
