# cython: language_level=3
cimport cython

from ..utils.containers cimport EOBParams, qp_param_t
from ..dynamics.Keplerian_evolution_equations_flags cimport edot_zdot_xavg_flags
from ..dynamics.secular_evolution_equations_flags cimport edot_zdot_xdot_flags
from .modes_ecc_corr_NS_v5EHM_v1_flags._implementation cimport BaseModesCalculation
from .modes_ecc_corr_NS_v5EHM_v1_flags cimport hlm_ecc_corr_NS_v5EHM_v1_flags
from .RRforce_NS_v5EHM_v1_flags cimport RRforce_ecc_corr_NS_v5EHM_v1_flags


cdef class RadiationReactionForceEcc:
    cdef public edot_zdot_xavg_flags evolution_equations
    cdef public edot_zdot_xdot_flags secular_evolution_equations

    cpdef initialize(self, EOBParams eob_pars)

    cpdef initialize_secular_evolution_equations(self, EOBParams eob_pars)

    cpdef (double, double) RR(
        self,
        qp_param_t q,
        qp_param_t p,
        (double, double, double) Kep,
        double omega,
        double omega_circ,
        double H,
        EOBParams eob_pars,
    )


cdef class SEOBNRv5RRForceEcc(RadiationReactionForceEcc):
    cdef public str RRForce
    cdef public hlm_ecc_corr_NS_v5EHM_v1_flags instance_hlm
    cdef public RRforce_ecc_corr_NS_v5EHM_v1_flags instance_forces

    cpdef initialize(self, EOBParams eob_pars)

    cpdef (double, double) RR(
        self,
        qp_param_t q,
        qp_param_t p,
        (double, double, double) Kep,
        double omega,
        double omega_circ,
        double H,
        EOBParams eob_pars,
    )


cdef double complex compute_mode_ecc(
    int l,
    int m,
    double phi,
    double Slm,
    double[] vs,
    double[] vhs,
    BaseModesCalculation instance_hlm,
    EOBParams eob_pars,
)


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
)
