# cython: language_level=3
from pyseobnr.eob.utils.containers cimport EOBParams, FluxParams, qp_param_t

cdef extern from "eob_parameters.h":
    const int PN_limit
    const int ell_max

cpdef (double, double) RR_force(
    qp_param_t q,
    qp_param_t p,
    double omega,
    double omega_circ,
    double H,
    EOBParams eob_pars)

cdef class RadiationReactionForce:
    cpdef (double, double) RR(
        self,
        qp_param_t q,
        qp_param_t p,
        double omega,
        double omega_circ,
        double H,
        EOBParams eob_pars)

cdef class SEOBNRv5RRForce(RadiationReactionForce):
    cpdef (double, double) RR(
        self,
        qp_param_t q,
        qp_param_t p,
        double omega,
        double omega_circ,
        double H,
        EOBParams eob_par)

cpdef void compute_tail(double omega, double H, double[:, :] Tlm)

cdef double complex compute_mode(
    double v_phi2,
    double phi,
    double Slm,
    double[] vs,
    double[] vhs,
    int l,
    int m,
    EOBParams eob_pars)

cdef void compute_rholm(double v, double vh, double nu, EOBParams eob_pars)

cpdef void compute_rho_coeffs(
    double nu,
    double dm,
    double a,
    double chiS,
    double chiA,
    double[:, :, :] rho_coeffs,
    double[:, :, :] rho_coeffs_log,
    double[:, :, :] f_coeffs,
    double complex[:, :, :] f_coeffs_vh,
    bint extra_PN_terms)

cdef double complex compute_rholm_single(
    double[] vs,
    double vh,
    int l,
    int m,
    EOBParams eob_pars)


cdef double complex compute_deltalm_single(
    double[] vs,
    double[] vhs,
    int l,
    int m,
    FluxParams fl)

cdef public void compute_delta_coeffs(
    double nu,
    double dm,
    double a,
    double chiS,
    double chiA,
    double complex[:, :, :] delta_coeffs,
    double complex[:, :, :] delta_coeffs_vh)

cpdef double  EOBFluxCalculateNewtonianMultipoleAbs(
    double x,
    double phi,
    int l,
    int m,
    double [:, :] params
)

cdef double complex EOBFluxCalculateNewtonianMultipole(
    double x,
    double phi,
    int l,
    int m,
    double complex[:, :] params
)
