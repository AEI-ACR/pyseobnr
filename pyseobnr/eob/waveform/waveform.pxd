# cython: language_level=3
from pyseobnr.eob.utils.containers cimport EOBParams, FluxParams
cdef extern from "eob_parameters.h":
    const int PN_limit
    const int ell_max

cpdef (double, double) RR_force(
    double[::1] q,
    double[::1] p,
    double omega,
    double omega_circ,
    double H,
    EOBParams eob_pars)

cdef class RadiationReactionForce:
    cpdef (double, double) RR(
        self,
        double[::1] q,
        double[::1] p,
        double omega,
        double omega_circ,
        double H,
        EOBParams eob_pars)

cdef class SEOBNRv5RRForce(RadiationReactionForce):
    cpdef (double, double) RR(
        self,
        double[::1] q,
        double[::1] p,
        double omega,
        double omega_circ,
        double H,
        EOBParams eob_par)
