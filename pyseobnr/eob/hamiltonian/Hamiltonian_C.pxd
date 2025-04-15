# cython: language_level=3

from ..utils.containers cimport EOBParams, CalibCoeffs, qp_param_t

# note: we need those types to be declared and named, so that we can use them in children classes
# and return variables. Using just the tuple definition does not seem to work well. See
# https://github.com/cython/cython/issues/6231 for details
ctypedef (double, double, double, double, double, double, double, double) Hamiltonian_C_call_return_t
ctypedef (double, double, double, double) Hamiltonian_C_grad_return_t
ctypedef (double, double, double, double, double, double) Hamiltonian_C_dynamics_return_t
ctypedef (double, double, double, double, double, double, double) Hamiltonian_C_auxderivs_return_t

cdef class Hamiltonian_C:

    cdef EOBParams EOBpars

    cpdef Hamiltonian_C_call_return_t _call(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2)

    cpdef Hamiltonian_C_grad_return_t grad(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2)

    cpdef hessian(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2)

    cpdef double csi(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2)

    cpdef Hamiltonian_C_dynamics_return_t dynamics(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2)

    cpdef double omega(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2)

    cpdef Hamiltonian_C_auxderivs_return_t auxderivs(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2)
