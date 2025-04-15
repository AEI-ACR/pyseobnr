# cython: language_level=3

from ..utils.containers cimport EOBParams, CalibCoeffs, qp_param_t, chiv_param_t

# note: we need those types to be declared and named, so that we can use them in children classes
# and return variables. Using just the tuple definition does not seem to work well. See
# https://github.com/cython/cython/issues/6231 for details
# type for the result to _call, see below for the details on other calls
ctypedef (
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double) Hamiltonian_v5PHM_C_call_result_t
ctypedef (double, double, double, double) Hamiltonian_v5PHM_C_grad_result_t
ctypedef (double, double, double, double, double, double) Hamiltonian_v5PHM_C_dynamics_result_t
ctypedef (
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double) Hamiltonian_v5PHM_C_auxderivs_result_t


cdef class Hamiltonian_v5PHM_C:
    cdef EOBParams EOBpars

    cpdef Hamiltonian_v5PHM_C_call_result_t _call(
        self,
        qp_param_t q,
        qp_param_t p,
        chiv_param_t chi1_v,
        chiv_param_t chi2_v,
        double m_1,
        double m_2,
        double chi_1,
        double chi_2,
        double chiL_1,
        double chiL_2)

    cpdef Hamiltonian_v5PHM_C_grad_result_t grad(
        self,
        qp_param_t q,
        qp_param_t p,
        chiv_param_t chi1_v,
        chiv_param_t chi2_v,
        double m_1,
        double m_2,
        double chi_1,
        double chi_2,
        double chiL_1,
        double chiL_2)

    cpdef hessian(
        self,
        qp_param_t q,
        qp_param_t p,
        chiv_param_t chi1_v,
        chiv_param_t chi2_v,
        double m_1,
        double m_2,
        double chi_1,
        double chi_2,
        double chiL_1,
        double chiL_2)

    cpdef double csi(
        self,
        qp_param_t q,
        qp_param_t p,
        chiv_param_t chi1_v,
        chiv_param_t chi2_v,
        double m_1,
        double m_2,
        double chi_1,
        double chi_2,
        double chiL_1,
        double chiL_2)

    cpdef Hamiltonian_v5PHM_C_dynamics_result_t dynamics(
        self,
        qp_param_t q,
        qp_param_t p,
        chiv_param_t chi1_v,
        chiv_param_t chi2_v,
        double m_1,
        double m_2,
        double chi_1,
        double chi_2,
        double chiL_1,
        double chiL_2)

    cpdef double omega(
        self,
        qp_param_t q,
        qp_param_t p,
        chiv_param_t chi1_v,
        chiv_param_t chi2_v,
        double m_1,
        double m_2,
        double chi_1,
        double chi_2,
        double chiL_1,
        double chiL_2)

    cpdef Hamiltonian_v5PHM_C_auxderivs_result_t auxderivs(
        self,
        qp_param_t q,
        qp_param_t p,
        chiv_param_t chi1_v,
        chiv_param_t chi2_v,
        double m_1,
        double m_2,
        double chi_1,
        double chi_2,
        double chiL_1,
        double chiL_2)
