# cython: language_level=3

from pyseobnr.eob.utils.containers cimport EOBParams


cdef class Hamiltonian_C:
    def __cinit__(self, EOBParams eob_params not None):
        self.EOBpars = eob_params

    @property
    def eob_params(self):
        return self.EOBpars

    @eob_params.setter
    def eob_params(self, value):
        self.EOBpars = value

    @property
    def calibration_coeffs(self):
        return self.EOBpars.c_coeffs

    @calibration_coeffs.setter
    def calibration_coeffs(self, value):
        self.EOBpars.c_coeffs = value

    cpdef Hamiltonian_C_call_return_t _call(
        self,
        double[:] q,
        double[:] p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2
    ):
        pass

    cpdef Hamiltonian_C_grad_return_t grad(
        self,
        double[:] q,
        double[:] p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2
    ):
        pass

    cpdef hessian(
        self,
        double[:] q,
        double[:] p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2
    ):
        pass

    cpdef double csi(
        self,
        double[:] q,
        double[:] p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2
    ):
        pass

    cpdef Hamiltonian_C_dynamics_return_t dynamics(
        self,
        double[:] q,
        double[:] p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2
    ):
        pass

    cpdef double omega(
        self,
        double[:] q,
        double[:] p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2
    ):
        pass

    cpdef Hamiltonian_C_auxderivs_return_t auxderivs(
        self,
        double[:] q,
        double[:] p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2
    ):
        pass
