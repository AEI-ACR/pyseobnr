# cython: language_level=3

cimport cython
from pyseobnr.eob.utils.containers cimport EOBParams

cdef class Hamiltonian_C:
    cdef public EOBParams EOBpars
    cpdef _call(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2)
    cpdef grad(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2)
    cpdef hessian(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2)
    cpdef double csi(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2)
    cpdef dynamics(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2)
    cpdef double omega(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2)
    cpdef auxderivs(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2)