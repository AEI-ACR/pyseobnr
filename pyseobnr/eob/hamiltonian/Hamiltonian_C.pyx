# cython: language_level=3

cimport cython
from pyseobnr.eob.utils.containers cimport EOBParams

cdef class Hamiltonian_C:
    def __cinit__(self,EOBParams):
        pass


    cpdef _call(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2):
        pass
    cpdef grad(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2):
        pass
    cpdef hessian(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2):
        pass
    cpdef double csi(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2):
        pass
    cpdef  dynamics(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2):
        pass
    cpdef double omega(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2):
        pass
    cpdef auxderivs(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2):
        pass