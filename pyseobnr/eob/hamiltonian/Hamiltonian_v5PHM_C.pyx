cimport cython
from pyseobnr.eob.utils.containers cimport EOBParams

cdef class Hamiltonian_v5PHM_C:
    def __cinit__(self,EOBParams):
        pass


    cpdef _call(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL_1,double chiL_2):
        pass
    cpdef grad(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL_1,double chiL_2):
        pass
    cpdef hessian(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL_1,double chiL_2):
        pass
    cpdef double csi(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL_1,double chiL_2):
        pass
    cpdef  dynamics(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL_1,double chiL_2):
        pass
    cpdef double omega(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL_1,double chiL_2):
        pass
    cpdef aux_derivs(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL_1,double chiL_2):
        pass
