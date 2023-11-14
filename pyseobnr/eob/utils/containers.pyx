
cimport cython
cimport numpy as np
import numpy as np

cdef class PhysicalParams:
    @cython.cdivision(True)
    @cython.embedsignature(True)
    def __cinit__(self,dict dc):
        self.m_1 = dc["m_1"]
        self.m_2 = dc["m_2"]
        self.M = self.m_1+self.m_2
        self.nu = self.m_1*self.m_2/(self.M)**2
        self.X_1 = self.m_1/self.M
        self.X_2 = self.m_2/self.M
        self.delta = self.m_1 - self.m_2
        self.a1 =  dc["a1"]
        self.a2 =  dc["a2"]
        self.chi_1 = dc["chi_1"]
        self.chi_2 = dc["chi_2"]
        self.chi1_v = dc["chi1_v"]
        self.chi2_v = dc["chi2_v"]
        self.lN = dc["lN"]
        self.omega = dc["omega"]
        self.omega_circ = dc["omega"]
        self.H_val = dc["H_val"]
        self._compute_derived_quants()

    cpdef update_spins(self,double chi_1,double chi_2):
        """
        Update the aligned spins and the derived quantities
        """
        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self._compute_derived_quants()

    @cython.cdivision(True)
    cdef _compute_derived_quants(self):
        """
        Compute auxiliary spin quantities
        """
        # Symmetric and anti-symmetric spins
        self.chi_S = (self.chi_1+self.chi_2)/2
        self.chi_A = (self.chi_1-self.chi_2)/2
        # Test-particle spin
        self.a = (1 - 2 * self.nu) * self.chi_S + (self.m_1 - self.m_2) / (
            self.m_1 + self.m_2) * self.chi_A
        # a_{+,-}
        self.ap = self.X_1*self.chi_1+self.X_2*self.chi_2
        self.am = self.X_1*self.chi_1-self.X_2*self.chi_2

cdef class CalibCoeffs():
    def __cinit__(self,dc):
        self.dc = dc

    def __getitem__(self,str x):
        return self.dc[x]
    def __setitem__(self,str x,double y):
        self.dc[x] = y

cdef class FluxParams:
    @cython.embedsignature(True)
    def __cinit__(self,special_modes,extra_PN_terms):
        self.Tlm = np.zeros((ell_max+1,ell_max+1),dtype=np.float64)
        self.rho_coeffs = np.zeros((ell_max+1,ell_max+1,PN_limit),dtype=np.float64)
        self.rho_coeffs_log = np.zeros((ell_max+1,ell_max+1,PN_limit),dtype=np.float64)
        self.f_coeffs = np.zeros((ell_max+1,ell_max+1,PN_limit),dtype=np.float64)
        self.f_coeffs_vh = np.zeros((ell_max+1,ell_max+1,PN_limit),dtype=np.complex128)
        self.delta_coeffs = np.zeros((ell_max+1,ell_max+1,PN_limit),dtype=np.complex128)
        self.delta_coeffs_vh = np.zeros((ell_max+1,ell_max+1,PN_limit),dtype=np.complex128)
        self.rholm = np.zeros((ell_max+1,ell_max+1),dtype=np.complex128)
        self.deltalm = np.zeros((ell_max+1,ell_max+1),dtype=np.complex128)
        self.prefixes = np.zeros((ell_max+1,ell_max+1),dtype=np.complex128)
        self.prefixes_abs = np.zeros((ell_max+1,ell_max+1),dtype=np.float64)
        self.nqc_coeffs = np.zeros((ell_max+1,ell_max+1,3),dtype=np.float64)
        self.extra_coeffs = np.zeros((ell_max+1,ell_max+1,PN_limit),dtype=np.float64)
        self.extra_coeffs_log = np.zeros((ell_max+1,ell_max+1,PN_limit),dtype=np.float64)
        self.special_modes = special_modes
        self.extra_PN_terms = extra_PN_terms

cdef class Dynamics:
    def __cinit__(self):
        self.p_circ = np.zeros(2,dtype=np.float64)


@cython.embedsignature(True)
cdef class EOBParams:
    """
    A class to hold various EOB related quantities
    """
    def __init__(self,physical_params,coeffs,aligned=True,extra_PN_terms=True,
                mode_array=[(2,2),(2,1),(3,3),(3,2),(4,4),(4,3),(5,5)],
                special_modes=[(2,1),(4,3),(5,5)]):
        # Needed to get the signature to embed
        pass

    def __cinit__(self,physical_params,coeffs,aligned=True,extra_PN_terms=True,
                mode_array=[(2,2),(2,1),(3,3),(3,2),(4,4),(4,3),(5,5)],
                special_modes=[(2,1),(4,3),(5,5)]):
        # Physical params (e.g. spins)
        self.p_params = PhysicalParams(physical_params)
        # Calibration coefficients
        #self.c_coeffs = CalibCoeffs(coeffs)
        # Flux/waveform quantities
        self.flux_params = FluxParams(special_modes,extra_PN_terms)
        # Dynamics related quantities
        self.dynamics = Dynamics()
        # Mode array to use for generating modes
        self.mode_array = mode_array
        self.aligned = aligned
