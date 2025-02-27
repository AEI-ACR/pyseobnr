
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

    cpdef void update_spins(self,double chi_1,double chi_2):
        """
        Update the aligned spins and the derived quantities
        """
        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self._compute_derived_quants()

    @cython.cdivision(True)
    cdef void _compute_derived_quants(self):
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
    def __cinit__(self, dc):
        self.a6 = dc.get("a6", 0)
        self.dSO = dc.get("dSO", 0)
        self.ddSO = dc.get("ddSO", 0)


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


cdef class EccParams:
    """
    Holds several parameters for the eccentric model.

    For initialization, a dictionary ``dc`` should be given with the following
    keys:

    * ``dissipative_ICs``
    * ``eccentricity``
    * ``EccIC``
    * ``flags_ecc``
    * ``IC_messages``
    * ``rel_anomaly``
    * ``r_min``
    * ``t_max``

    All the other quantities are set to a default value.

    .. versionadded:: 0.3
    """

    def __init__(
        self,
        dc: dict):
        """
        Initializes the eccentric parameters.

        :param dict dc: dictionary containing initial values for the
            :py:attr:`~.EccParams.dissipative_ICs`
            :py:attr:`~.EccParams.eccentricity`
            :py:attr:`~.EccParams.EccIC`
            :py:attr:`~.EccParams.flags_ecc`
            :py:attr:`~.EccParams.IC_messages`
            :py:attr:`~.EccParams.rel_anomaly`
            :py:attr:`~.EccParams.r_min`
            :py:attr:`~.EccParams.t_max`
        """

    def __cinit__(self, dict dc):

        self.dissipative_ICs = dc["dissipative_ICs"]
        self.eccentricity = dc["eccentricity"]
        self.EccIC = dc["EccIC"]
        self.flags_ecc = dc["flags_ecc"]
        self.IC_messages = dc["IC_messages"]
        self.rel_anomaly = dc["rel_anomaly"]
        self.r_min = dc["r_min"]
        self.t_max = dc["t_max"]
        self.stopping_condition = ""
        self.validate_separation = True
        self.attachment_check_ecc = 0.0
        self.attachment_check_qc = 0.0
        self.NR_deltaT = 0.0
        self.omega_avg = 0.0
        self.omega_inst = 0.0
        self.omega_start_qc = 0.0
        self.r_final = 0.0
        self.r_ISCO = 0.0
        self.r_start_guess = 0.0
        self.r_start_ICs = 0.0
        self.t_attach_ecc = 0.0
        self.t_attach_ecc_predicted = 0.0
        self.t_attach_qc = 0.0
        self.t_attach_qc_predicted = 0.0
        self.t_ISCO_ecc = 0.0
        self.t_ISCO_qc = 0.0
        self.x_avg = 0.0


#@cython.embedsignature(True)
cdef class EOBParams:
    """
    Holds EOB quantities
    """

    def __init__(
        self,
        physical_params: dict,
        coeffs,
        aligned=True,
        extra_PN_terms=True,
        mode_array=[(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)],
        special_modes=[(2, 1), (4, 3), (5, 5)],
        ecc_model=False):
        """
        Initializes the EOB parameters.

        :param bool ecc_model: if set to ``True``, the EOB params are associated
            to the eccentric model and will initialize the class
            :py:class:`.EccParams`.
            This would require the ``physical_params`` to contain the keys required
            by the instantiation of :py:class:`.EccParams`.
        """
        # Needed to get the signature to embed
        pass

    def __cinit__(
        self,
        physical_params,
        coeffs,
        aligned=True,
        extra_PN_terms=True,
        mode_array=[(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)],
        special_modes=[(2, 1), (4, 3), (5, 5)],
        ecc_model=False):

        # Physical params (e.g. spins)
        self.p_params = PhysicalParams(physical_params)
        # Calibration coefficients
        self.c_coeffs = CalibCoeffs(coeffs)
        # Flux/waveform quantities
        self.flux_params = FluxParams(special_modes,extra_PN_terms)
        # Dynamics related quantities
        self.dynamics = Dynamics()
        # Mode array to use for generating modes
        self.mode_array = mode_array
        self.aligned = aligned

        # Parameters for the eccentric model
        if ecc_model:
            self.ecc_params = EccParams(physical_params)
