# cython: language_level=3
cdef extern from "eob_parameters.h":
    int PN_limit
    int ell_max



cdef class PhysicalParams:
    cdef public double m_1,m_2,nu,M,X_1,X_2,delta,chi_1,chi_2,chi_S,chi_A,a,ap,am
    cdef public double chi1_L, chi2_L,a1,a2, H_val
    cdef public double[:] chi1_v
    cdef public double[:] chi2_v
    cdef public double[:] lN
    cdef public double omega,omega_circ
    cpdef update_spins(self,double chi_1,double chi_2)
    cdef _compute_derived_quants(self)

'''
cdef class CalibCoeffs:
    cdef public double a6,d5,dSO,dSS
    cdef public double flagNLOSO,flagNLOSO2,flagNLOSO3,flagNLOSS,flagNLOSS2,flagS3
'''
cdef class CalibCoeffs:
    cdef public dict dc

cdef class Dynamics:
    cdef public double[:] p_circ

cdef class FluxParams:
    # Coefficients entering the amplitude residual
    cdef public double[:,:,:] rho_coeffs
    cdef public double[:,:,:] rho_coeffs_log
    cdef public double[:,:,:] f_coeffs
    # Notice that f_coeffs_vh are allowed to be complex
    cdef public double complex[:,:,:] f_coeffs_vh

    # Coefficients entering the phase residual
    cdef public double complex[:,:,:] delta_coeffs
    cdef public double complex[:,:,:] delta_coeffs_vh

    # Newtonian prefixes
    # For waveform (complex)
    cdef public double complex[:,:] prefixes
    # For flux (real)
    cdef public double[:,:] prefixes_abs

    # These will hold the *results* of computing various pieces
    # of the waveform
    cdef public double[:,:] Tlm
    cdef public double complex[:,:] rholm
    cdef public double complex[:,:] deltalm

    # Holds nqc coeffs *for the flux*
    cdef public double[:,:,:] nqc_coeffs

    # Holds any extra calibration params that will be used in rholm
    # CAVEAT EMPTOR: this does *exactly* what you tell it to.
    cdef public double [:,:,:] extra_coeffs
    cdef public double [:,:,:] extra_coeffs_log

    cdef public list special_modes
    cdef public bint rho_initialized

    # Include more PN information than in SEOBNRv4HM
    cdef public bint extra_PN_terms

cdef class EOBParams:
    cdef public PhysicalParams p_params
    cdef public CalibCoeffs c_coeffs
    cdef public FluxParams flux_params
    cdef public Dynamics dynamics
    cdef public list mode_array
    cdef public bint aligned
