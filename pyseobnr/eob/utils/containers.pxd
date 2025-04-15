# cython: language_level=3
cdef extern from "eob_parameters.h":
    int PN_limit
    int ell_max

# Generic type for either q or p
ctypedef (double, double) qp_param_t

# Generic type for a 3 dimensional vector
ctypedef (double, double, double) chiv_param_t

cdef class PhysicalParams:
    cdef public:
        double m_1
        double m_2
        double nu
        double M
        double X_1
        double X_2
        double delta
        double chi_1
        double chi_2
        double chi_S
        double chi_A
        double a
        double ap
        double am
        double chi1_L
        double chi2_L
        double a1
        double a2
        double H_val

        chiv_param_t chi1_v
        chiv_param_t chi2_v
        double[:] lN

        double omega
        double omega_circ

    cpdef void update_spins(self, double chi_1, double chi_2)
    cdef void _compute_derived_quants(self)


cdef class CalibCoeffs:

    # Parameters involved in the calibration
    cdef public:
        double a6
        double dSO
        double ddSO


cdef class Dynamics:
    cdef public qp_param_t p_circ

cdef class FluxParams:
    # Coefficients entering the amplitude residual
    cdef public double[:, :, :] rho_coeffs
    cdef public double[:, :, :] rho_coeffs_log
    cdef public double[:, :, :] f_coeffs
    # Notice that f_coeffs_vh are allowed to be complex
    cdef public double complex[:, :, :] f_coeffs_vh

    # Coefficients entering the phase residual
    cdef public double complex[:, :, :] delta_coeffs
    cdef public double complex[:, :, :] delta_coeffs_vh

    # Newtonian prefixes
    # For waveform (complex)
    cdef public double complex[:, :] prefixes
    # For flux (real)
    cdef public double[:, :] prefixes_abs

    # These will hold the *results* of computing various pieces
    # of the waveform
    cdef public double[:, :] Tlm
    cdef public double complex[:, :] rholm
    cdef public double complex[:, :] deltalm

    # Holds nqc coeffs *for the flux*
    cdef public double[:, :, :] nqc_coeffs

    # Holds any extra calibration params that will be used in rholm
    # CAVEAT EMPTOR: this does *exactly* what you tell it to.
    cdef public double [:, :, :] extra_coeffs
    cdef public double [:, :, :] extra_coeffs_log

    cdef public list special_modes
    cdef public bint rho_initialized

    # Include more PN information than in SEOBNRv4HM
    cdef public bint extra_PN_terms


cdef class EccParams:

    cdef public bint IC_messages
    cdef public bint validate_separation
    cdef public dict flags_ecc
    cdef public int EccIC
    cdef public double attachment_check_ecc
    cdef public double attachment_check_qc
    cdef public double eccentricity
    cdef public double NR_deltaT
    cdef public double omega_avg
    cdef public double omega_inst
    cdef public double omega_start_qc
    cdef public double rel_anomaly
    cdef public double r_final
    cdef public double r_ISCO
    cdef public double r_min
    cdef public double r_start_guess
    cdef public double r_start_ICs
    cdef public double t_attach_ecc
    cdef public double t_attach_ecc_predicted
    cdef public double t_attach_qc
    cdef public double t_attach_qc_predicted
    cdef public double t_ISCO_ecc
    cdef public double t_ISCO_qc
    cdef public double t_max
    cdef public double x_avg
    cdef public str dissipative_ICs
    cdef public str stopping_condition


cdef class EOBParams:
    cdef public PhysicalParams p_params
    cdef public CalibCoeffs c_coeffs
    cdef public FluxParams flux_params
    cdef public Dynamics dynamics
    cdef public list mode_array
    cdef public bint aligned

    cdef public EccParams ecc_params
    """
    Parameters for the eccentric model.

        :type: :py:class:`.EccParams`
    """
