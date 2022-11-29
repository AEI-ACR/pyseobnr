# cython: language_level=3

from pyseobnr.eob.utils.containers_coeffs_PN cimport PNCoeffs
from pyseobnr.eob.utils.utils_pn_opt cimport vpowers, spinVars

cpdef list compute_s1dot_opt(vpowers v_powers, spinVars spin_vars, PNCoeffs pn_coeffs)
cpdef list compute_s2dot_opt(vpowers v_powers, spinVars spin_vars, PNCoeffs pn_coeffs)
cpdef list compute_lNdot_opt(vpowers v_powers, spinVars spin_vars, PNCoeffs pn_coeffs)
cpdef list compute_lhat_opt(double nu, vpowers v_powers, spinVars spin_vars, PNCoeffs pn_coeffs)
cpdef list prec_eqns_20102022_cython_opt(double t, double[:] z, double nu, PNCoeffs pn_coeffs)
cpdef list prec_eqns_20102022_cython_opt1(double t, double[:] z, double nu, PNCoeffs pn_coeffs)
