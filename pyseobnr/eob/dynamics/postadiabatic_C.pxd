# cython: language_level=3

cimport numpy as cnp

from ..utils.containers cimport qp_param_t
from ..hamiltonian.Hamiltonian_C cimport Hamiltonian_C

cpdef cnp.ndarray[double, ndim=1, mode="c"] fin_diff_derivative(
    cnp.ndarray[double, ndim=1] x,
    cnp.ndarray[double, ndim=1] y,
)

cpdef cnp.ndarray[double, ndim=1, mode="c"] cumulative_integral(
    cnp.ndarray[double, ndim=1] x,
    cnp.ndarray[double, ndim=1] y,
    int order=?,
)

cpdef (double, double) Kerr_ISCO(
    double chi1,
    double chi2,
    double m1,
    double m2,
)

cpdef cnp.ndarray[double, ndim=1, mode="c"] compute_adiabatic_solution(
    cnp.ndarray[double, ndim=1, mode="c"] r,
    Hamiltonian_C H,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    qp_param_t q,
    qp_param_t p,
    double tol=?,
)

cpdef Newtonian_j0(cnp.ndarray[double, ndim=1] r)

cpdef univariate_spline_integral(
    x: np.array,
    y: np.array,
)
