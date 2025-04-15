# cython: language_level=3

from ..utils.containers cimport qp_param_t
from ..hamiltonian.Hamiltonian_C cimport Hamiltonian_C

cpdef fin_diff_derivative(
    x: np.array,
    y: np.array,
    int n=?,
)

cpdef cumulative_integral(
    x: np.array,
    y: np.array,
    int order=?,
)

cpdef Kerr_ISCO(
    double chi1,
    double chi2,
    double m1,
    double m2,
)

cpdef compute_adiabatic_solution(
    double[:] r,
    Hamiltonian_C H,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    qp_param_t q,
    qp_param_t p,
    double tol=?,
)
