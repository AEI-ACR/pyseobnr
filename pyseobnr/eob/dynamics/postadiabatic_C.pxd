from pyseobnr.eob.hamiltonian.Hamiltonian_C cimport Hamiltonian_C

cpdef fin_diff_derivative(
    x,
    y,
    n=?,
)

cpdef cumulative_integral(
    x: np.array,
    y: np.array,
    order = ?,
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
    double[::1] q,
    double[::1] p,
    double tol=?,
)