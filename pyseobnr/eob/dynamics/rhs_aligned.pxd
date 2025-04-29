# cython: language_level=3
from pyseobnr.eob.hamiltonian.Hamiltonian_C cimport Hamiltonian_C


cpdef augment_dynamics(
    const double[:, :] dynamics,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    Hamiltonian_C H
)
