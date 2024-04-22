# cython: language_level=3

cimport cython
from pyseobnr.eob.utils.containers cimport EOBParams
from pyseobnr.eob.waveform.waveform cimport RR_force, RadiationReactionForce
from pyseobnr.eob.hamiltonian.Hamiltonian_v5PHM_C cimport Hamiltonian_v5PHM_C

cpdef get_rhs_prec(
    double t,
    double[::1] z,
    Hamiltonian_v5PHM_C H,
    RadiationReactionForce RR,
    double m_1,
    double m_2,
    EOBParams params)
