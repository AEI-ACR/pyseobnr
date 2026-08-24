
# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
# cython: profile=False, linetrace=False, cpow=True

"""
Additional utility functions to compute the dynamical tides due to f-mode resonance,
which directly influence the dynamics, as well as the PN contributions
"""

from pyseobnr.eob.utils.containers cimport EOBParams

cpdef double tidal_contribution(
    EOBParams EOBpars,
    double r,
    int num)

cpdef (double, double) tidal_and_d_tidal_contribution(
    EOBParams EOBpars,
    double r,
    int num)

cpdef (double, double, double) tidal_and_d_tidal_and_d2_tidal_contribution(
    EOBParams EOBpars,
    double r,
    int num)
