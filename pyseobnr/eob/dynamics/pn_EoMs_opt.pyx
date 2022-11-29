cimport cython

from pyseobnr.eob.utils.containers_coeffs_PN cimport PNCoeffs
from pyseobnr.eob.utils.utils_pn_opt cimport vpowers, spinVars, my_norm_cy


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef list compute_s1dot_opt(vpowers v_powers, spinVars spin_vars, PNCoeffs pn_coeffs):
    """
    Evolution equation for S1dot up to 4PN order
    """
    # Initialize memory arrays
    cdef double sdotv110
    cdef double sdotv19
    cdef double sdotv18
    cdef double sdotv17
    cdef double sdotv16
    cdef double sdotv15
    cdef double s1dotx, s1doty, s1dotz

    # coefficients entering s1dot

    # x-component
    sdotv110 = spin_vars.SxS12x*pn_coeffs.s1dot_coeffs.asdotv11061 + spin_vars.lNxS1x*(pn_coeffs.s1dot_coeffs.asdotv110621*spin_vars.lNS1 + pn_coeffs.s1dot_coeffs.asdotv110622*spin_vars.lNS2)
    sdotv19 = pn_coeffs.s1dot_coeffs.asdotv1950*spin_vars.lNxS1x
    sdotv18 = spin_vars.SxS12x*pn_coeffs.s1dot_coeffs.asdotv1841 + spin_vars.lNxS1x*(pn_coeffs.s1dot_coeffs.asdotv18421*spin_vars.lNS2 + pn_coeffs.s1dot_coeffs.asdotv18422*spin_vars.lNS1)
    sdotv17 = pn_coeffs.s1dot_coeffs.asdotv1730*spin_vars.lNxS1x
    sdotv16 = spin_vars.SxS12x*pn_coeffs.s1dot_coeffs.asdotv1621 + spin_vars.lNxS1x*(pn_coeffs.s1dot_coeffs.asdotv16221*spin_vars.lNS2 + pn_coeffs.s1dot_coeffs.asdotv16222*spin_vars.lNS1)
    sdotv15 = pn_coeffs.s1dot_coeffs.asdotv1510*spin_vars.lNxS1x
    s1dotx = sdotv110*v_powers.v10 + sdotv15*v_powers.v5 + sdotv16*v_powers.v6 + sdotv17*v_powers.v7 + sdotv18*v_powers.v8 + sdotv19*v_powers.v9

    # y-component
    sdotv110 = spin_vars.SxS12y*pn_coeffs.s1dot_coeffs.asdotv11061 + spin_vars.lNxS1y*(pn_coeffs.s1dot_coeffs.asdotv110621*spin_vars.lNS1 + pn_coeffs.s1dot_coeffs.asdotv110622*spin_vars.lNS2)
    sdotv19 = pn_coeffs.s1dot_coeffs.asdotv1950*spin_vars.lNxS1y
    sdotv18 = spin_vars.SxS12y*pn_coeffs.s1dot_coeffs.asdotv1841 + spin_vars.lNxS1y*(pn_coeffs.s1dot_coeffs.asdotv18421*spin_vars.lNS2 + pn_coeffs.s1dot_coeffs.asdotv18422*spin_vars.lNS1)
    sdotv17 = pn_coeffs.s1dot_coeffs.asdotv1730*spin_vars.lNxS1y
    sdotv16 = spin_vars.SxS12y*pn_coeffs.s1dot_coeffs.asdotv1621 + spin_vars.lNxS1y*(pn_coeffs.s1dot_coeffs.asdotv16221*spin_vars.lNS2 + pn_coeffs.s1dot_coeffs.asdotv16222*spin_vars.lNS1)
    sdotv15 = pn_coeffs.s1dot_coeffs.asdotv1510*spin_vars.lNxS1y
    s1doty = sdotv110*v_powers.v10 + sdotv15*v_powers.v5 + sdotv16*v_powers.v6 + sdotv17*v_powers.v7 + sdotv18*v_powers.v8 + sdotv19*v_powers.v9

    # z-component
    sdotv110 = spin_vars.SxS12z*pn_coeffs.s1dot_coeffs.asdotv11061 + spin_vars.lNxS1z*(pn_coeffs.s1dot_coeffs.asdotv110621*spin_vars.lNS1 + pn_coeffs.s1dot_coeffs.asdotv110622*spin_vars.lNS2)
    sdotv19 = pn_coeffs.s1dot_coeffs.asdotv1950*spin_vars.lNxS1z
    sdotv18 = spin_vars.SxS12z*pn_coeffs.s1dot_coeffs.asdotv1841 + spin_vars.lNxS1z*(pn_coeffs.s1dot_coeffs.asdotv18421*spin_vars.lNS2 + pn_coeffs.s1dot_coeffs.asdotv18422*spin_vars.lNS1)
    sdotv17 = pn_coeffs.s1dot_coeffs.asdotv1730*spin_vars.lNxS1z
    sdotv16 = spin_vars.SxS12z*pn_coeffs.s1dot_coeffs.asdotv1621 + spin_vars.lNxS1z*(pn_coeffs.s1dot_coeffs.asdotv16221*spin_vars.lNS2 + pn_coeffs.s1dot_coeffs.asdotv16222*spin_vars.lNS1)
    sdotv15 = pn_coeffs.s1dot_coeffs.asdotv1510*spin_vars.lNxS1z
    s1dotz = sdotv110*v_powers.v10 + sdotv15*v_powers.v5 + sdotv16*v_powers.v6 + sdotv17*v_powers.v7 + sdotv18*v_powers.v8 + sdotv19*v_powers.v9

    return [s1dotx,s1doty,s1dotz]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef list compute_s2dot_opt(vpowers v_powers, spinVars spin_vars, PNCoeffs pn_coeffs):
    """
    Evolution equation for S2dot up to 4PN order
    """

    # Initialize memory arrays
    cdef double sdotv210
    cdef double sdotv29
    cdef double sdotv28
    cdef double sdotv27
    cdef double sdotv26
    cdef double sdotv25
    cdef double s1dotx, s1doty, s1dotz

    # coefficients entering s2dot

    # x-component
    sdotv210 = spin_vars.SxS12x*pn_coeffs.s2dot_coeffs.asdotv21061 + spin_vars.lNxS2x*(pn_coeffs.s2dot_coeffs.asdotv210621*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv210622*spin_vars.lNS2)
    sdotv29 = pn_coeffs.s2dot_coeffs.asdotv2950*spin_vars.lNxS2x
    sdotv28 = spin_vars.SxS12x*pn_coeffs.s2dot_coeffs.asdotv2841 + spin_vars.lNxS2x*(pn_coeffs.s2dot_coeffs.asdotv28421*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv28422*spin_vars.lNS2)
    sdotv27 = pn_coeffs.s2dot_coeffs.asdotv2730*spin_vars.lNxS2x
    sdotv26 = spin_vars.SxS12x*pn_coeffs.s2dot_coeffs.asdotv2621 + spin_vars.lNxS2x*(pn_coeffs.s2dot_coeffs.asdotv26221*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv26222*spin_vars.lNS2)
    sdotv25 = pn_coeffs.s2dot_coeffs.asdotv2510*spin_vars.lNxS2x
    s2dotx = sdotv210*v_powers.v10 + sdotv25*v_powers.v5 + sdotv26*v_powers.v6 + sdotv27*v_powers.v7 + sdotv28*v_powers.v8 + sdotv29*v_powers.v9

    # y-component
    sdotv210 = spin_vars.SxS12y*pn_coeffs.s2dot_coeffs.asdotv21061 + spin_vars.lNxS2y*(pn_coeffs.s2dot_coeffs.asdotv210621*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv210622*spin_vars.lNS2)
    sdotv29 = pn_coeffs.s2dot_coeffs.asdotv2950*spin_vars.lNxS2y
    sdotv28 = spin_vars.SxS12y*pn_coeffs.s2dot_coeffs.asdotv2841 + spin_vars.lNxS2y*(pn_coeffs.s2dot_coeffs.asdotv28421*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv28422*spin_vars.lNS2)
    sdotv27 = pn_coeffs.s2dot_coeffs.asdotv2730*spin_vars.lNxS2y
    sdotv26 = spin_vars.SxS12y*pn_coeffs.s2dot_coeffs.asdotv2621 + spin_vars.lNxS2y*(pn_coeffs.s2dot_coeffs.asdotv26221*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv26222*spin_vars.lNS2)
    sdotv25 = pn_coeffs.s2dot_coeffs.asdotv2510*spin_vars.lNxS2y
    s2doty = sdotv210*v_powers.v10 + sdotv25*v_powers.v5 + sdotv26*v_powers.v6 + sdotv27*v_powers.v7 + sdotv28*v_powers.v8 + sdotv29*v_powers.v9

    # z-component
    sdotv210 = spin_vars.SxS12z*pn_coeffs.s2dot_coeffs.asdotv21061 + spin_vars.lNxS2z*(pn_coeffs.s2dot_coeffs.asdotv210621*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv210622*spin_vars.lNS2)
    sdotv29 = pn_coeffs.s2dot_coeffs.asdotv2950*spin_vars.lNxS2z
    sdotv28 = spin_vars.SxS12z*pn_coeffs.s2dot_coeffs.asdotv2841 + spin_vars.lNxS2z*(pn_coeffs.s2dot_coeffs.asdotv28421*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv28422*spin_vars.lNS2)
    sdotv27 = pn_coeffs.s2dot_coeffs.asdotv2730*spin_vars.lNxS2z
    sdotv26 = spin_vars.SxS12z*pn_coeffs.s2dot_coeffs.asdotv2621 + spin_vars.lNxS2z*(pn_coeffs.s2dot_coeffs.asdotv26221*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv26222*spin_vars.lNS2)
    sdotv25 = pn_coeffs.s2dot_coeffs.asdotv2510*spin_vars.lNxS2z
    s2dotz = sdotv210*v_powers.v10 + sdotv25*v_powers.v5 + sdotv26*v_powers.v6 + sdotv27*v_powers.v7 + sdotv28*v_powers.v8 + sdotv29*v_powers.v9

    return [s2dotx,s2doty,s2dotz]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef list compute_lNdot_opt(vpowers v_powers, spinVars spin_vars, PNCoeffs pn_coeffs):
    """
    Evolution equation for lNdot up to 4PN order
    """
    # Initialize memory arrays
    cdef double lNdotv11
    cdef double lNdotv10
    cdef double lNdotv9
    cdef double lNdotv8
    cdef double lNdotv7
    cdef double lNdotv6
    cdef double lNdotx, lNdoty, lNdotz

    # coefficients entering lNdot

    # x-component
    lNdotv11 = spin_vars.SxS12x*pn_coeffs.lNdot_coeffs.alNdotv1162 + pn_coeffs.lNdot_coeffs.alNdotv1161*spin_vars.lNx*spin_vars.lNSxS12 + spin_vars.lNxS1x*(pn_coeffs.lNdot_coeffs.alNdotv11641*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv11642*spin_vars.lNS1) + spin_vars.lNxS2x*(pn_coeffs.lNdot_coeffs.alNdotv11631*spin_vars.lNS1 + pn_coeffs.lNdot_coeffs.alNdotv11632*spin_vars.lNS2)
    lNdotv10 = pn_coeffs.lNdot_coeffs.alNdotv1051*spin_vars.lNxS2x + pn_coeffs.lNdot_coeffs.alNdotv1052*spin_vars.lNxS1x
    lNdotv9 = spin_vars.SxS12x*pn_coeffs.lNdot_coeffs.alNdotv942 + pn_coeffs.lNdot_coeffs.alNdotv941*spin_vars.lNx*spin_vars.lNSxS12 + spin_vars.lNxS1x*(pn_coeffs.lNdot_coeffs.alNdotv9431*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv9432*spin_vars.lNS1) + spin_vars.lNxS2x*(pn_coeffs.lNdot_coeffs.alNdotv9441*spin_vars.lNS1 + pn_coeffs.lNdot_coeffs.alNdotv9442*spin_vars.lNS2)
    lNdotv8 = pn_coeffs.lNdot_coeffs.alNdotv831*spin_vars.lNxS1x + pn_coeffs.lNdot_coeffs.alNdotv832*spin_vars.lNxS2x
    lNdotv7 = spin_vars.lNxS1x*(pn_coeffs.lNdot_coeffs.alNdotv7221*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv7222*spin_vars.lNS1) + spin_vars.lNxS2x*(pn_coeffs.lNdot_coeffs.alNdotv7211*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv7212*spin_vars.lNS1)
    lNdotv6 = pn_coeffs.lNdot_coeffs.alNdotv611*spin_vars.lNxS1x + pn_coeffs.lNdot_coeffs.alNdotv612*spin_vars.lNxS2x
    lNdotx = lNdotv10*v_powers.v10 + lNdotv11*v_powers.v11 + lNdotv6*v_powers.v6 + lNdotv7*v_powers.v7 + lNdotv8*v_powers.v8 + lNdotv9*v_powers.v9

    # y-component
    lNdotv11 = spin_vars.SxS12y*pn_coeffs.lNdot_coeffs.alNdotv1162 + pn_coeffs.lNdot_coeffs.alNdotv1161*spin_vars.lNy*spin_vars.lNSxS12 + spin_vars.lNxS1y*(pn_coeffs.lNdot_coeffs.alNdotv11641*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv11642*spin_vars.lNS1) + spin_vars.lNxS2y*(pn_coeffs.lNdot_coeffs.alNdotv11631*spin_vars.lNS1 + pn_coeffs.lNdot_coeffs.alNdotv11632*spin_vars.lNS2)
    lNdotv10 = pn_coeffs.lNdot_coeffs.alNdotv1051*spin_vars.lNxS2y + pn_coeffs.lNdot_coeffs.alNdotv1052*spin_vars.lNxS1y
    lNdotv9 = spin_vars.SxS12y*pn_coeffs.lNdot_coeffs.alNdotv942 + pn_coeffs.lNdot_coeffs.alNdotv941*spin_vars.lNy*spin_vars.lNSxS12 + spin_vars.lNxS1y*(pn_coeffs.lNdot_coeffs.alNdotv9431*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv9432*spin_vars.lNS1) + spin_vars.lNxS2y*(pn_coeffs.lNdot_coeffs.alNdotv9441*spin_vars.lNS1 + pn_coeffs.lNdot_coeffs.alNdotv9442*spin_vars.lNS2)
    lNdotv8 = pn_coeffs.lNdot_coeffs.alNdotv831*spin_vars.lNxS1y + pn_coeffs.lNdot_coeffs.alNdotv832*spin_vars.lNxS2y
    lNdotv7 = spin_vars.lNxS1y*(pn_coeffs.lNdot_coeffs.alNdotv7221*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv7222*spin_vars.lNS1) + spin_vars.lNxS2y*(pn_coeffs.lNdot_coeffs.alNdotv7211*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv7212*spin_vars.lNS1)
    lNdotv6 = pn_coeffs.lNdot_coeffs.alNdotv611*spin_vars.lNxS1y + pn_coeffs.lNdot_coeffs.alNdotv612*spin_vars.lNxS2y
    lNdoty = lNdotv10*v_powers.v10 + lNdotv11*v_powers.v11 + lNdotv6*v_powers.v6 + lNdotv7*v_powers.v7 + lNdotv8*v_powers.v8 + lNdotv9*v_powers.v9

    # z-component
    lNdotv11 = spin_vars.SxS12z*pn_coeffs.lNdot_coeffs.alNdotv1162 + pn_coeffs.lNdot_coeffs.alNdotv1161*spin_vars.lNz*spin_vars.lNSxS12 + spin_vars.lNxS1z*(pn_coeffs.lNdot_coeffs.alNdotv11641*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv11642*spin_vars.lNS1) + spin_vars.lNxS2z*(pn_coeffs.lNdot_coeffs.alNdotv11631*spin_vars.lNS1 + pn_coeffs.lNdot_coeffs.alNdotv11632*spin_vars.lNS2)
    lNdotv10 = pn_coeffs.lNdot_coeffs.alNdotv1051*spin_vars.lNxS2z + pn_coeffs.lNdot_coeffs.alNdotv1052*spin_vars.lNxS1z
    lNdotv9 = spin_vars.SxS12z*pn_coeffs.lNdot_coeffs.alNdotv942 + pn_coeffs.lNdot_coeffs.alNdotv941*spin_vars.lNz*spin_vars.lNSxS12 + spin_vars.lNxS1z*(pn_coeffs.lNdot_coeffs.alNdotv9431*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv9432*spin_vars.lNS1) + spin_vars.lNxS2z*(pn_coeffs.lNdot_coeffs.alNdotv9441*spin_vars.lNS1 + pn_coeffs.lNdot_coeffs.alNdotv9442*spin_vars.lNS2)
    lNdotv8 = pn_coeffs.lNdot_coeffs.alNdotv831*spin_vars.lNxS1z + pn_coeffs.lNdot_coeffs.alNdotv832*spin_vars.lNxS2z
    lNdotv7 = spin_vars.lNxS1z*(pn_coeffs.lNdot_coeffs.alNdotv7221*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv7222*spin_vars.lNS1) + spin_vars.lNxS2z*(pn_coeffs.lNdot_coeffs.alNdotv7211*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv7212*spin_vars.lNS1)
    lNdotv6 = pn_coeffs.lNdot_coeffs.alNdotv611*spin_vars.lNxS1z + pn_coeffs.lNdot_coeffs.alNdotv612*spin_vars.lNxS2z
    lNdotz = lNdotv10*v_powers.v10 + lNdotv11*v_powers.v11 + lNdotv6*v_powers.v6 + lNdotv7*v_powers.v7 + lNdotv8*v_powers.v8 + lNdotv9*v_powers.v9

    return [lNdotx,lNdoty,lNdotz]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef double compute_omegadot(double nu, vpowers v_powers, spinVars spin_vars, PNCoeffs pn_coeffs):
    """
    Evolution equation for lNdot up to 4PN order
    """

    # coefficients entering omegadot
    cdef double omegadotv8 = spin_vars.SS12*(pn_coeffs.omegadot_coeffs.aomegadotv8208 + pn_coeffs.omegadot_coeffs.aomegadotv828 + pn_coeffs.omegadot_coeffs.aomegadotv8328) + spin_vars.Ssq1*(pn_coeffs.omegadot_coeffs.aomegadotv8138 + pn_coeffs.omegadot_coeffs.aomegadotv8248 + pn_coeffs.omegadot_coeffs.aomegadotv8298 + pn_coeffs.omegadot_coeffs.aomegadotv8378 + pn_coeffs.omegadot_coeffs.aomegadotv8438 + pn_coeffs.omegadot_coeffs.aomegadotv878) + spin_vars.Ssq2*(pn_coeffs.omegadot_coeffs.aomegadotv8148 + pn_coeffs.omegadot_coeffs.aomegadotv8168 + pn_coeffs.omegadot_coeffs.aomegadotv8188 + pn_coeffs.omegadot_coeffs.aomegadotv8258 + pn_coeffs.omegadot_coeffs.aomegadotv8308 + pn_coeffs.omegadot_coeffs.aomegadotv8388 + pn_coeffs.omegadot_coeffs.aomegadotv8448 + pn_coeffs.omegadot_coeffs.aomegadotv888) + spin_vars.lNS1*(pn_coeffs.omegadot_coeffs.aomegadotv8118 + pn_coeffs.omegadot_coeffs.aomegadotv8358 + pn_coeffs.omegadot_coeffs.aomegadotv8418 + pn_coeffs.omegadot_coeffs.aomegadotv858 + spin_vars.lNS2*(pn_coeffs.omegadot_coeffs.aomegadotv818 + pn_coeffs.omegadot_coeffs.aomegadotv8198 + pn_coeffs.omegadot_coeffs.aomegadotv8318)) + spin_vars.lNS2*(pn_coeffs.omegadot_coeffs.aomegadotv8128 + pn_coeffs.omegadot_coeffs.aomegadotv8238 + pn_coeffs.omegadot_coeffs.aomegadotv8288 + pn_coeffs.omegadot_coeffs.aomegadotv8368 + pn_coeffs.omegadot_coeffs.aomegadotv8428 + pn_coeffs.omegadot_coeffs.aomegadotv868) + spin_vars.lNSsq1*(pn_coeffs.omegadot_coeffs.aomegadotv8218 + pn_coeffs.omegadot_coeffs.aomegadotv8268 + pn_coeffs.omegadot_coeffs.aomegadotv8338 + pn_coeffs.omegadot_coeffs.aomegadotv838 + pn_coeffs.omegadot_coeffs.aomegadotv8398 + pn_coeffs.omegadot_coeffs.aomegadotv898) + spin_vars.lNSsq2*(pn_coeffs.omegadot_coeffs.aomegadotv8108 + pn_coeffs.omegadot_coeffs.aomegadotv8158 + pn_coeffs.omegadot_coeffs.aomegadotv8178 + pn_coeffs.omegadot_coeffs.aomegadotv8228 + pn_coeffs.omegadot_coeffs.aomegadotv8278 + pn_coeffs.omegadot_coeffs.aomegadotv8348 + pn_coeffs.omegadot_coeffs.aomegadotv8408 + pn_coeffs.omegadot_coeffs.aomegadotv848)
    cdef double omegadotv7 = spin_vars.SS12*pn_coeffs.omegadot_coeffs.aomegadotv7117 + spin_vars.Ssq1*(pn_coeffs.omegadot_coeffs.aomegadotv7157 + pn_coeffs.omegadot_coeffs.aomegadotv7207) + spin_vars.Ssq2*(pn_coeffs.omegadot_coeffs.aomegadotv7167 + pn_coeffs.omegadot_coeffs.aomegadotv7217 + pn_coeffs.omegadot_coeffs.aomegadotv777 + pn_coeffs.omegadot_coeffs.aomegadotv797) + pn_coeffs.omegadot_coeffs.aomegadotv717 + pn_coeffs.omegadot_coeffs.aomegadotv7227 + pn_coeffs.omegadot_coeffs.aomegadotv7277 + spin_vars.lNS1*(pn_coeffs.omegadot_coeffs.aomegadotv7107*spin_vars.lNS2 + pn_coeffs.omegadot_coeffs.aomegadotv7237 + pn_coeffs.omegadot_coeffs.aomegadotv7257 + pn_coeffs.omegadot_coeffs.aomegadotv727 + pn_coeffs.omegadot_coeffs.aomegadotv7287 + pn_coeffs.omegadot_coeffs.aomegadotv7307 + pn_coeffs.omegadot_coeffs.aomegadotv747) + spin_vars.lNS2*(pn_coeffs.omegadot_coeffs.aomegadotv7127 + pn_coeffs.omegadot_coeffs.aomegadotv7177 + pn_coeffs.omegadot_coeffs.aomegadotv7247 + pn_coeffs.omegadot_coeffs.aomegadotv7267 + pn_coeffs.omegadot_coeffs.aomegadotv7297 + pn_coeffs.omegadot_coeffs.aomegadotv7317 + pn_coeffs.omegadot_coeffs.aomegadotv737 + pn_coeffs.omegadot_coeffs.aomegadotv757) + spin_vars.lNSsq1*(pn_coeffs.omegadot_coeffs.aomegadotv7137 + pn_coeffs.omegadot_coeffs.aomegadotv7187) + spin_vars.lNSsq2*(pn_coeffs.omegadot_coeffs.aomegadotv7147 + pn_coeffs.omegadot_coeffs.aomegadotv7197 + pn_coeffs.omegadot_coeffs.aomegadotv767 + pn_coeffs.omegadot_coeffs.aomegadotv787)
    cdef double omegadotv6 = spin_vars.SS12*(pn_coeffs.omegadot_coeffs.aomegadotv6246 + pn_coeffs.omegadot_coeffs.aomegadotv666) + spin_vars.Ssq1*(pn_coeffs.omegadot_coeffs.aomegadotv6116 + pn_coeffs.omegadot_coeffs.aomegadotv6166 + pn_coeffs.omegadot_coeffs.aomegadotv6276 + pn_coeffs.omegadot_coeffs.aomegadotv6316) + spin_vars.Ssq2*(pn_coeffs.omegadot_coeffs.aomegadotv6126 + pn_coeffs.omegadot_coeffs.aomegadotv6176 + pn_coeffs.omegadot_coeffs.aomegadotv6196 + pn_coeffs.omegadot_coeffs.aomegadotv6216 + pn_coeffs.omegadot_coeffs.aomegadotv6286 + pn_coeffs.omegadot_coeffs.aomegadotv6326) + pn_coeffs.omegadot_coeffs.aomegadotv616 + pn_coeffs.omegadot_coeffs.aomegadotv626 + pn_coeffs.omegadot_coeffs.aomegadotv6336 + pn_coeffs.omegadot_coeffs.aomegadotv6346 + pn_coeffs.omegadot_coeffs.aomegadotv6356 + pn_coeffs.omegadot_coeffs.aomegadotv6366 + pn_coeffs.omegadot_coeffs.aomegadotv6376 + pn_coeffs.omegadot_coeffs.aomegadotv656 + pn_coeffs.omegadot_coeffs.aomegadotvLog6*v_powers.logv + spin_vars.lNS1*(pn_coeffs.omegadot_coeffs.aomegadotv6156 + pn_coeffs.omegadot_coeffs.aomegadotv696 + spin_vars.lNS2*(pn_coeffs.omegadot_coeffs.aomegadotv6226 + pn_coeffs.omegadot_coeffs.aomegadotv636)) + spin_vars.lNS2*(pn_coeffs.omegadot_coeffs.aomegadotv6106 + pn_coeffs.omegadot_coeffs.aomegadotv6236 + pn_coeffs.omegadot_coeffs.aomegadotv646) + spin_vars.lNSsq1*(pn_coeffs.omegadot_coeffs.aomegadotv6136 + pn_coeffs.omegadot_coeffs.aomegadotv6256 + pn_coeffs.omegadot_coeffs.aomegadotv6296 + pn_coeffs.omegadot_coeffs.aomegadotv676) + spin_vars.lNSsq2*(pn_coeffs.omegadot_coeffs.aomegadotv6146 + pn_coeffs.omegadot_coeffs.aomegadotv6186 + pn_coeffs.omegadot_coeffs.aomegadotv6206 + pn_coeffs.omegadot_coeffs.aomegadotv6266 + pn_coeffs.omegadot_coeffs.aomegadotv6306 + pn_coeffs.omegadot_coeffs.aomegadotv686)
    cdef double omegadotv5 = pn_coeffs.omegadot_coeffs.aomegadotv515 + pn_coeffs.omegadot_coeffs.aomegadotv585 + spin_vars.lNS1*(pn_coeffs.omegadot_coeffs.aomegadotv5115 + pn_coeffs.omegadot_coeffs.aomegadotv525 + pn_coeffs.omegadot_coeffs.aomegadotv545 + pn_coeffs.omegadot_coeffs.aomegadotv595) + spin_vars.lNS2*(pn_coeffs.omegadot_coeffs.aomegadotv5105 + pn_coeffs.omegadot_coeffs.aomegadotv5125 + pn_coeffs.omegadot_coeffs.aomegadotv535 + pn_coeffs.omegadot_coeffs.aomegadotv555 + pn_coeffs.omegadot_coeffs.aomegadotv565 + pn_coeffs.omegadot_coeffs.aomegadotv575)
    cdef double omegadotv4 = spin_vars.SS12*pn_coeffs.omegadot_coeffs.aomegadotv474 + spin_vars.Ssq1*(pn_coeffs.omegadot_coeffs.aomegadotv4104 + pn_coeffs.omegadot_coeffs.aomegadotv4144) + spin_vars.Ssq2*(pn_coeffs.omegadot_coeffs.aomegadotv4114 + pn_coeffs.omegadot_coeffs.aomegadotv4154 + pn_coeffs.omegadot_coeffs.aomegadotv434 + pn_coeffs.omegadot_coeffs.aomegadotv454) + pn_coeffs.omegadot_coeffs.aomegadotv414 + pn_coeffs.omegadot_coeffs.aomegadotv4164 + pn_coeffs.omegadot_coeffs.aomegadotv4174 + pn_coeffs.omegadot_coeffs.aomegadotv464*spin_vars.lNS1*spin_vars.lNS2 + spin_vars.lNSsq1*(pn_coeffs.omegadot_coeffs.aomegadotv4124 + pn_coeffs.omegadot_coeffs.aomegadotv484) + spin_vars.lNSsq2*(pn_coeffs.omegadot_coeffs.aomegadotv4134 + pn_coeffs.omegadot_coeffs.aomegadotv424 + pn_coeffs.omegadot_coeffs.aomegadotv444 + pn_coeffs.omegadot_coeffs.aomegadotv494)
    cdef double omegadotv3 = pn_coeffs.omegadot_coeffs.aomegadotv313 + spin_vars.lNS1*(pn_coeffs.omegadot_coeffs.aomegadotv323 + pn_coeffs.omegadot_coeffs.aomegadotv343) + spin_vars.lNS2*(pn_coeffs.omegadot_coeffs.aomegadotv333 + pn_coeffs.omegadot_coeffs.aomegadotv353 + pn_coeffs.omegadot_coeffs.aomegadotv363)
    cdef double omegadotv2 = pn_coeffs.omegadot_coeffs.aomegadotv212 + pn_coeffs.omegadot_coeffs.aomegadotv222
    cdef double omegadotv0 = pn_coeffs.omegadot_coeffs.aomegadotv0
    cdef double fact0 = 96.*nu*v_powers.v11/5.
    cdef double omegadotPy = fact0*(omegadotv0 + omegadotv2*v_powers.v2 + omegadotv3*v_powers.v3 + omegadotv4*v_powers.v4 + omegadotv5*v_powers.v5 + omegadotv6*v_powers.v6 + omegadotv7*v_powers.v7 + omegadotv8*v_powers.v8)

    return omegadotPy





@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef list compute_lhat_opt(double nu, vpowers v_powers, spinVars spin_vars, PNCoeffs pn_coeffs):
    """
    Equation to compute Lhat up to 4PN order
    """
    # Initialize memory arrays
    cdef double lhatv8
    cdef double lhatv7
    cdef double lhatv6
    cdef double lhatv5
    cdef double lhatv4
    cdef double lhatv3
    cdef double lhatv2
    cdef double lhatv0
    cdef double fact0
    cdef double lhatPy
    cdef double lhatx, lhaty, lhatz

    # coefficients entering Lhat
    fact0 = nu/v_powers.v1

    # x-component
    lhatv8 = spin_vars.lNx*(spin_vars.SS12*(pn_coeffs.lhat_coeffs.alhatv8318 + pn_coeffs.lhat_coeffs.alhatv8448 + pn_coeffs.lhat_coeffs.alhatv868) + spin_vars.Ssq1*(pn_coeffs.lhat_coeffs.alhatv8128 + pn_coeffs.lhat_coeffs.alhatv8208 + pn_coeffs.lhat_coeffs.alhatv8498 + pn_coeffs.lhat_coeffs.alhatv8548 + pn_coeffs.lhat_coeffs.alhatv878) + spin_vars.Ssq2*(pn_coeffs.lhat_coeffs.alhatv8148 + pn_coeffs.lhat_coeffs.alhatv8178 + pn_coeffs.lhat_coeffs.alhatv8248 + pn_coeffs.lhat_coeffs.alhatv8268 + pn_coeffs.lhat_coeffs.alhatv8288 + pn_coeffs.lhat_coeffs.alhatv8358 + pn_coeffs.lhat_coeffs.alhatv8378 + pn_coeffs.lhat_coeffs.alhatv8518 + pn_coeffs.lhat_coeffs.alhatv8568) + pn_coeffs.lhat_coeffs.alhatv818 + pn_coeffs.lhat_coeffs.alhatv8388 + pn_coeffs.lhat_coeffs.alhatv8398 + pn_coeffs.lhat_coeffs.alhatv8418 + pn_coeffs.lhat_coeffs.alhatv8578 + pn_coeffs.lhat_coeffs.alhatv8588 + pn_coeffs.lhat_coeffs.alhatv8598 + pn_coeffs.lhat_coeffs.alhatv8608 + pn_coeffs.lhat_coeffs.alhatv8618 + pn_coeffs.lhat_coeffs.alhatvLog8*v_powers.logv + spin_vars.lNSsq1*(pn_coeffs.lhat_coeffs.alhatv8188 + pn_coeffs.lhat_coeffs.alhatv828 + pn_coeffs.lhat_coeffs.alhatv8468 + pn_coeffs.lhat_coeffs.alhatv8528 + pn_coeffs.lhat_coeffs.alhatv898) + spin_vars.lNSsq2*(pn_coeffs.lhat_coeffs.alhatv8108 + pn_coeffs.lhat_coeffs.alhatv8158 + pn_coeffs.lhat_coeffs.alhatv8238 + pn_coeffs.lhat_coeffs.alhatv8258 + pn_coeffs.lhat_coeffs.alhatv8278 + pn_coeffs.lhat_coeffs.alhatv8348 + pn_coeffs.lhat_coeffs.alhatv8368 + pn_coeffs.lhat_coeffs.alhatv8478 + pn_coeffs.lhat_coeffs.alhatv8538)) + spin_vars.lNS1*(spin_vars.S1x*(pn_coeffs.lhat_coeffs.alhatv8118 + pn_coeffs.lhat_coeffs.alhatv8198 + pn_coeffs.lhat_coeffs.alhatv8428 + pn_coeffs.lhat_coeffs.alhatv848 + pn_coeffs.lhat_coeffs.alhatv8488) + spin_vars.S2x*(pn_coeffs.lhat_coeffs.alhatv8328 + pn_coeffs.lhat_coeffs.alhatv8458 + pn_coeffs.lhat_coeffs.alhatv888) + spin_vars.lNx*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv8298 + pn_coeffs.lhat_coeffs.alhatv838 + pn_coeffs.lhat_coeffs.alhatv8408)) + spin_vars.lNS2*(spin_vars.S1x*(pn_coeffs.lhat_coeffs.alhatv8308 + pn_coeffs.lhat_coeffs.alhatv8438 + pn_coeffs.lhat_coeffs.alhatv858) + spin_vars.S2x*(pn_coeffs.lhat_coeffs.alhatv8138 + pn_coeffs.lhat_coeffs.alhatv8168 + pn_coeffs.lhat_coeffs.alhatv8218 + pn_coeffs.lhat_coeffs.alhatv8228 + pn_coeffs.lhat_coeffs.alhatv8338 + pn_coeffs.lhat_coeffs.alhatv8508 + pn_coeffs.lhat_coeffs.alhatv8558))
    lhatv7 = spin_vars.S1x*(pn_coeffs.lhat_coeffs.alhatv7157 + pn_coeffs.lhat_coeffs.alhatv7197 + pn_coeffs.lhat_coeffs.alhatv7237 + pn_coeffs.lhat_coeffs.alhatv7277 + pn_coeffs.lhat_coeffs.alhatv737 + pn_coeffs.lhat_coeffs.alhatv777) + spin_vars.S2x*(pn_coeffs.lhat_coeffs.alhatv7107 + pn_coeffs.lhat_coeffs.alhatv7127 + pn_coeffs.lhat_coeffs.alhatv7167 + pn_coeffs.lhat_coeffs.alhatv7207 + pn_coeffs.lhat_coeffs.alhatv7247 + pn_coeffs.lhat_coeffs.alhatv7287 + pn_coeffs.lhat_coeffs.alhatv747 + pn_coeffs.lhat_coeffs.alhatv787) + spin_vars.lNx*spin_vars.lNS1*(pn_coeffs.lhat_coeffs.alhatv7137 + pn_coeffs.lhat_coeffs.alhatv717 + pn_coeffs.lhat_coeffs.alhatv7177 + pn_coeffs.lhat_coeffs.alhatv7217 + pn_coeffs.lhat_coeffs.alhatv7257 + pn_coeffs.lhat_coeffs.alhatv757) + spin_vars.lNx*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv7117 + pn_coeffs.lhat_coeffs.alhatv7147 + pn_coeffs.lhat_coeffs.alhatv7187 + pn_coeffs.lhat_coeffs.alhatv7227 + pn_coeffs.lhat_coeffs.alhatv7267 + pn_coeffs.lhat_coeffs.alhatv727 + pn_coeffs.lhat_coeffs.alhatv767 + pn_coeffs.lhat_coeffs.alhatv797)
    lhatv6 = spin_vars.lNx*(spin_vars.SS12*pn_coeffs.lhat_coeffs.alhatv676 + spin_vars.Ssq1*(pn_coeffs.lhat_coeffs.alhatv6136 + pn_coeffs.lhat_coeffs.alhatv6166 + pn_coeffs.lhat_coeffs.alhatv6206) + spin_vars.Ssq2*(pn_coeffs.lhat_coeffs.alhatv6156 + pn_coeffs.lhat_coeffs.alhatv6236 + pn_coeffs.lhat_coeffs.alhatv6266 + pn_coeffs.lhat_coeffs.alhatv6326 + pn_coeffs.lhat_coeffs.alhatv696) + pn_coeffs.lhat_coeffs.alhatv616 + pn_coeffs.lhat_coeffs.alhatv6336 + pn_coeffs.lhat_coeffs.alhatv6346 + pn_coeffs.lhat_coeffs.alhatv6356 + pn_coeffs.lhat_coeffs.alhatv6366 + spin_vars.lNSsq1*(pn_coeffs.lhat_coeffs.alhatv6106 + pn_coeffs.lhat_coeffs.alhatv6186 + pn_coeffs.lhat_coeffs.alhatv626) + spin_vars.lNSsq2*(pn_coeffs.lhat_coeffs.alhatv6116 + pn_coeffs.lhat_coeffs.alhatv6216 + pn_coeffs.lhat_coeffs.alhatv6246 + pn_coeffs.lhat_coeffs.alhatv6286 + pn_coeffs.lhat_coeffs.alhatv646)) + spin_vars.lNS1*(spin_vars.S1x*(pn_coeffs.lhat_coeffs.alhatv6126 + pn_coeffs.lhat_coeffs.alhatv6196 + pn_coeffs.lhat_coeffs.alhatv656) + spin_vars.S2x*(pn_coeffs.lhat_coeffs.alhatv6306 + pn_coeffs.lhat_coeffs.alhatv686) + spin_vars.lNx*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv6276 + pn_coeffs.lhat_coeffs.alhatv636)) + spin_vars.lNS2*(spin_vars.S1x*(pn_coeffs.lhat_coeffs.alhatv6296 + pn_coeffs.lhat_coeffs.alhatv666) + spin_vars.S2x*(pn_coeffs.lhat_coeffs.alhatv6146 + pn_coeffs.lhat_coeffs.alhatv6176 + pn_coeffs.lhat_coeffs.alhatv6226 + pn_coeffs.lhat_coeffs.alhatv6256 + pn_coeffs.lhat_coeffs.alhatv6316))
    lhatv5 = spin_vars.S1x*(pn_coeffs.lhat_coeffs.alhatv5155 + pn_coeffs.lhat_coeffs.alhatv5195 + pn_coeffs.lhat_coeffs.alhatv535 + pn_coeffs.lhat_coeffs.alhatv575) + spin_vars.S2x*(pn_coeffs.lhat_coeffs.alhatv5105 + pn_coeffs.lhat_coeffs.alhatv5125 + pn_coeffs.lhat_coeffs.alhatv5165 + pn_coeffs.lhat_coeffs.alhatv5205 + pn_coeffs.lhat_coeffs.alhatv545 + pn_coeffs.lhat_coeffs.alhatv585) + spin_vars.lNx*spin_vars.lNS1*(pn_coeffs.lhat_coeffs.alhatv5135 + pn_coeffs.lhat_coeffs.alhatv515 + pn_coeffs.lhat_coeffs.alhatv5175 + pn_coeffs.lhat_coeffs.alhatv555) + spin_vars.lNx*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv5115 + pn_coeffs.lhat_coeffs.alhatv5145 + pn_coeffs.lhat_coeffs.alhatv5185 + pn_coeffs.lhat_coeffs.alhatv525 + pn_coeffs.lhat_coeffs.alhatv565 + pn_coeffs.lhat_coeffs.alhatv595)
    lhatv4 = spin_vars.lNx*(spin_vars.SS12*pn_coeffs.lhat_coeffs.alhatv4144 + spin_vars.Ssq1*pn_coeffs.lhat_coeffs.alhatv444 + spin_vars.Ssq2*(pn_coeffs.lhat_coeffs.alhatv4104 + pn_coeffs.lhat_coeffs.alhatv4174 + pn_coeffs.lhat_coeffs.alhatv474) + pn_coeffs.lhat_coeffs.alhatv414 + pn_coeffs.lhat_coeffs.alhatv4184 + pn_coeffs.lhat_coeffs.alhatv4194 + pn_coeffs.lhat_coeffs.alhatv424*spin_vars.lNSsq1 + spin_vars.lNSsq2*(pn_coeffs.lhat_coeffs.alhatv4124 + pn_coeffs.lhat_coeffs.alhatv454 + pn_coeffs.lhat_coeffs.alhatv484)) + spin_vars.lNS1*(spin_vars.S1x*pn_coeffs.lhat_coeffs.alhatv434 + spin_vars.S2x*pn_coeffs.lhat_coeffs.alhatv4154 + pn_coeffs.lhat_coeffs.alhatv4114*spin_vars.lNx*spin_vars.lNS2) + spin_vars.lNS2*(spin_vars.S1x*pn_coeffs.lhat_coeffs.alhatv4134 + spin_vars.S2x*(pn_coeffs.lhat_coeffs.alhatv4164 + pn_coeffs.lhat_coeffs.alhatv464 + pn_coeffs.lhat_coeffs.alhatv494))
    lhatv3 = spin_vars.S1x*(pn_coeffs.lhat_coeffs.alhatv333 + pn_coeffs.lhat_coeffs.alhatv373) + spin_vars.S2x*(pn_coeffs.lhat_coeffs.alhatv343 + pn_coeffs.lhat_coeffs.alhatv383 + pn_coeffs.lhat_coeffs.alhatv393) + spin_vars.lNx*spin_vars.lNS1*(pn_coeffs.lhat_coeffs.alhatv313 + pn_coeffs.lhat_coeffs.alhatv353) + spin_vars.lNx*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv3103 + pn_coeffs.lhat_coeffs.alhatv3113 + pn_coeffs.lhat_coeffs.alhatv323 + pn_coeffs.lhat_coeffs.alhatv363)
    lhatv2 = spin_vars.lNx*(pn_coeffs.lhat_coeffs.alhatv212 + pn_coeffs.lhat_coeffs.alhatv222)
    lhatv0 = pn_coeffs.lhat_coeffs.alhatv01*spin_vars.lNx
    lhatx = fact0*(lhatv0 + lhatv2*v_powers.v2 + lhatv3*v_powers.v3 + lhatv4*v_powers.v4 + lhatv5*v_powers.v5 + lhatv6*v_powers.v6 + lhatv7*v_powers.v7 + lhatv8*v_powers.v8)

    # y-component
    lhatv8 = spin_vars.lNy*(spin_vars.SS12*(pn_coeffs.lhat_coeffs.alhatv8318 + pn_coeffs.lhat_coeffs.alhatv8448 + pn_coeffs.lhat_coeffs.alhatv868) + spin_vars.Ssq1*(pn_coeffs.lhat_coeffs.alhatv8128 + pn_coeffs.lhat_coeffs.alhatv8208 + pn_coeffs.lhat_coeffs.alhatv8498 + pn_coeffs.lhat_coeffs.alhatv8548 + pn_coeffs.lhat_coeffs.alhatv878) + spin_vars.Ssq2*(pn_coeffs.lhat_coeffs.alhatv8148 + pn_coeffs.lhat_coeffs.alhatv8178 + pn_coeffs.lhat_coeffs.alhatv8248 + pn_coeffs.lhat_coeffs.alhatv8268 + pn_coeffs.lhat_coeffs.alhatv8288 + pn_coeffs.lhat_coeffs.alhatv8358 + pn_coeffs.lhat_coeffs.alhatv8378 + pn_coeffs.lhat_coeffs.alhatv8518 + pn_coeffs.lhat_coeffs.alhatv8568) + pn_coeffs.lhat_coeffs.alhatv818 + pn_coeffs.lhat_coeffs.alhatv8388 + pn_coeffs.lhat_coeffs.alhatv8398 + pn_coeffs.lhat_coeffs.alhatv8418 + pn_coeffs.lhat_coeffs.alhatv8578 + pn_coeffs.lhat_coeffs.alhatv8588 + pn_coeffs.lhat_coeffs.alhatv8598 + pn_coeffs.lhat_coeffs.alhatv8608 + pn_coeffs.lhat_coeffs.alhatv8618 + pn_coeffs.lhat_coeffs.alhatvLog8*v_powers.logv + spin_vars.lNSsq1*(pn_coeffs.lhat_coeffs.alhatv8188 + pn_coeffs.lhat_coeffs.alhatv828 + pn_coeffs.lhat_coeffs.alhatv8468 + pn_coeffs.lhat_coeffs.alhatv8528 + pn_coeffs.lhat_coeffs.alhatv898) + spin_vars.lNSsq2*(pn_coeffs.lhat_coeffs.alhatv8108 + pn_coeffs.lhat_coeffs.alhatv8158 + pn_coeffs.lhat_coeffs.alhatv8238 + pn_coeffs.lhat_coeffs.alhatv8258 + pn_coeffs.lhat_coeffs.alhatv8278 + pn_coeffs.lhat_coeffs.alhatv8348 + pn_coeffs.lhat_coeffs.alhatv8368 + pn_coeffs.lhat_coeffs.alhatv8478 + pn_coeffs.lhat_coeffs.alhatv8538)) + spin_vars.lNS1*(spin_vars.S1y*(pn_coeffs.lhat_coeffs.alhatv8118 + pn_coeffs.lhat_coeffs.alhatv8198 + pn_coeffs.lhat_coeffs.alhatv8428 + pn_coeffs.lhat_coeffs.alhatv848 + pn_coeffs.lhat_coeffs.alhatv8488) + spin_vars.S2y*(pn_coeffs.lhat_coeffs.alhatv8328 + pn_coeffs.lhat_coeffs.alhatv8458 + pn_coeffs.lhat_coeffs.alhatv888) + spin_vars.lNy*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv8298 + pn_coeffs.lhat_coeffs.alhatv838 + pn_coeffs.lhat_coeffs.alhatv8408)) + spin_vars.lNS2*(spin_vars.S1y*(pn_coeffs.lhat_coeffs.alhatv8308 + pn_coeffs.lhat_coeffs.alhatv8438 + pn_coeffs.lhat_coeffs.alhatv858) + spin_vars.S2y*(pn_coeffs.lhat_coeffs.alhatv8138 + pn_coeffs.lhat_coeffs.alhatv8168 + pn_coeffs.lhat_coeffs.alhatv8218 + pn_coeffs.lhat_coeffs.alhatv8228 + pn_coeffs.lhat_coeffs.alhatv8338 + pn_coeffs.lhat_coeffs.alhatv8508 + pn_coeffs.lhat_coeffs.alhatv8558))
    lhatv7 = spin_vars.S1y*(pn_coeffs.lhat_coeffs.alhatv7157 + pn_coeffs.lhat_coeffs.alhatv7197 + pn_coeffs.lhat_coeffs.alhatv7237 + pn_coeffs.lhat_coeffs.alhatv7277 + pn_coeffs.lhat_coeffs.alhatv737 + pn_coeffs.lhat_coeffs.alhatv777) + spin_vars.S2y*(pn_coeffs.lhat_coeffs.alhatv7107 + pn_coeffs.lhat_coeffs.alhatv7127 + pn_coeffs.lhat_coeffs.alhatv7167 + pn_coeffs.lhat_coeffs.alhatv7207 + pn_coeffs.lhat_coeffs.alhatv7247 + pn_coeffs.lhat_coeffs.alhatv7287 + pn_coeffs.lhat_coeffs.alhatv747 + pn_coeffs.lhat_coeffs.alhatv787) + spin_vars.lNy*spin_vars.lNS1*(pn_coeffs.lhat_coeffs.alhatv7137 + pn_coeffs.lhat_coeffs.alhatv717 + pn_coeffs.lhat_coeffs.alhatv7177 + pn_coeffs.lhat_coeffs.alhatv7217 + pn_coeffs.lhat_coeffs.alhatv7257 + pn_coeffs.lhat_coeffs.alhatv757) + spin_vars.lNy*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv7117 + pn_coeffs.lhat_coeffs.alhatv7147 + pn_coeffs.lhat_coeffs.alhatv7187 + pn_coeffs.lhat_coeffs.alhatv7227 + pn_coeffs.lhat_coeffs.alhatv7267 + pn_coeffs.lhat_coeffs.alhatv727 + pn_coeffs.lhat_coeffs.alhatv767 + pn_coeffs.lhat_coeffs.alhatv797)
    lhatv6 = spin_vars.lNy*(spin_vars.SS12*pn_coeffs.lhat_coeffs.alhatv676 + spin_vars.Ssq1*(pn_coeffs.lhat_coeffs.alhatv6136 + pn_coeffs.lhat_coeffs.alhatv6166 + pn_coeffs.lhat_coeffs.alhatv6206) + spin_vars.Ssq2*(pn_coeffs.lhat_coeffs.alhatv6156 + pn_coeffs.lhat_coeffs.alhatv6236 + pn_coeffs.lhat_coeffs.alhatv6266 + pn_coeffs.lhat_coeffs.alhatv6326 + pn_coeffs.lhat_coeffs.alhatv696) + pn_coeffs.lhat_coeffs.alhatv616 + pn_coeffs.lhat_coeffs.alhatv6336 + pn_coeffs.lhat_coeffs.alhatv6346 + pn_coeffs.lhat_coeffs.alhatv6356 + pn_coeffs.lhat_coeffs.alhatv6366 + spin_vars.lNSsq1*(pn_coeffs.lhat_coeffs.alhatv6106 + pn_coeffs.lhat_coeffs.alhatv6186 + pn_coeffs.lhat_coeffs.alhatv626) + spin_vars.lNSsq2*(pn_coeffs.lhat_coeffs.alhatv6116 + pn_coeffs.lhat_coeffs.alhatv6216 + pn_coeffs.lhat_coeffs.alhatv6246 + pn_coeffs.lhat_coeffs.alhatv6286 + pn_coeffs.lhat_coeffs.alhatv646)) + spin_vars.lNS1*(spin_vars.S1y*(pn_coeffs.lhat_coeffs.alhatv6126 + pn_coeffs.lhat_coeffs.alhatv6196 + pn_coeffs.lhat_coeffs.alhatv656) + spin_vars.S2y*(pn_coeffs.lhat_coeffs.alhatv6306 + pn_coeffs.lhat_coeffs.alhatv686) + spin_vars.lNy*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv6276 + pn_coeffs.lhat_coeffs.alhatv636)) + spin_vars.lNS2*(spin_vars.S1y*(pn_coeffs.lhat_coeffs.alhatv6296 + pn_coeffs.lhat_coeffs.alhatv666) + spin_vars.S2y*(pn_coeffs.lhat_coeffs.alhatv6146 + pn_coeffs.lhat_coeffs.alhatv6176 + pn_coeffs.lhat_coeffs.alhatv6226 + pn_coeffs.lhat_coeffs.alhatv6256 + pn_coeffs.lhat_coeffs.alhatv6316))
    lhatv5 = spin_vars.S1y*(pn_coeffs.lhat_coeffs.alhatv5155 + pn_coeffs.lhat_coeffs.alhatv5195 + pn_coeffs.lhat_coeffs.alhatv535 + pn_coeffs.lhat_coeffs.alhatv575) + spin_vars.S2y*(pn_coeffs.lhat_coeffs.alhatv5105 + pn_coeffs.lhat_coeffs.alhatv5125 + pn_coeffs.lhat_coeffs.alhatv5165 + pn_coeffs.lhat_coeffs.alhatv5205 + pn_coeffs.lhat_coeffs.alhatv545 + pn_coeffs.lhat_coeffs.alhatv585) + spin_vars.lNy*spin_vars.lNS1*(pn_coeffs.lhat_coeffs.alhatv5135 + pn_coeffs.lhat_coeffs.alhatv515 + pn_coeffs.lhat_coeffs.alhatv5175 + pn_coeffs.lhat_coeffs.alhatv555) + spin_vars.lNy*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv5115 + pn_coeffs.lhat_coeffs.alhatv5145 + pn_coeffs.lhat_coeffs.alhatv5185 + pn_coeffs.lhat_coeffs.alhatv525 + pn_coeffs.lhat_coeffs.alhatv565 + pn_coeffs.lhat_coeffs.alhatv595)
    lhatv4 = spin_vars.lNy*(spin_vars.SS12*pn_coeffs.lhat_coeffs.alhatv4144 + spin_vars.Ssq1*pn_coeffs.lhat_coeffs.alhatv444 + spin_vars.Ssq2*(pn_coeffs.lhat_coeffs.alhatv4104 + pn_coeffs.lhat_coeffs.alhatv4174 + pn_coeffs.lhat_coeffs.alhatv474) + pn_coeffs.lhat_coeffs.alhatv414 + pn_coeffs.lhat_coeffs.alhatv4184 + pn_coeffs.lhat_coeffs.alhatv4194 + pn_coeffs.lhat_coeffs.alhatv424*spin_vars.lNSsq1 + spin_vars.lNSsq2*(pn_coeffs.lhat_coeffs.alhatv4124 + pn_coeffs.lhat_coeffs.alhatv454 + pn_coeffs.lhat_coeffs.alhatv484)) + spin_vars.lNS1*(spin_vars.S1y*pn_coeffs.lhat_coeffs.alhatv434 + spin_vars.S2y*pn_coeffs.lhat_coeffs.alhatv4154 + pn_coeffs.lhat_coeffs.alhatv4114*spin_vars.lNy*spin_vars.lNS2) + spin_vars.lNS2*(spin_vars.S1y*pn_coeffs.lhat_coeffs.alhatv4134 + spin_vars.S2y*(pn_coeffs.lhat_coeffs.alhatv4164 + pn_coeffs.lhat_coeffs.alhatv464 + pn_coeffs.lhat_coeffs.alhatv494))
    lhatv3 = spin_vars.S1y*(pn_coeffs.lhat_coeffs.alhatv333 + pn_coeffs.lhat_coeffs.alhatv373) + spin_vars.S2y*(pn_coeffs.lhat_coeffs.alhatv343 + pn_coeffs.lhat_coeffs.alhatv383 + pn_coeffs.lhat_coeffs.alhatv393) + spin_vars.lNy*spin_vars.lNS1*(pn_coeffs.lhat_coeffs.alhatv313 + pn_coeffs.lhat_coeffs.alhatv353) + spin_vars.lNy*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv3103 + pn_coeffs.lhat_coeffs.alhatv3113 + pn_coeffs.lhat_coeffs.alhatv323 + pn_coeffs.lhat_coeffs.alhatv363)
    lhatv2 = spin_vars.lNy*(pn_coeffs.lhat_coeffs.alhatv212 + pn_coeffs.lhat_coeffs.alhatv222)
    lhatv0 = pn_coeffs.lhat_coeffs.alhatv01*spin_vars.lNy
    lhaty = fact0*(lhatv0 + lhatv2*v_powers.v2 + lhatv3*v_powers.v3 + lhatv4*v_powers.v4 + lhatv5*v_powers.v5 + lhatv6*v_powers.v6 + lhatv7*v_powers.v7 + lhatv8*v_powers.v8)

    # z-component
    lhatv8 = spin_vars.lNz*(spin_vars.SS12*(pn_coeffs.lhat_coeffs.alhatv8318 + pn_coeffs.lhat_coeffs.alhatv8448 + pn_coeffs.lhat_coeffs.alhatv868) + spin_vars.Ssq1*(pn_coeffs.lhat_coeffs.alhatv8128 + pn_coeffs.lhat_coeffs.alhatv8208 + pn_coeffs.lhat_coeffs.alhatv8498 + pn_coeffs.lhat_coeffs.alhatv8548 + pn_coeffs.lhat_coeffs.alhatv878) + spin_vars.Ssq2*(pn_coeffs.lhat_coeffs.alhatv8148 + pn_coeffs.lhat_coeffs.alhatv8178 + pn_coeffs.lhat_coeffs.alhatv8248 + pn_coeffs.lhat_coeffs.alhatv8268 + pn_coeffs.lhat_coeffs.alhatv8288 + pn_coeffs.lhat_coeffs.alhatv8358 + pn_coeffs.lhat_coeffs.alhatv8378 + pn_coeffs.lhat_coeffs.alhatv8518 + pn_coeffs.lhat_coeffs.alhatv8568) + pn_coeffs.lhat_coeffs.alhatv818 + pn_coeffs.lhat_coeffs.alhatv8388 + pn_coeffs.lhat_coeffs.alhatv8398 + pn_coeffs.lhat_coeffs.alhatv8418 + pn_coeffs.lhat_coeffs.alhatv8578 + pn_coeffs.lhat_coeffs.alhatv8588 + pn_coeffs.lhat_coeffs.alhatv8598 + pn_coeffs.lhat_coeffs.alhatv8608 + pn_coeffs.lhat_coeffs.alhatv8618 + pn_coeffs.lhat_coeffs.alhatvLog8*v_powers.logv + spin_vars.lNSsq1*(pn_coeffs.lhat_coeffs.alhatv8188 + pn_coeffs.lhat_coeffs.alhatv828 + pn_coeffs.lhat_coeffs.alhatv8468 + pn_coeffs.lhat_coeffs.alhatv8528 + pn_coeffs.lhat_coeffs.alhatv898) + spin_vars.lNSsq2*(pn_coeffs.lhat_coeffs.alhatv8108 + pn_coeffs.lhat_coeffs.alhatv8158 + pn_coeffs.lhat_coeffs.alhatv8238 + pn_coeffs.lhat_coeffs.alhatv8258 + pn_coeffs.lhat_coeffs.alhatv8278 + pn_coeffs.lhat_coeffs.alhatv8348 + pn_coeffs.lhat_coeffs.alhatv8368 + pn_coeffs.lhat_coeffs.alhatv8478 + pn_coeffs.lhat_coeffs.alhatv8538)) + spin_vars.lNS1*(spin_vars.S1z*(pn_coeffs.lhat_coeffs.alhatv8118 + pn_coeffs.lhat_coeffs.alhatv8198 + pn_coeffs.lhat_coeffs.alhatv8428 + pn_coeffs.lhat_coeffs.alhatv848 + pn_coeffs.lhat_coeffs.alhatv8488) + spin_vars.S2z*(pn_coeffs.lhat_coeffs.alhatv8328 + pn_coeffs.lhat_coeffs.alhatv8458 + pn_coeffs.lhat_coeffs.alhatv888) + spin_vars.lNz*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv8298 + pn_coeffs.lhat_coeffs.alhatv838 + pn_coeffs.lhat_coeffs.alhatv8408)) + spin_vars.lNS2*(spin_vars.S1z*(pn_coeffs.lhat_coeffs.alhatv8308 + pn_coeffs.lhat_coeffs.alhatv8438 + pn_coeffs.lhat_coeffs.alhatv858) + spin_vars.S2z*(pn_coeffs.lhat_coeffs.alhatv8138 + pn_coeffs.lhat_coeffs.alhatv8168 + pn_coeffs.lhat_coeffs.alhatv8218 + pn_coeffs.lhat_coeffs.alhatv8228 + pn_coeffs.lhat_coeffs.alhatv8338 + pn_coeffs.lhat_coeffs.alhatv8508 + pn_coeffs.lhat_coeffs.alhatv8558))
    lhatv7 = spin_vars.S1z*(pn_coeffs.lhat_coeffs.alhatv7157 + pn_coeffs.lhat_coeffs.alhatv7197 + pn_coeffs.lhat_coeffs.alhatv7237 + pn_coeffs.lhat_coeffs.alhatv7277 + pn_coeffs.lhat_coeffs.alhatv737 + pn_coeffs.lhat_coeffs.alhatv777) + spin_vars.S2z*(pn_coeffs.lhat_coeffs.alhatv7107 + pn_coeffs.lhat_coeffs.alhatv7127 + pn_coeffs.lhat_coeffs.alhatv7167 + pn_coeffs.lhat_coeffs.alhatv7207 + pn_coeffs.lhat_coeffs.alhatv7247 + pn_coeffs.lhat_coeffs.alhatv7287 + pn_coeffs.lhat_coeffs.alhatv747 + pn_coeffs.lhat_coeffs.alhatv787) + spin_vars.lNz*spin_vars.lNS1*(pn_coeffs.lhat_coeffs.alhatv7137 + pn_coeffs.lhat_coeffs.alhatv717 + pn_coeffs.lhat_coeffs.alhatv7177 + pn_coeffs.lhat_coeffs.alhatv7217 + pn_coeffs.lhat_coeffs.alhatv7257 + pn_coeffs.lhat_coeffs.alhatv757) + spin_vars.lNz*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv7117 + pn_coeffs.lhat_coeffs.alhatv7147 + pn_coeffs.lhat_coeffs.alhatv7187 + pn_coeffs.lhat_coeffs.alhatv7227 + pn_coeffs.lhat_coeffs.alhatv7267 + pn_coeffs.lhat_coeffs.alhatv727 + pn_coeffs.lhat_coeffs.alhatv767 + pn_coeffs.lhat_coeffs.alhatv797)
    lhatv6 = spin_vars.lNz*(spin_vars.SS12*pn_coeffs.lhat_coeffs.alhatv676 + spin_vars.Ssq1*(pn_coeffs.lhat_coeffs.alhatv6136 + pn_coeffs.lhat_coeffs.alhatv6166 + pn_coeffs.lhat_coeffs.alhatv6206) + spin_vars.Ssq2*(pn_coeffs.lhat_coeffs.alhatv6156 + pn_coeffs.lhat_coeffs.alhatv6236 + pn_coeffs.lhat_coeffs.alhatv6266 + pn_coeffs.lhat_coeffs.alhatv6326 + pn_coeffs.lhat_coeffs.alhatv696) + pn_coeffs.lhat_coeffs.alhatv616 + pn_coeffs.lhat_coeffs.alhatv6336 + pn_coeffs.lhat_coeffs.alhatv6346 + pn_coeffs.lhat_coeffs.alhatv6356 + pn_coeffs.lhat_coeffs.alhatv6366 + spin_vars.lNSsq1*(pn_coeffs.lhat_coeffs.alhatv6106 + pn_coeffs.lhat_coeffs.alhatv6186 + pn_coeffs.lhat_coeffs.alhatv626) + spin_vars.lNSsq2*(pn_coeffs.lhat_coeffs.alhatv6116 + pn_coeffs.lhat_coeffs.alhatv6216 + pn_coeffs.lhat_coeffs.alhatv6246 + pn_coeffs.lhat_coeffs.alhatv6286 + pn_coeffs.lhat_coeffs.alhatv646)) + spin_vars.lNS1*(spin_vars.S1z*(pn_coeffs.lhat_coeffs.alhatv6126 + pn_coeffs.lhat_coeffs.alhatv6196 + pn_coeffs.lhat_coeffs.alhatv656) + spin_vars.S2z*(pn_coeffs.lhat_coeffs.alhatv6306 + pn_coeffs.lhat_coeffs.alhatv686) + spin_vars.lNz*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv6276 + pn_coeffs.lhat_coeffs.alhatv636)) + spin_vars.lNS2*(spin_vars.S1z*(pn_coeffs.lhat_coeffs.alhatv6296 + pn_coeffs.lhat_coeffs.alhatv666) + spin_vars.S2z*(pn_coeffs.lhat_coeffs.alhatv6146 + pn_coeffs.lhat_coeffs.alhatv6176 + pn_coeffs.lhat_coeffs.alhatv6226 + pn_coeffs.lhat_coeffs.alhatv6256 + pn_coeffs.lhat_coeffs.alhatv6316))
    lhatv5 = spin_vars.S1z*(pn_coeffs.lhat_coeffs.alhatv5155 + pn_coeffs.lhat_coeffs.alhatv5195 + pn_coeffs.lhat_coeffs.alhatv535 + pn_coeffs.lhat_coeffs.alhatv575) + spin_vars.S2z*(pn_coeffs.lhat_coeffs.alhatv5105 + pn_coeffs.lhat_coeffs.alhatv5125 + pn_coeffs.lhat_coeffs.alhatv5165 + pn_coeffs.lhat_coeffs.alhatv5205 + pn_coeffs.lhat_coeffs.alhatv545 + pn_coeffs.lhat_coeffs.alhatv585) + spin_vars.lNz*spin_vars.lNS1*(pn_coeffs.lhat_coeffs.alhatv5135 + pn_coeffs.lhat_coeffs.alhatv515 + pn_coeffs.lhat_coeffs.alhatv5175 + pn_coeffs.lhat_coeffs.alhatv555) + spin_vars.lNz*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv5115 + pn_coeffs.lhat_coeffs.alhatv5145 + pn_coeffs.lhat_coeffs.alhatv5185 + pn_coeffs.lhat_coeffs.alhatv525 + pn_coeffs.lhat_coeffs.alhatv565 + pn_coeffs.lhat_coeffs.alhatv595)
    lhatv4 = spin_vars.lNz*(spin_vars.SS12*pn_coeffs.lhat_coeffs.alhatv4144 + spin_vars.Ssq1*pn_coeffs.lhat_coeffs.alhatv444 + spin_vars.Ssq2*(pn_coeffs.lhat_coeffs.alhatv4104 + pn_coeffs.lhat_coeffs.alhatv4174 + pn_coeffs.lhat_coeffs.alhatv474) + pn_coeffs.lhat_coeffs.alhatv414 + pn_coeffs.lhat_coeffs.alhatv4184 + pn_coeffs.lhat_coeffs.alhatv4194 + pn_coeffs.lhat_coeffs.alhatv424*spin_vars.lNSsq1 + spin_vars.lNSsq2*(pn_coeffs.lhat_coeffs.alhatv4124 + pn_coeffs.lhat_coeffs.alhatv454 + pn_coeffs.lhat_coeffs.alhatv484)) + spin_vars.lNS1*(spin_vars.S1z*pn_coeffs.lhat_coeffs.alhatv434 + spin_vars.S2z*pn_coeffs.lhat_coeffs.alhatv4154 + pn_coeffs.lhat_coeffs.alhatv4114*spin_vars.lNz*spin_vars.lNS2) + spin_vars.lNS2*(spin_vars.S1z*pn_coeffs.lhat_coeffs.alhatv4134 + spin_vars.S2z*(pn_coeffs.lhat_coeffs.alhatv4164 + pn_coeffs.lhat_coeffs.alhatv464 + pn_coeffs.lhat_coeffs.alhatv494))
    lhatv3 = spin_vars.S1z*(pn_coeffs.lhat_coeffs.alhatv333 + pn_coeffs.lhat_coeffs.alhatv373) + spin_vars.S2z*(pn_coeffs.lhat_coeffs.alhatv343 + pn_coeffs.lhat_coeffs.alhatv383 + pn_coeffs.lhat_coeffs.alhatv393) + spin_vars.lNz*spin_vars.lNS1*(pn_coeffs.lhat_coeffs.alhatv313 + pn_coeffs.lhat_coeffs.alhatv353) + spin_vars.lNz*spin_vars.lNS2*(pn_coeffs.lhat_coeffs.alhatv3103 + pn_coeffs.lhat_coeffs.alhatv3113 + pn_coeffs.lhat_coeffs.alhatv323 + pn_coeffs.lhat_coeffs.alhatv363)
    lhatv2 = spin_vars.lNz*(pn_coeffs.lhat_coeffs.alhatv212 + pn_coeffs.lhat_coeffs.alhatv222)
    lhatv0 = pn_coeffs.lhat_coeffs.alhatv01*spin_vars.lNz

    lhatz = fact0*(lhatv0 + lhatv2*v_powers.v2 + lhatv3*v_powers.v3 + lhatv4*v_powers.v4 + lhatv5*v_powers.v5 + lhatv6*v_powers.v6 + lhatv7*v_powers.v7 + lhatv8*v_powers.v8)

    return [lhatx,lhaty,lhatz]


############ Wrapper for the RHS evaluation



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef list prec_eqns_20102022_cython_opt(double t, double[:] z, double nu, PNCoeffs pn_coeffs):
  """Post-Newtonian precession equations, as
  well as the evolution of the orbital frequency

  Args:
      t (float): Time
      z (np.array): Vector of unknowns

  Raises:
      NotImplementedError: If Taylor approximant is unknown

  Returns:
      np.array: RHS of equations
  """

  cdef double Lhx, Lhy, Lhz
  cdef double lNx, lNy, lNz
  cdef double s1x, s1y, s1z
  cdef double s2x, s2y, s2z
  cdef double omega,domdt
  cdef double s1dotx, s1doty, s1dotz
  cdef double s2dotx, s2doty, s2dotz
  cdef double lNdotx, lNdoty, lNdotz



  Lhx = z[0]
  Lhy = z[1]
  Lhz = z[2]

  s1x = z[3]
  s1y = z[4]
  s1z = z[5]

  s2x = z[6]
  s2y = z[7]
  s2z = z[8]
  omega = z[9]

  Lh_norm = my_norm_cy(Lhx,Lhy,Lhz,Lhx,Lhy,Lhz)

  lNx = Lhx/Lh_norm
  lNy = Lhy/Lh_norm
  lNz = Lhz/Lh_norm


  cdef vpowers v_powers = vpowers(omega)

  cdef spinVars spin_vars = spinVars(lNx,lNy,lNz,
                                s1x, s1y, s1z
                                ,s2x, s2y, s2z)

  cdef double s1_dot[3]
  s1_dot[:] = compute_s1dot_opt(v_powers, spin_vars, pn_coeffs)
  s1dotx = s1_dot[0]
  s1doty = s1_dot[1]
  s1dotz = s1_dot[2]
  cdef double s2_dot[3]
  s2_dot[:] = compute_s2dot_opt(v_powers, spin_vars, pn_coeffs)
  s2dotx = s2_dot[0]
  s2doty = s2_dot[1]
  s2dotz = s2_dot[2]
  cdef double lN_dot[3]
  lN_dot[:] = compute_lNdot_opt(v_powers, spin_vars, pn_coeffs)
  lNdotx = lN_dot[0]
  lNdoty = lN_dot[1]
  lNdotz = lN_dot[2]

  #compute_lhat_opt(nu, v_powers, spin_vars, pn_coeffs,lhat_cy)
  domdt = compute_omegadot(nu, v_powers, spin_vars, pn_coeffs)

  return [s1dotx, s1doty, s1dotz, s2dotx, s2doty, s2dotz, lNdotx, lNdoty, lNdotz, domdt]




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef list prec_eqns_20102022_cython_opt1(double t, double[:] z, double nu, PNCoeffs pn_coeffs):
  """Post-Newtonian precession equations, as
  well as the evolution of the orbital frequency

  Args:
      t (float): Time
      z (np.array): Vector of unknowns

  Raises:
      NotImplementedError: If Taylor approximant is unknown

  Returns:
      np.array: RHS of equations
  """

  cdef double Lhx, Lhy, Lhz
  cdef double lNx, lNy, lNz
  cdef double s1x, s1y, s1z
  cdef double s2x, s2y, s2z
  cdef double omega,domdt


  Lhx,Lhy,Lhz = z[:3]
  s1x, s1y, s1z = z[3:6]
  s2x, s2y, s2z = z[6:9]
  omega = z[9]

  Lh_norm = my_norm_cy(Lhx,Lhy,Lhz,Lhx,Lhy,Lhz)

  lNx = Lhx/Lh_norm
  lNy = Lhy/Lh_norm
  lNz = Lhz/Lh_norm


  cdef vpowers v_powers = vpowers(omega)

  cdef spinVars spin_vars = spinVars(lNx,lNy,lNz,
                                s1x, s1y, s1z
                                ,s2x, s2y, s2z)

  ############## s1dot ################


  cdef double sdotv110
  cdef double sdotv19
  cdef double sdotv18
  cdef double sdotv17
  cdef double sdotv16
  cdef double sdotv15
  cdef double s1dotx, s1doty, s1dotz

  # coefficients entering s1dot

  # x-component
  sdotv110 = spin_vars.SxS12x*pn_coeffs.s1dot_coeffs.asdotv11061 + spin_vars.lNxS1x*(pn_coeffs.s1dot_coeffs.asdotv110621*spin_vars.lNS1 + pn_coeffs.s1dot_coeffs.asdotv110622*spin_vars.lNS2)
  sdotv19 = pn_coeffs.s1dot_coeffs.asdotv1950*spin_vars.lNxS1x
  sdotv18 = spin_vars.SxS12x*pn_coeffs.s1dot_coeffs.asdotv1841 + spin_vars.lNxS1x*(pn_coeffs.s1dot_coeffs.asdotv18421*spin_vars.lNS2 + pn_coeffs.s1dot_coeffs.asdotv18422*spin_vars.lNS1)
  sdotv17 = pn_coeffs.s1dot_coeffs.asdotv1730*spin_vars.lNxS1x
  sdotv16 = spin_vars.SxS12x*pn_coeffs.s1dot_coeffs.asdotv1621 + spin_vars.lNxS1x*(pn_coeffs.s1dot_coeffs.asdotv16221*spin_vars.lNS2 + pn_coeffs.s1dot_coeffs.asdotv16222*spin_vars.lNS1)
  sdotv15 = pn_coeffs.s1dot_coeffs.asdotv1510*spin_vars.lNxS1x
  s1dotx = sdotv110*v_powers.v10 + sdotv15*v_powers.v5 + sdotv16*v_powers.v6 + sdotv17*v_powers.v7 + sdotv18*v_powers.v8 + sdotv19*v_powers.v9

  # y-component
  sdotv110 = spin_vars.SxS12y*pn_coeffs.s1dot_coeffs.asdotv11061 + spin_vars.lNxS1y*(pn_coeffs.s1dot_coeffs.asdotv110621*spin_vars.lNS1 + pn_coeffs.s1dot_coeffs.asdotv110622*spin_vars.lNS2)
  sdotv19 = pn_coeffs.s1dot_coeffs.asdotv1950*spin_vars.lNxS1y
  sdotv18 = spin_vars.SxS12y*pn_coeffs.s1dot_coeffs.asdotv1841 + spin_vars.lNxS1y*(pn_coeffs.s1dot_coeffs.asdotv18421*spin_vars.lNS2 + pn_coeffs.s1dot_coeffs.asdotv18422*spin_vars.lNS1)
  sdotv17 = pn_coeffs.s1dot_coeffs.asdotv1730*spin_vars.lNxS1y
  sdotv16 = spin_vars.SxS12y*pn_coeffs.s1dot_coeffs.asdotv1621 + spin_vars.lNxS1y*(pn_coeffs.s1dot_coeffs.asdotv16221*spin_vars.lNS2 + pn_coeffs.s1dot_coeffs.asdotv16222*spin_vars.lNS1)
  sdotv15 = pn_coeffs.s1dot_coeffs.asdotv1510*spin_vars.lNxS1y
  s1doty = sdotv110*v_powers.v10 + sdotv15*v_powers.v5 + sdotv16*v_powers.v6 + sdotv17*v_powers.v7 + sdotv18*v_powers.v8 + sdotv19*v_powers.v9

  # z-component
  sdotv110 = spin_vars.SxS12z*pn_coeffs.s1dot_coeffs.asdotv11061 + spin_vars.lNxS1z*(pn_coeffs.s1dot_coeffs.asdotv110621*spin_vars.lNS1 + pn_coeffs.s1dot_coeffs.asdotv110622*spin_vars.lNS2)
  sdotv19 = pn_coeffs.s1dot_coeffs.asdotv1950*spin_vars.lNxS1z
  sdotv18 = spin_vars.SxS12z*pn_coeffs.s1dot_coeffs.asdotv1841 + spin_vars.lNxS1z*(pn_coeffs.s1dot_coeffs.asdotv18421*spin_vars.lNS2 + pn_coeffs.s1dot_coeffs.asdotv18422*spin_vars.lNS1)
  sdotv17 = pn_coeffs.s1dot_coeffs.asdotv1730*spin_vars.lNxS1z
  sdotv16 = spin_vars.SxS12z*pn_coeffs.s1dot_coeffs.asdotv1621 + spin_vars.lNxS1z*(pn_coeffs.s1dot_coeffs.asdotv16221*spin_vars.lNS2 + pn_coeffs.s1dot_coeffs.asdotv16222*spin_vars.lNS1)
  sdotv15 = pn_coeffs.s1dot_coeffs.asdotv1510*spin_vars.lNxS1z
  s1dotz = sdotv110*v_powers.v10 + sdotv15*v_powers.v5 + sdotv16*v_powers.v6 + sdotv17*v_powers.v7 + sdotv18*v_powers.v8 + sdotv19*v_powers.v9

  ############## s2dot ################

  # Initialize variables
  cdef double sdotv210
  cdef double sdotv29
  cdef double sdotv28
  cdef double sdotv27
  cdef double sdotv26
  cdef double sdotv25
  cdef double s2dotx, s2doty, s2dotz

  # coefficients entering s2dot

  # x-component
  sdotv210 = spin_vars.SxS12x*pn_coeffs.s2dot_coeffs.asdotv21061 + spin_vars.lNxS2x*(pn_coeffs.s2dot_coeffs.asdotv210621*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv210622*spin_vars.lNS2)
  sdotv29 = pn_coeffs.s2dot_coeffs.asdotv2950*spin_vars.lNxS2x
  sdotv28 = spin_vars.SxS12x*pn_coeffs.s2dot_coeffs.asdotv2841 + spin_vars.lNxS2x*(pn_coeffs.s2dot_coeffs.asdotv28421*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv28422*spin_vars.lNS2)
  sdotv27 = pn_coeffs.s2dot_coeffs.asdotv2730*spin_vars.lNxS2x
  sdotv26 = spin_vars.SxS12x*pn_coeffs.s2dot_coeffs.asdotv2621 + spin_vars.lNxS2x*(pn_coeffs.s2dot_coeffs.asdotv26221*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv26222*spin_vars.lNS2)
  sdotv25 = pn_coeffs.s2dot_coeffs.asdotv2510*spin_vars.lNxS2x
  s2dotx = sdotv210*v_powers.v10 + sdotv25*v_powers.v5 + sdotv26*v_powers.v6 + sdotv27*v_powers.v7 + sdotv28*v_powers.v8 + sdotv29*v_powers.v9

  # y-component
  sdotv210 = spin_vars.SxS12y*pn_coeffs.s2dot_coeffs.asdotv21061 + spin_vars.lNxS2y*(pn_coeffs.s2dot_coeffs.asdotv210621*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv210622*spin_vars.lNS2)
  sdotv29 = pn_coeffs.s2dot_coeffs.asdotv2950*spin_vars.lNxS2y
  sdotv28 = spin_vars.SxS12y*pn_coeffs.s2dot_coeffs.asdotv2841 + spin_vars.lNxS2y*(pn_coeffs.s2dot_coeffs.asdotv28421*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv28422*spin_vars.lNS2)
  sdotv27 = pn_coeffs.s2dot_coeffs.asdotv2730*spin_vars.lNxS2y
  sdotv26 = spin_vars.SxS12y*pn_coeffs.s2dot_coeffs.asdotv2621 + spin_vars.lNxS2y*(pn_coeffs.s2dot_coeffs.asdotv26221*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv26222*spin_vars.lNS2)
  sdotv25 = pn_coeffs.s2dot_coeffs.asdotv2510*spin_vars.lNxS2y
  s2doty = sdotv210*v_powers.v10 + sdotv25*v_powers.v5 + sdotv26*v_powers.v6 + sdotv27*v_powers.v7 + sdotv28*v_powers.v8 + sdotv29*v_powers.v9

  # z-component
  sdotv210 = spin_vars.SxS12z*pn_coeffs.s2dot_coeffs.asdotv21061 + spin_vars.lNxS2z*(pn_coeffs.s2dot_coeffs.asdotv210621*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv210622*spin_vars.lNS2)
  sdotv29 = pn_coeffs.s2dot_coeffs.asdotv2950*spin_vars.lNxS2z
  sdotv28 = spin_vars.SxS12z*pn_coeffs.s2dot_coeffs.asdotv2841 + spin_vars.lNxS2z*(pn_coeffs.s2dot_coeffs.asdotv28421*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv28422*spin_vars.lNS2)
  sdotv27 = pn_coeffs.s2dot_coeffs.asdotv2730*spin_vars.lNxS2z
  sdotv26 = spin_vars.SxS12z*pn_coeffs.s2dot_coeffs.asdotv2621 + spin_vars.lNxS2z*(pn_coeffs.s2dot_coeffs.asdotv26221*spin_vars.lNS1 + pn_coeffs.s2dot_coeffs.asdotv26222*spin_vars.lNS2)
  sdotv25 = pn_coeffs.s2dot_coeffs.asdotv2510*spin_vars.lNxS2z
  s2dotz = sdotv210*v_powers.v10 + sdotv25*v_powers.v5 + sdotv26*v_powers.v6 + sdotv27*v_powers.v7 + sdotv28*v_powers.v8 + sdotv29*v_powers.v9


  ############## lNdot ################

  cdef double lNdotv11
  cdef double lNdotv10
  cdef double lNdotv9
  cdef double lNdotv8
  cdef double lNdotv7
  cdef double lNdotv6
  cdef double lNdotx, lNdoty, lNdotz

  # coefficients entering lNdot

  # x-component
  lNdotv11 = spin_vars.SxS12x*pn_coeffs.lNdot_coeffs.alNdotv1162 + pn_coeffs.lNdot_coeffs.alNdotv1161*spin_vars.lNx*spin_vars.lNSxS12 + spin_vars.lNxS1x*(pn_coeffs.lNdot_coeffs.alNdotv11641*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv11642*spin_vars.lNS1) + spin_vars.lNxS2x*(pn_coeffs.lNdot_coeffs.alNdotv11631*spin_vars.lNS1 + pn_coeffs.lNdot_coeffs.alNdotv11632*spin_vars.lNS2)
  lNdotv10 = pn_coeffs.lNdot_coeffs.alNdotv1051*spin_vars.lNxS2x + pn_coeffs.lNdot_coeffs.alNdotv1052*spin_vars.lNxS1x
  lNdotv9 = spin_vars.SxS12x*pn_coeffs.lNdot_coeffs.alNdotv942 + pn_coeffs.lNdot_coeffs.alNdotv941*spin_vars.lNx*spin_vars.lNSxS12 + spin_vars.lNxS1x*(pn_coeffs.lNdot_coeffs.alNdotv9431*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv9432*spin_vars.lNS1) + spin_vars.lNxS2x*(pn_coeffs.lNdot_coeffs.alNdotv9441*spin_vars.lNS1 + pn_coeffs.lNdot_coeffs.alNdotv9442*spin_vars.lNS2)
  lNdotv8 = pn_coeffs.lNdot_coeffs.alNdotv831*spin_vars.lNxS1x + pn_coeffs.lNdot_coeffs.alNdotv832*spin_vars.lNxS2x
  lNdotv7 = spin_vars.lNxS1x*(pn_coeffs.lNdot_coeffs.alNdotv7221*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv7222*spin_vars.lNS1) + spin_vars.lNxS2x*(pn_coeffs.lNdot_coeffs.alNdotv7211*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv7212*spin_vars.lNS1)
  lNdotv6 = pn_coeffs.lNdot_coeffs.alNdotv611*spin_vars.lNxS1x + pn_coeffs.lNdot_coeffs.alNdotv612*spin_vars.lNxS2x
  lNdotx = lNdotv10*v_powers.v10 + lNdotv11*v_powers.v11 + lNdotv6*v_powers.v6 + lNdotv7*v_powers.v7 + lNdotv8*v_powers.v8 + lNdotv9*v_powers.v9

  # y-component
  lNdotv11 = spin_vars.SxS12y*pn_coeffs.lNdot_coeffs.alNdotv1162 + pn_coeffs.lNdot_coeffs.alNdotv1161*spin_vars.lNy*spin_vars.lNSxS12 + spin_vars.lNxS1y*(pn_coeffs.lNdot_coeffs.alNdotv11641*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv11642*spin_vars.lNS1) + spin_vars.lNxS2y*(pn_coeffs.lNdot_coeffs.alNdotv11631*spin_vars.lNS1 + pn_coeffs.lNdot_coeffs.alNdotv11632*spin_vars.lNS2)
  lNdotv10 = pn_coeffs.lNdot_coeffs.alNdotv1051*spin_vars.lNxS2y + pn_coeffs.lNdot_coeffs.alNdotv1052*spin_vars.lNxS1y
  lNdotv9 = spin_vars.SxS12y*pn_coeffs.lNdot_coeffs.alNdotv942 + pn_coeffs.lNdot_coeffs.alNdotv941*spin_vars.lNy*spin_vars.lNSxS12 + spin_vars.lNxS1y*(pn_coeffs.lNdot_coeffs.alNdotv9431*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv9432*spin_vars.lNS1) + spin_vars.lNxS2y*(pn_coeffs.lNdot_coeffs.alNdotv9441*spin_vars.lNS1 + pn_coeffs.lNdot_coeffs.alNdotv9442*spin_vars.lNS2)
  lNdotv8 = pn_coeffs.lNdot_coeffs.alNdotv831*spin_vars.lNxS1y + pn_coeffs.lNdot_coeffs.alNdotv832*spin_vars.lNxS2y
  lNdotv7 = spin_vars.lNxS1y*(pn_coeffs.lNdot_coeffs.alNdotv7221*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv7222*spin_vars.lNS1) + spin_vars.lNxS2y*(pn_coeffs.lNdot_coeffs.alNdotv7211*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv7212*spin_vars.lNS1)
  lNdotv6 = pn_coeffs.lNdot_coeffs.alNdotv611*spin_vars.lNxS1y + pn_coeffs.lNdot_coeffs.alNdotv612*spin_vars.lNxS2y
  lNdoty = lNdotv10*v_powers.v10 + lNdotv11*v_powers.v11 + lNdotv6*v_powers.v6 + lNdotv7*v_powers.v7 + lNdotv8*v_powers.v8 + lNdotv9*v_powers.v9

  # z-component
  lNdotv11 = spin_vars.SxS12z*pn_coeffs.lNdot_coeffs.alNdotv1162 + pn_coeffs.lNdot_coeffs.alNdotv1161*spin_vars.lNz*spin_vars.lNSxS12 + spin_vars.lNxS1z*(pn_coeffs.lNdot_coeffs.alNdotv11641*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv11642*spin_vars.lNS1) + spin_vars.lNxS2z*(pn_coeffs.lNdot_coeffs.alNdotv11631*spin_vars.lNS1 + pn_coeffs.lNdot_coeffs.alNdotv11632*spin_vars.lNS2)
  lNdotv10 = pn_coeffs.lNdot_coeffs.alNdotv1051*spin_vars.lNxS2z + pn_coeffs.lNdot_coeffs.alNdotv1052*spin_vars.lNxS1z
  lNdotv9 = spin_vars.SxS12z*pn_coeffs.lNdot_coeffs.alNdotv942 + pn_coeffs.lNdot_coeffs.alNdotv941*spin_vars.lNz*spin_vars.lNSxS12 + spin_vars.lNxS1z*(pn_coeffs.lNdot_coeffs.alNdotv9431*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv9432*spin_vars.lNS1) + spin_vars.lNxS2z*(pn_coeffs.lNdot_coeffs.alNdotv9441*spin_vars.lNS1 + pn_coeffs.lNdot_coeffs.alNdotv9442*spin_vars.lNS2)
  lNdotv8 = pn_coeffs.lNdot_coeffs.alNdotv831*spin_vars.lNxS1z + pn_coeffs.lNdot_coeffs.alNdotv832*spin_vars.lNxS2z
  lNdotv7 = spin_vars.lNxS1z*(pn_coeffs.lNdot_coeffs.alNdotv7221*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv7222*spin_vars.lNS1) + spin_vars.lNxS2z*(pn_coeffs.lNdot_coeffs.alNdotv7211*spin_vars.lNS2 + pn_coeffs.lNdot_coeffs.alNdotv7212*spin_vars.lNS1)
  lNdotv6 = pn_coeffs.lNdot_coeffs.alNdotv611*spin_vars.lNxS1z + pn_coeffs.lNdot_coeffs.alNdotv612*spin_vars.lNxS2z
  lNdotz = lNdotv10*v_powers.v10 + lNdotv11*v_powers.v11 + lNdotv6*v_powers.v6 + lNdotv7*v_powers.v7 + lNdotv8*v_powers.v8 + lNdotv9*v_powers.v9


  ############## omegadot ################

  domdt = compute_omegadot(nu, v_powers, spin_vars, pn_coeffs)

  return [s1dotx, s1doty, s1dotz, s2dotx, s2doty, s2dotz, lNdotx, lNdoty, lNdotz, domdt]
