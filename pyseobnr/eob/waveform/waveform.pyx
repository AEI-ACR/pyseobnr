# cython: language_level=3
cimport cython
from libc.math cimport log, sqrt, exp, abs,fabs, tgamma,sin,cos, tanh, sinh, asinh
from libc.math cimport M_PI as pi
from libc.math cimport M_E, M_PI_2
from scipy.special.cython_special cimport sph_harm, loggamma
from scipy.special import  factorial2
from pyseobnr.eob.utils.containers cimport EOBParams,FluxParams

cimport scipy.linalg.cython_blas as blas
import numpy as np
cimport numpy as np
import sys

from scipy.interpolate import CubicSpline

cdef extern from "complex.h":
    double cabs(double complex z)
    double creal(double complex z)
    double complex cexp(double complex z)
    double carg(double complex z)
    double complex I


DEF euler_gamma=0.5772156649015329

# Lookup table of fast spin-weighted spherical harmonics
# Used to compute Newtonian prefactors in factorized waveform
# Follows the scipy convention: ylms[m][ell]
cdef double ylms[9][9]
ylms[:] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [
        0.0,
        0.3454941494713355,
        0.0,
        0.3231801841141506,
        4.055554344078009e-17,
        0.32028164857621516,
        8.000317029649286e-17,
        0.31937046138540076,
        1.1920022960993937e-16,
    ],
    [
        0.0,
        0.0,
        0.3862742020231896,
        0.0,
        0.33452327177864466,
        4.447491653287784e-17,
        0.32569524293385776,
        8.531070568407408e-17,
        0.32254835519288305,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.4172238236327842,
        0.0,
        0.34594371914684025,
        4.8749755603568385e-17,
        0.331899519333737,
        9.130149959227967e-17,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.4425326924449826,
        0.0,
        0.3567812628539981,
        5.310595586255289e-17,
        0.3382915688890245,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.46413220344085826,
        0.0,
        0.3669287245764378,
        5.745136532071183e-17,
    ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.48308411358006625, 0.0, 0.3764161087284946],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5000395635705508, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5154289843972844],
]


DTYPE = np.float64
ctypedef np.float64_t DTYPE_T

DTYPE_c = np.complex128
ctypedef np.complex128_t DTYPE_tc

# Lookup table for factorial until n=21
cdef double[21] LOOKUP_TABLE
LOOKUP_TABLE[:] = [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ]



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef double complex calculate_multipole_prefix(double m1, double m2, int l, int m):
    """
    Calculates the Newtonian multipole prefactors, see Eq. 25-27 in https://dcc.ligo.org/LIGO-T2300060 (SEOBNRv5HM.pdf).
    See also Sec. 2A of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.79.064004
    """
    cdef double complex n = 0.0
    cdef double complex prefix

    cdef double totalMass = m1 + m2

    cdef double epsilon = (l + m) % 2

    cdef double x1 = m1 / totalMass
    cdef double x2 = m2 / totalMass

    cdef double eta = m1 * m2 / (totalMass * totalMass)

    cdef int sign
    if abs(m % 2) == 0:
        sign = 1
    else:
        sign = -1

    #
    # Eq. 7 of Damour, Iyer and Nagar 2008.
    # For odd m, c is proportional to dM = m1-m2. In the equal-mass case, c = dM = 0.
    # In the equal-mass unequal-spin case, however, when spins are different, the odd m term is generally not zero.
    # In this case, c can be written as c0 * dM, while spins terms in PN expansion may take the form chiA/dM.
    # Although the dM's cancel analytically, we can not implement c and chiA/dM with the possibility of dM -> 0.
    # Therefore, for this case, we give numerical values of c0 for relevant modes, and c0 is calculated as
    # c / dM in the limit of dM -> 0. Consistently, for this case, we implement chiA instead of chiA/dM
    # below.
    # Note that for equal masses and odd m modes we currently only have non-zero c up to l=5. If new spinning terms
    # are added to modes above l=5 this will need to be revisited.
    cdef double c,mult1,mult2
    if m1 != m2 or sign == 1:
        c = pow(x2, l + epsilon - 1) + sign * pow(x1, l + epsilon - 1)

    else:
        if l == 2:
            c = -1.0
        elif l == 3:
            c = -1.0
        elif l == 4:
            c = -0.5
        elif l == 5:
            c = -0.5
        else:
            c = 0.0

    # Eqs 5 and 6. Dependent on the value of epsilon (parity), we get different n
    if epsilon == 0:
        n = 1j * m
        n = n ** l

        mult1 = 8.0 * pi / factorial2(2 * l + 1)
        mult2 = ((l + 1) * (l + 2)) / (l * (l - 1))
        mult2 = sqrt(mult2)

        n *= mult1
        n *= mult2

    elif epsilon == 1:

        n = 1j * m
        n = n ** l
        n = -n

        mult1 = 16.0 * pi / factorial2(2 * l + 1)

        mult2 = (2 * l + 1) * (l + 2) * (l * l - m * m)
        mult2 /= (2 * l - 1) * (l + 1) * l * (l - 1)
        mult2 = sqrt(mult2)

        n *= 1j * mult1
        n *= mult2

    prefix = n * eta * c
    return prefix


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef compute_newtonian_prefixes(double m1, double m2):
    """
    Loop to set the Newtonian multipole prefactors, see Eq. 25-27 in https://dcc.ligo.org/LIGO-T2300060 (SEOBNRv5HM.pdf).
    """
    cdef double complex prefixes[9][9]
    for i in range(9):
        for j in range(9):
            prefixes[i][j]=0.0

    for l in range(2, ell_max + 1):
        for m in range(1, l + 1):
            prefixes[l][m] = calculate_multipole_prefix(m1, m2, l, m)
    return prefixes

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double evaluate_nqc_correction_flux(double r, double pr,  double omega, double[:] coeffs):
    """
    Calculate the NQC amplitude correction, see Eq. 35 in https://dcc.ligo.org/LIGO-T2300060 (SEOBNRv5HM.pdf).
    """
    cdef double sqrtR = sqrt(r)
    cdef double rOmega = r * omega
    cdef double rOmegaSq = rOmega * rOmega
    cdef double p = pr
    cdef double mag = 1.0 + (p * p / rOmegaSq) * (
        coeffs[0]
        + coeffs[1] / r
        + coeffs[2]  / (r * sqrtR)
    )
    return mag

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef compute_tail(double omega, double H, double[:,:] Tlm):
    """
    Calculate the resummed Tail effects, see Eq. 32 in https://dcc.ligo.org/LIGO-T2300060 (SEOBNRv5HM.pdf).
    See also Sec. 2B of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.79.064004
    and Eq. (42) of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.87.084035
    """
    cdef int m,j
    cdef double k, hathatk,hathatksq4,hathatk4pi,z2
    cdef double tlmprod_fac = 1.0
    cdef double tlmprefac

    for m in range(1,ell_max+1):
        k = m*omega
        hathatk = H*k
        hathatksq4 = 4.0 * hathatk * hathatk
        hathatk4pi = 4.0 * pi * hathatk
        tlmprod_fac = 1.0
        tlmprefac = sqrt(hathatk4pi / (1.0 - exp(-hathatk4pi)))

        for j in range(1,ell_max+1):
            z2 = LOOKUP_TABLE[j]
            tlmprod_fac*= hathatksq4 + j**2
            if m>j:
                continue
            Tlm[j][m] = tlmprefac*sqrt(tlmprod_fac)/z2



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef void compute_rho_coeffs(double nu,double dm, double a,double chiS,double chiA,
    double[:,:,:] rho_coeffs,double[:,:,:] rho_coeffs_log, double[:,:,:] f_coeffs,
    double complex[:,:,:] f_coeffs_vh, bint extra_PN_terms):

    """
    Compute the amplitude residual coefficients.
    See Sec. 2C and 2D of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.79.064004
    See Eq. 59-63 of https://dcc.ligo.org/LIGO-T2300060 (SEOBNRv5_theory.pdf) for new terms, rest copied from SEOBNRv4HM LAL code.

    Coefficients can be found in:
    - https://dcc.ligo.org/LIGO-T2300060 (SEOBNRv5_theory.pdf and SEOBNRv5HM.pdf)
    - Appendix A of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.98.084028
    - Eqs. (2.4) to (2.6) of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.95.044028
    - https://journals.aps.org/prd/pdf/10.1103/PhysRevD.83.064003
    - https://journals.aps.org/prd/pdf/10.1103/PhysRevD.86.024011
    """

    cdef double nu2 = nu*nu
    cdef double nu3 = nu*nu2

    cdef double a2 = a*a
    cdef double a3 = a*a2
    cdef double dm2 = dm*dm
    cdef double atemp = a

    cdef double chiA2 = chiA * chiA
    cdef double chiS2 = chiS * chiS
    cdef double chiA3 = chiA2 * chiA
    cdef double chiS3 = chiS2 * chiS

    cdef double m1Plus3nu = -1.0 + 3.0 * nu
    cdef double m1Plus3nu2 = m1Plus3nu * m1Plus3nu
    cdef double m1Plus3nu3 = m1Plus3nu * m1Plus3nu2

    # (2,2) mode begins
    rho_coeffs[2,2][2] = -43.0 / 42.0 + (55.0 * nu) / 84.0
    rho_coeffs[2,2][3]  = (-2.0 * (chiS + chiA * dm - chiS * nu)) / 3.0

    rho_coeffs[2,2][4]  = (
        -20555.0 / 10584.0
        + 0.5 * (chiS + chiA * dm) * (chiS + chiA * dm)
        - (33025.0 * nu) / 21168.0
        + (19583.0 * nu2) / 42336.0
    )

    rho_coeffs[2,2][5]  = (
        -34.0 / 21.0 + 49.0 * nu / 18.0 + 209.0 * nu2 / 126.0
    ) * chiS + (-34.0 / 21.0 - 19.0 * nu / 42.0) * dm * chiA

    rho_coeffs[2,2][6]  = (
        1556919113.0 / 122245200.0
        + (89.0 * a2) / 252.0
        - (48993925.0 * nu) / 9779616.0
        - (6292061.0 * nu2) / 3259872.0
        + (10620745.0 * nu3) / 39118464.0
        + (41.0 * nu * pi * pi) / 192.0
    )
    rho_coeffs_log[2,2][6] = -428.0 / 105.0

    # See https://dcc.ligo.org/T1600383
    rho_coeffs[2,2][7]  = (
        a3 / 3.0
        + chiA
        * dm
        * (18733.0 / 15876.0 + (50140.0 * nu) / 3969.0 + (97865.0 * nu2) / 63504.0)
        + chiS
        * (
            18733.0 / 15876.0
            + (74749.0 * nu) / 5292.0
            - (245717.0 * nu2) / 63504.0
            + (50803.0 * nu3) / 63504.0
        )
    )

    rho_coeffs[2,2][8]  = (
        -387216563023.0 / 160190110080.0 +
            (18353.0 * a2) / 21168.0 - a2 * a2 / 8.0
    )

    rho_coeffs_log[2,2][8] = 9202.0 / 2205.0
    rho_coeffs[2,2][10]  = -16094530514677.0 / 533967033600.0
    rho_coeffs_log[2,2][10] = 439877.0 / 55566.0
    # (2,2) mode ends

    # We set test-spin terms to 0 (as in SEOBNRv4HM)
    # as no major improvement was found when trying to include them
    a = 0.0
    a2 = 0.0
    a3 = 0.0
    # (2,1) mode begins
    if dm2:
        rho_coeffs[2,1][1] = 0.0
        rho_coeffs[2,1][2] = -59.0 / 56 + (23.0 * nu) / 84.0
        rho_coeffs[2,1][3] = 0.0

        rho_coeffs[2,1][4] = (
            -47009.0 / 56448.0
            - (865.0 * a2) / 1792.0
            - (405.0 * a2 * a2) / 2048.0
            - (10993.0 * nu) / 14112.0
            + (617.0 * nu2) / 4704.0
        )
        rho_coeffs[2,1][5] = (
            (-98635.0 * a) / 75264.0
            + (2031.0 * a * a2) / 7168.0
            - (1701.0 * a2 * a3) / 8192.0
        )
        rho_coeffs[2,1][6] = (
            7613184941.0 / 2607897600.0
            + (9032393.0 * a2) / 1806336.0
            + (3897.0 * a2 * a2) / 16384.0
            - (15309.0 * a3 * a3) / 65536.0
        )
        rho_coeffs_log[2,1][6] = -107.0 / 105.0
        rho_coeffs[2,1][7] = (
            (-3859374457.0 * a) / 1159065600.0
            - (55169.0 * a3) / 16384.0
            + (18603.0 * a2 * a3) / 65536.0
            - (72171.0 * a2 * a2 * a3) / 262144.0
        )
        rho_coeffs_log[2,1][7] = 107.0 * a / 140.0
        rho_coeffs[2,1][8] = -1168617463883.0 / 911303737344.0
        rho_coeffs_log[2,1][8] = 6313.0 / 5880.0
        rho_coeffs[2,1][10] = -63735873771463.0 / 16569158860800.0
        rho_coeffs_log[2,1][10] = 5029963.0 / 5927040.0

        f_coeffs[2,1][1] = (-3.0 * (chiS + chiA / dm)) / (2.0)

        f_coeffs[2,1][3] = (
            (
                chiS * dm * (427.0 + 79.0 * nu)
                + chiA * (147.0 + 280.0 * dm * dm + 1251.0 * nu)
            )
            / 84.0
            / dm
        )
        # RC: New terms for SEOBNRv4HM, they are put to zero if use_hm == 0


        # RC: This terms are in Eq.A11 in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.084028
        f_coeffs[2,1][4] = (
            (-3.0 - 2.0 * nu) * chiA2
            + (-3.0 + nu / 2.0) * chiS2
            + (-6.0 + 21.0 * nu / 2.0) * chiS * chiA / dm
        )
        f_coeffs[2,1][5] = (
            (3.0 / 4.0 - 3.0 * nu) * chiA3 / dm
            + (
                -81.0 / 16.0
                + 1709.0 * nu / 1008.0
                + 613.0 * nu2 / 1008.0
                + (9.0 / 4.0 - 3 * nu) * chiA2
            )
            * chiS
            + 3.0 / 4.0 * chiS3
            + (
                -81.0 / 16.0
                - 703.0 * nu2 / 112.0
                + 8797.0 * nu / 1008.0
                + (9.0 / 4.0 - 6.0 * nu) * chiS2
            )
            * chiA
            / dm
        )
        '''
        This was in SEOBNRv4HM
        f_coeffs[2,1][6] = (
            (4163.0 / 252.0 - 9287.0 * nu /
                1008.0 - 85.0 * nu2 / 112.0) * chiA2
            + (4163.0 / 252.0 - 2633.0 * nu / 1008.0 + 461.0 * nu2 / 1008.0)
            * chiS2
            + (4163.0 / 126.0 - 1636.0 * nu / 21.0 + 1088.0 * nu2 / 63.0)
            * chiS
            * chiA
            / dm
        )
        '''
        f_coeffs[2,1][6] = (((16652 - 9287*nu + 720*nu**2)*chiA**2)/1008 +
                            ((16652 - 39264*nu + 9487*nu**2)*chiA*chiS)/(504*dm) +
                            ((16652 - 2633*nu + 1946*nu**2)*chiS**2)/1008)
    else:
        f_coeffs[2,1][1] = -3.0 * chiA / 2.0
        f_coeffs[2,1][3] = (
            chiS * dm * (427.0 + 79.0 * nu)
            + chiA * (147.0 + 280.0 * dm * dm + 1251.0 * nu)
        ) / 84.0
        # New terms for SEOBNRv4HM, they are put to zero if use_hm == 0

        # RC: This terms are in Eq.A11 in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.084028
        f_coeffs[2,1][4] = (-6 + 21 * nu / 2.0) * chiS * chiA
        f_coeffs[2,1][5] = (3.0 / 4.0 - 3.0 * nu) * chiA3 + (
            -81.0 / 16.0
            - 703.0 * nu2 / 112.0
            + 8797.0 * nu / 1008.0
            + (9.0 / 4.0 - 6.0 * nu) * chiS2
        ) * chiA
        '''
        This was in SEOBNRv4HM
        f_coeffs[2,1][6] = (
            (4163.0 / 126.0 - 1636.0 * nu / 21.0 + 1088.0 * nu2 / 63.0)
            * chiS
            * chiA
        )
        '''
        f_coeffs[2,1][6] = ((16652 - 39264*nu + 9487*nu**2)*chiA*chiS)/504

    # (2,1) mode ends

    # (3,3) mode begins

    if dm2:
        rho_coeffs[3,3][2] = -7.0 / 6.0 + (2.0 * nu) / 3.0
        rho_coeffs[3,3][3] = 0.0
        rho_coeffs[3,3][4] = (
            -6719.0 / 3960.0
            + a2 / 2.0
            - (1861.0 * nu) / 990.0
            + (149.0 * nu2) / 330.0
        )
        rho_coeffs[3,3][5] = (-4.0 * a) / 3.0
        rho_coeffs[3,3][6] = (
                3203101567.0 / 227026800.0
                + (5.0 * a2) / 36.0
                + (-129509.0 / 25740.0 + 41.0 / 192.0 * pi * pi) * nu
                - 274621.0 / 154440.0 * nu2
                + 12011.0 / 46332.0 * nu3
            )
        rho_coeffs_log[3,3][6] = -26.0 / 7.0
        rho_coeffs[3,3][7] = (5297.0 * a) / 2970.0 + a * a2 / 3.0
        rho_coeffs[3,3][8] = -57566572157.0 / 8562153600.0
        rho_coeffs_log[3,3][8] = 13.0 / 3.0


        rho_coeffs[3,3][10] = -903823148417327.0 / 30566888352000.0
        rho_coeffs_log[3,3][10] = 87347.0 / 13860.0

        f_coeffs[3,3][3]= (
            chiS * dm * (-4.0 + 5.0 * nu) + chiA * (-4.0 + 19.0 * nu)
        ) / (2.0 * dm)

        f_coeffs[3,3][4]= (
            3.0 / 2.0 * chiS2 * dm
            + (3.0 - 12 * nu) * chiA * chiS
            + dm * (3.0 / 2.0 - 6.0 * nu) * chiA2
        ) / (dm)
        f_coeffs[3,3][5] = (
            dm * (241.0 / 30.0 * nu2 + 11.0 /
                    20.0 * nu + 2.0 / 3.0) * chiS
            + (407.0 / 30.0 * nu2 - 593.0 / 60.0 * nu + 2.0 / 3.0) * chiA
        ) / (dm)
        f_coeffs[3,3][6] = (
            dm * (6.0 * nu2 - 27.0 / 2.0 * nu - 7.0 / 4.0) * chiS2
            + (44.0 * nu2 - 1.0 * nu - 7.0 / 2.0) * chiA * chiS
            + dm * (-12 * nu2 + 11.0 / 2.0 * nu - 7.0 / 4.0) * chiA2
        ) / dm
        f_coeffs_vh[3,3][6] =  (
                dm * (593.0 / 108.0 * nu - 81.0 / 20.0) * chiS
                + (7339.0 / 540.0 * nu - 81.0 / 20.0) * chiA
            ) / (dm)

    else:
        f_coeffs[3,3][3] = chiA * 3.0 / 8.0

        # RC: This terms are in Eq.A10 in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.084028
        f_coeffs[3,3][4] = (3.0 - 12 * nu) * chiA * chiS
        f_coeffs[3,3][5] = (
            407.0 / 30.0 * nu2 - 593.0 / 60.0 * nu + 2.0 / 3.0
        ) * chiA
        f_coeffs[3,3][6] = (44.0 * nu2 - 1.0 * nu - 7.0 / 2.0) * chiA * chiS
        f_coeffs_vh[3,3][6] = (7339.0 / 540.0 * nu - 81.0 / 20.0) * chiA

    # (3,3) mode ends

    # (3,2) mode begins
    rho_coeffs[3,2][1] = (4.0 * chiS * nu) / (-3.0 * m1Plus3nu)
    rho_coeffs[3,2][2] = (328.0 - 1115.0 * nu + 320.0 *
                      nu2) / (270.0 * m1Plus3nu)

    rho_coeffs[3,2][3] = (
        2.0
        * (
            45.0 * a * m1Plus3nu3
            - a
            * nu
            * (328.0 - 2099.0 * nu + 5.0 * (733.0 + 20.0 * a2) * nu2 - 960.0 * nu3)
        )
    ) / (405.0 * m1Plus3nu3)

    rho_coeffs[3,2][4] = a2 / 3.0 + (
        -1444528.0
        + 8050045.0 * nu
        - 4725605.0 * nu2
        - 20338960.0 * nu3
        + 3085640.0 * nu2 * nu2
    ) / (1603800.0 * m1Plus3nu2)
    rho_coeffs[3,2][5] = (-2788.0 * a) / 1215.0
    rho_coeffs[3,2][6] = 5849948554.0 / 940355325.0 + (488.0 * a2) / 405.0
    rho_coeffs_log[3,2][6] = -104.0 / 63.0
    rho_coeffs[3,2][8] = -10607269449358.0 / 3072140846775.0
    rho_coeffs_log[3,2][8] = 17056.0 / 8505.0
    # (3,2) mode ends

    # (3,1) mode begins
    if dm2:

        rho_coeffs[3,1][2] = -13.0 / 18.0 - (2.0 * nu) / 9.0
        rho_coeffs[3,1][3] = 0.0
        rho_coeffs[3,1][4] = (
            101.0 / 7128.0
            - (5.0 * a2) / 6.0
            - (1685.0 * nu) / 1782.0
            - (829.0 * nu2) / 1782.0
        )
        rho_coeffs[3,1][5] = (4.0 * a) / 9.0
        rho_coeffs[3,1][6] = 11706720301.0 / 6129723600.0 - (49.0 * a2) / 108.0
        rho_coeffs_log[3,1][6] = -26.0 / 63.0
        rho_coeffs[3,1][7] = (-2579.0 * a) / 5346.0 + a * a2 / 9.0
        rho_coeffs[3,1][8] = 2606097992581.0 / 4854741091200.0
        rho_coeffs_log[3,1][8] = 169.0 / 567.0

        f_coeffs[3,1][3] = (
            chiA * (-4.0 + 11.0 * nu) + chiS * dm * (-4.0 + 13.0 * nu)
        ) / (2.0 * dm)

    else:
        f_coeffs[3,1][3]= -chiA * 5.0 / 8.0
    # (3,1) mode ends

    # (4,4) mode begins
    rho_coeffs[4,4][2] = (1614.0 - 5870.0 * nu + 2625.0 *
                      nu2) / (1320.0 * m1Plus3nu)
    rho_coeffs[4,4][3] = (
        chiA * (10.0 - 39.0 * nu) * dm + chiS *
                (10.0 - 41.0 * nu + 42.0 * nu2)
    ) / (15.0 * m1Plus3nu)

    # RC: This terms are in Eq.A8 in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.084028
    rho_coeffs[4,4][4] = (
        (
            -511573572.0
            + 2338945704.0 * nu
            - 313857376.0 * nu2
            - 6733146000.0 * nu3
            + 1252563795.0 * nu2 * nu2
        )
        / (317116800.0 * m1Plus3nu2)
        + chiS2 / 2.0
        + dm * chiS * chiA
        + dm2 * chiA2 / 2.0
    )
    rho_coeffs[4,4][5] = chiA * dm * (
        -8280.0 + 42716.0 * nu - 57990.0 * nu2 + 8955 * nu3
    ) / (6600.0 * m1Plus3nu2) + chiS * (
        -8280.0
        + 66284.0 * nu
        - 176418.0 * nu2
        + 128085.0 * nu3
        + 88650 * nu2 * nu2
    ) / (
        6600.0 * m1Plus3nu2
    )
    rho_coeffs[4,4][8] = -172066910136202271.0 / 19426955708160000.0
    rho_coeffs_log[4,4][8] = 845198.0 / 190575.0
    rho_coeffs[4,4][10] = -17154485653213713419357.0 / 568432724020761600000.0
    rho_coeffs_log[4,4][10] = 22324502267.0 / 3815311500.0

    rho_coeffs[4,4][6] = 16600939332793.0 / 1098809712000.0 + (217.0 * a2) / 3960.0
    rho_coeffs_log[4,4][6] = -12568.0 / 3465.0
    # (4,4) mode ends
    # (4,3) mode begins
    if dm2:
        rho_coeffs[4,3][1] = 0.0
        rho_coeffs[4,3][2] = (222.0 - 547.0 * nu + 160.0 * nu2) / (
            176.0 * (-1.0 + 2.0 * nu)
        )
        rho_coeffs[4,3][4] = -6894273.0 / 7047040.0 + (3.0 * a2) / 8.0
        rho_coeffs[4,3][5] = (-12113.0 * a) / 6160.0
        rho_coeffs[4,3][6] = 1664224207351.0 / 195343948800.0
        rho_coeffs_log[4,3][6] = -1571.0 / 770.0
        f_coeffs[4,3][1] = (5.0 * (chiA - chiS * dm) * nu) / (
            2.0 * dm * (-1.0 + 2.0 * nu)
        )

    else:
        f_coeffs[4,3][1] = -5.0 * chiA / 4.0
    # (4,3) mode ends

    # (4,2) mode begins
    rho_coeffs[4,2][2] = (1146.0 - 3530.0 * nu + 285.0 *
                      nu2) / (1320.0 * m1Plus3nu)
    rho_coeffs[4,2][3] = (
        chiA * (10.0 - 21.0 * nu) * dm + chiS *
                (10.0 - 59.0 * nu + 78.0 * nu2)
    ) / (15.0 * m1Plus3nu)
    rho_coeffs[4,2][4] = a2 / 2.0 + (
        -114859044.0
        + 295834536.0 * nu
        + 1204388696.0 * nu2
        - 3047981160.0 * nu3
        - 379526805.0 * nu2 * nu2
    ) / (317116800.0 * m1Plus3nu2)
    rho_coeffs[4,2][5] = (-7.0 * a) / 110.0
    rho_coeffs[4,2][6] = 848238724511.0 / 219761942400.0 + (2323.0 * a2) / 3960.0
    rho_coeffs_log[4,2][6] = -3142.0 / 3465.0
    # (4,2) mode ends

    # (4,1) mode begins
    if dm2:

        rho_coeffs[4,1][1] = 0.0
        rho_coeffs[4,1][2] = (602.0 - 1385.0 * nu + 288.0 * nu2) / (
            528.0 * (-1.0 + 2.0 * nu)
        )
        rho_coeffs[4,1][4] = -7775491.0 / 21141120.0 + (3.0 * a2) / 8.0
        rho_coeffs[4,1][5] = (-20033.0 * a) / 55440.0 - (5 * a * a2) / 6.0
        rho_coeffs[4,1][6] = 1227423222031.0 / 1758095539200.0
        rho_coeffs_log[4,1][6] = -1571.0 / 6930.0
        f_coeffs[4,1][1] = (5.0 * (chiA - chiS * dm) * nu) / (
            2.0 * dm * (-1.0 + 2.0 * nu)
        )

    else:
        f_coeffs[4,1][1] = -5.0 * chiA / 4.0

    # (4,1) mode ends

    # (5,5) mode begins
    if dm2:

        rho_coeffs[5,5][2] = (487.0 - 1298.0 * nu + 512.0 * nu2) / (
            390.0 * (-1.0 + 2.0 * nu)
        )
        rho_coeffs[5,5][3] = (-2.0 * a) / 3.0
        rho_coeffs[5,5][4] = -3353747.0 / 2129400.0 + a2 / 2.0
        rho_coeffs[5,5][5] = -241.0 * a / 195.0



        # RC: This terms are in Eq.A9 in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.084028
        rho_coeffs[5,5][6] = 190606537999247.0 / 11957879934000.0
        rho_coeffs_log[5,5][6] = -1546.0 / 429.0
        rho_coeffs[5,5][8] = -1213641959949291437.0 / 118143853747920000.0
        rho_coeffs_log[5,5][8] = 376451.0 / 83655.0
        rho_coeffs[5,5][10] = -150082616449726042201261.0 / 4837990810977324000000.0
        rho_coeffs_log[5,5][10] = 2592446431.0 / 456756300.0

        f_coeffs[5,5][3] = chiA / dm * (
            10.0 / (3.0 * (-1.0 + 2.0 * nu))
            - 70.0 * nu / (3.0 * (-1.0 + 2.0 * nu))
            + 110.0 * nu2 / (3.0 * (-1.0 + 2.0 * nu))
        ) + chiS * (
            10.0 / (3.0 * (-1.0 + 2.0 * nu))
            - 10.0 * nu / (-1.0 + 2.0 * nu)
            + 10 * nu2 / (-1.0 + 2.0 * nu)
        )
        f_coeffs[5,5][4] = (
            chiS2
            * (-5.0 / (2.0 * (-1.0 + 2.0 * nu)) + 5.0 * nu / (-1.0 + 2.0 * nu))
            + chiA
            * chiS
            / dm
            * (
                -5.0 / (-1.0 + 2.0 * nu)
                + 30.0 * nu / (-1.0 + 2.0 * nu)
                - 40.0 * nu2 / (-1.0 + 2.0 * nu)
            )
            + chiA2
            * (
                -5.0 / (2.0 * (-1.0 + 2.0 * nu))
                + 15.0 * nu / (-1.0 + 2.0 * nu)
                - 20.0 * nu2 / (-1.0 + 2.0 * nu)
            )
        )

    else:

        # RC: This terms are in Eq.A12 in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.084028
        f_coeffs[5,5][3] = chiA * (
            10.0 / (3.0 * (-1.0 + 2.0 * nu))
            - 70.0 * nu / (3.0 * (-1.0 + 2.0 * nu))
            + 110.0 * nu2 / (3.0 * (-1.0 + 2.0 * nu))
        )
        f_coeffs[5,5][4] = (
            chiA
            * chiS
            * (
                -5.0 / (-1.0 + 2.0 * nu)
                + 30.0 * nu / (-1.0 + 2.0 * nu)
                - 40.0 * nu2 / (-1.0 + 2.0 * nu)
            )
        )
    # (5,5) mode ends

    # (5,4) mode begins
    rho_coeffs[5,4][2] = (
        -17448.0 + 96019.0 * nu - 127610.0 * nu2 + 33320.0 * nu3
    ) / (13650.0 * (1.0 - 5.0 * nu + 5.0 * nu2))
    rho_coeffs[5,4][3] = (-2.0 * a) / 15.0
    rho_coeffs[5,4][4] = -16213384.0 / 15526875.0 + (2.0 * a2) / 5.0
    # (5,4) mode ends

    # (5,3)  mode begins
    if dm2:

        rho_coeffs[5,3][2] = (375.0 - 850.0 * nu + 176.0 * nu2) / (
            390.0 * (-1.0 + 2.0 * nu)
        )
        rho_coeffs[5,3][3] = (-2.0 * a) / 3.0
        rho_coeffs[5,3][4] = -410833.0 / 709800.0 + a2 / 2.0
        rho_coeffs[5,3][5] = -103.0 * a / 325.0
    # (5,3) mode ends

    # (5,2) mode begins
    rho_coeffs[5,2][2] = (
        -15828.0 + 84679.0 * nu - 104930.0 * nu2 + 21980.0 * nu3
    ) / (13650.0 * (1.0 - 5.0 * nu + 5.0 * nu2))
    rho_coeffs[5,2][3] = (-2.0 * a) / 15.0
    rho_coeffs[5,2][4] = -7187914.0 / 15526875.0 + (2.0 * a2) / 5.0
    # (5,2) mode ends

    # (5,1) mode begins
    if dm2:
        rho_coeffs[5,1][2] = (319.0 - 626.0 * nu + 8.0 * nu2) / (
            390.0 * (-1.0 + 2.0 * nu)
        )
        rho_coeffs[5,1][3] = (-2.0 * a) / 3.0
        rho_coeffs[5,1][4] = -31877.0 / 304200.0 + a2 / 2.0
        rho_coeffs[5,1][5] = 139.0 * a / 975.0
    # (5,1) mode ends

    # (6,6) mode begins
    rho_coeffs[6,6][2] = (-106.0 + 602.0 * nu - 861.0 * nu2 + 273.0 * nu3) / (
        84.0 * (1.0 - 5.0 * nu + 5.0 * nu2)
    )
    rho_coeffs[6,6][3] = (-2.0 * a) / 3.0
    rho_coeffs[6,6][4] = -1025435.0 / 659736.0 + a2 / 2.0
    # (6,6) mode ends

    # (6,5) mode begins
    if dm2:
        rho_coeffs[6,5][2] = (-185.0 + 838.0 * nu - 910.0 * nu2 + 220.0 * nu3) / (
            144.0 * (dm2 + 3.0 * nu2)
        )
        rho_coeffs[6,5][3] = -2.0 * a / 9.0
    # (6,5) mode ends

    # (6,4) mode begins
    rho_coeffs[6,4][2] = (-86.0 + 462.0 * nu - 581.0 * nu2 + 133.0 * nu3) / (
        84.0 * (1.0 - 5.0 * nu + 5.0 * nu2)
    )
    rho_coeffs[6,4][3] = (-2.0 * a) / 3.0
    rho_coeffs[6,4][4] = -476887.0 / 659736.0 + a2 / 2.0
    # (6,4) mode ends

    # (6,3) mode begins
    if dm2:
        rho_coeffs[6,3][2] = (-169.0 + 742.0 * nu - 750.0 * nu2 + 156.0 * nu3) / (
            144.0 * (dm2 + 3.0 * nu2)
        )
        rho_coeffs[6,3][3] = -2.0 * a / 9.0
    # (6,3) mode ends

    # (6,2) mode begins
    rho_coeffs[6,2][2] = (-74.0 + 378.0 * nu - 413.0 * nu2 + 49.0 * nu3) / (
        84.0 * (1.0 - 5.0 * nu + 5.0 * nu2)
    )
    rho_coeffs[6,2][3] = (-2.0 * a) / 3.0
    rho_coeffs[6,2][4] = -817991.0 / 3298680.0 + a2 / 2.0
    # (6,2) mode ends

    # (6,1) mode begins
    if dm2:
        rho_coeffs[6,1][2] = (-161.0 + 694.0 * nu - 670.0 * nu2 + 124.0 * nu3) / (
            144.0 * (dm2 + 3.0 * nu2)
        )
        rho_coeffs[6,1][3] = -2.0 * a / 9.0
    # (6,1) mode ends

    # l=7 modes begin
    if dm2:

        rho_coeffs[7,7][2] = (-906.0 + 4246.0 * nu - 4963.0 * nu2 + 1380.0 * nu3) / (
            714.0 * (dm2 + 3.0 * nu2)
        )
        rho_coeffs[7,7][3] = -2.0 * a / 3.0

    rho_coeffs[7,6][2] = (
        2144.0 - 16185.0 * nu + 37828.0 * nu2 - 29351.0 * nu3 + 6104.0 * nu2 * nu2
    ) / (1666.0 * (-1 + 7 * nu - 14 * nu2 + 7 * nu3))

    if dm2:
        rho_coeffs[7,5][2] = (-762.0 + 3382.0 * nu - 3523.0 * nu2 + 804.0 * nu3) / (
            714.0 * (dm2 + 3.0 * nu2)
        )
        rho_coeffs[7,5][3] = -2.0 * a / 3.0

    rho_coeffs[7,4][2] = (
        17756.0
        - 131805.0 * nu
        + 298872.0 * nu2
        - 217959.0 * nu3
        + 41076.0 * nu2 * nu2
    ) / (14994.0 * (-1.0 + 7.0 * nu - 14.0 * nu2 + 7.0 * nu3))

    if dm2:
        rho_coeffs[7,3][2] = (-666.0 + 2806.0 * nu - 2563.0 * nu2 + 420.0 * nu3) / (
            714.0 * (dm2 + 3.0 * nu2)
        )
        rho_coeffs[7,3][3] = -2.0 * a / 3.0

    rho_coeffs[7,2][2] = (
        16832.0
        - 123489.0 * nu
        + 273924.0 * nu2
        - 190239.0 * nu3
        + 32760.0 * nu2 * nu2
    ) / (14994.0 * (-1.0 + 7.0 * nu - 14.0 * nu2 + 7.0 * nu3))

    if dm2:
        rho_coeffs[7,1][2] = (-618.0 + 2518.0 * nu - 2083.0 * nu2 + 228.0 * nu3) / (
            714.0 * (dm2 + 3.0 * nu2)
        )
        rho_coeffs[7,1][3] = -2.0 * a / 3.0

    # l =7 modes end

    # l=8 modes begin
    rho_coeffs[8,8][2] = (
        3482.0 - 26778.0 * nu + 64659.0 * nu2 -
            53445.0 * nu3 + 12243.0 * nu2 * nu2
    ) / (2736.0 * (-1.0 + 7.0 * nu - 14.0 * nu2 + 7.0 * nu3))

    if dm2:
        rho_coeffs[8,7][2] = (
            23478.0
            - 154099.0 * nu
            + 309498.0 * nu2
            - 207550.0 * nu3
            + 38920 * nu2 * nu2
        ) / (18240.0 * (-1 + 6 * nu - 10 * nu2 + 4 * nu3))

    rho_coeffs[8,6][2] = (
        1002.0 - 7498.0 * nu + 17269.0 * nu2 - 13055.0 * nu3 + 2653.0 * nu2 * nu2
    ) / (912.0 * (-1.0 + 7.0 * nu - 14.0 * nu2 + 7.0 * nu3))

    if dm2:
        rho_coeffs[8,5][2] = (
            4350.0
            - 28055.0 * nu
            + 54642.0 * nu2
            - 34598.0 * nu3
            + 6056.0 * nu2 * nu2
        ) / (3648.0 * (-1.0 + 6.0 * nu - 10.0 * nu2 + 4.0 * nu3))

    rho_coeffs[8,4][2] = (
        2666.0 - 19434.0 * nu + 42627.0 * nu2 - 28965.0 * nu3 + 4899.0 * nu2 * nu2
    ) / (2736.0 * (-1.0 + 7.0 * nu - 14.0 * nu2 + 7.0 * nu3))

    if dm2:
        rho_coeffs[8,3][2] = (
            20598.0
            - 131059.0 * nu
            + 249018.0 * nu2
            - 149950.0 * nu3
            + 24520.0 * nu2 * nu2
        ) / (18240.0 * (-1.0 + 6.0 * nu - 10.0 * nu2 + 4.0 * nu3))

    rho_coeffs[8,2][2] = (
        2462.0 - 17598.0 * nu + 37119.0 * nu2 - 22845.0 * nu3 + 3063.0 * nu2 * nu2
    ) / (2736.0 * (-1.0 + 7.0 * nu - 14.0 * nu2 + 7.0 * nu3))

    if dm2:
        rho_coeffs[8,1][2] = (
            20022.0
            - 126451.0 * nu
            + 236922.0 * nu2
            - 138430.0 * nu3
            + 21640.0 * nu2 * nu2
        ) / (18240.0 * (-1.0 + 6.0 * nu - 10.0 * nu2 + 4.0 * nu3))

    if extra_PN_terms:
        # NB: One needs to be careful when adding new terms here.
        # If terms at the given order already exist then it may be
        # appropriate to _add_ new terms to values defined above,
        # i.e with "+="". However, for brand new terms they should
        # just be defined with "=" since the coefficients are not
        # reset to 0 at every step.
        # These terms were not in SEOBNRv4HM
        # Add the NLO  spin-squared term at 3PN
        # We need to get rid of the test-spin term
        rho_coeffs[2,2][6] -=(89.0 * atemp*atemp) / 252.0
        rho_coeffs[2,2][6] += (
            ((178. - 457.*nu- 972.*nu**2)*chiA**2)/504. +
            (dm*(178. - 781.*nu)*chiA*chiS)/252. +
            ((178. - 1817.*nu+ 560.*nu**2)*chiS**2)/504.)
        # Add LO spin-cubed term at 3.5 PN
        # We need to get rid of the test-spin term
        # We need atemp because we have set a to zero afer the (2,2) mode
        rho_coeffs[2,2][7] -= atemp**3/3.0
        rho_coeffs[2,2][7] += (((dm - 4.*dm*nu)*chiA**3)/3. +
                (1. - 3.*nu - 4.*nu**2)*chiA**2*chiS +
                ( dm + 2.*dm*nu)*chiA*chiS**2 +
                (1./3. + nu)*chiS**3)

        # NB: from now on we don't need to worry about subtracting out the
        # test-spin terms since they were set to 0 for modes above (2,2)

        # All the known spin terms in (3,2)
        rho_coeffs[3,2][2] +=  -(16.*nu**2*chiS**2)/(9.*(1. - 3.*nu)**2)
        rho_coeffs[3,2][3] += ((dm*(2. + 13.*nu)*chiA)/(9. - 27.*nu) +
                                ((90. - 1478.*nu + 2515.*nu**2 + 3035.*nu**3)*chiS)/
                                (405.*(1. - 3.*nu)**2) - (320.*nu**3*chiS**3)/
                                (81.*(-1. + 3.*nu)**3))
        rho_coeffs[3,2][4] +=  (((1. - 9.*nu + 12.*nu**2)*chiA**2)/(3. - 9.*nu) -
                                (2.*dm*(-9. + 44.*nu + 25.*nu**2)*chiA*chiS)/
                                (27.*(1. - 3.*nu)**2) + ((-81. + 387.*nu - 1435.*nu**2 +
                                1997.*nu**3 + 2452.*nu**4)*chiS**2)/(243.*(-1 + 3.*nu)**3))
        rho_coeffs[3,2][5] += ((dm*(-245344. + 1128531.*nu - 1514740.*nu**2 +889673.*nu**3)*chiA)/(106920.*(1 - 3*nu)**2) +
                            ((2208096. - 20471053.*nu + 70519165.*nu**2 - 101706029.*nu**3 +40204523.*nu**4 + 11842250.*nu**5)*chiS)/
                            (962280.*(-1. + 3.*nu)**3) - (8.*nu*(1. - 9.*nu + 12.*nu**2)*chiA**2*chiS)/(9.*(1. - 3.*nu)**2) -
                            (16.*dm*nu*(-9. + 46.*nu + 38.*nu**2)*chiA*chiS**2)/(81.*(-1. + 3.*nu)**3) + (8.*nu*(-243. + 1269.*nu - 5029.*nu**2 +
                            5441.*nu**3 + 12022.*nu**4)*chiS**3)/(2187.*(1. - 3.*nu)**4))

        # All the known spin terms in (4,3)
        # NB: odd m so these are f-coeffs and we must treat the dm->0 issue as always

        if dm2:

            f_coeffs[4,3][3] = ((nu*(-2661. + 3143.*nu)*chiA)/(132.*dm*(-1. + 2.*nu)) +
                            (23.*nu*(87. + 23.*nu)*chiS)/(132.*(-1. + 2.*nu)))


            f_coeffs[4,3][4] = (((9. - 74.*nu + 72.*nu**2)*chiA**2)/(6. - 12.*nu) +
                            ((18. - 108.*nu + 137.*nu**2)*chiA*chiS)/(6.*dm - 12.*dm*nu) + ((9. + 2.*nu + 35.*nu**2)*
                            chiS**2)/(6. - 12.*nu))


        else:
            f_coeffs[4,3][3] = ((nu*(-2661. + 3143.*nu)*chiA)/(132.*(-1. + 2.*nu)))
            f_coeffs[4,3][4] = ((18. - 108.*nu + 137.*nu**2)*chiA*chiS)/(6. - 12.*nu)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef public void compute_delta_coeffs(double nu,double dm, double a,double chiS,double chiA,
    double complex[:,:,:] delta_coeffs, double complex[:,:,:] delta_coeffs_vh):

    """
    Compute the phase residual coefficients.
    See Sec. 2B of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.79.064004
    See Eq. 59-63 of https://dcc.ligo.org/LIGO-T2300060 (SEOBNRv5_theory.pdf) for new terms, rest copied from SEOBNRv4HM LAL code

    Coefficients can be found in:
    - https://dcc.ligo.org/LIGO-T2300060 (SEOBNRv5_theory.pdf and SEOBNRv5HM.pdf)
    - Appendix A of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.98.084028
    - Eqs. (2.4) to (2.6) of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.95.044028
    - https://journals.aps.org/prd/pdf/10.1103/PhysRevD.83.064003
    - https://journals.aps.org/prd/pdf/10.1103/PhysRevD.86.024011
    """

    cdef double nu2 = nu*nu
    cdef double nu3 = nu*nu2

    cdef double a2 = a*a
    cdef double a3 = a*a2
    cdef double dm2 = dm*dm

    cdef double chiA2 = chiA * chiA
    cdef double chiS2 = chiS * chiS
    cdef double chiA3 = chiA2 * chiA
    cdef double chiS3 = chiS2 * chiS

    cdef double m1Plus3nu = -1.0 + 3.0 * nu
    cdef double m1Plus3nu2 = m1Plus3nu * m1Plus3nu
    cdef double m1Plus3nu3 = m1Plus3nu * m1Plus3nu2

    cdef double aDelta = 0.0

    #(2,2) mode begins
    delta_coeffs_vh[2,2][3] = 7.0 / 3.0
    # See https://dcc.ligo.org/T1600383

    delta_coeffs_vh[2,2][6] = (
        -4.0 / 3.0 * (dm * chiA + chiS * (1 - 2 * nu)) + (428.0 * pi) / 105.0
    )

    delta_coeffs[2,2][8] = (20.0 * aDelta) / 63.0
    delta_coeffs_vh[2,2][9] = -2203.0 / 81.0 + (1712.0 * pi * pi) / 315.0
    delta_coeffs[2,2][5] = -24.0 * nu
    delta_coeffs[2,2][6] = 0.0

    # (2,2) mode ends

    # (2,1) mode begins

    delta_coeffs_vh[2,1][3] = 2.0 / 3.0
    delta_coeffs_vh[2,1][6] = (-17.0 * aDelta) / 35.0 + (107.0 * pi) / 105.0
    delta_coeffs_vh[2,1][7] = (3.0 * aDelta * aDelta) / 140.0
    delta_coeffs_vh[2,1][9] = -272.0 / 81.0 + (214.0 * pi * pi) / 315.0
    """
    This was in SEOBNRv4HM
    delta_coeffs[2,1][5] = -493.0 * nu / 42.0
    """
    delta_coeffs[2,1][5] = -25.0 * nu /2

    # (2,1) mode ends

    # (3,3) mode begins
    # l = 3, Eqs. A9a - A9c for rho, Eqs. A15b and A15c for f,
    # Eqs. 22 - 24 of DIN and Eqs. 27c - 27e of PBFRT for delta
    delta_coeffs_vh[3,3][3] = 13.0 / 10.0
    delta_coeffs_vh[3,3][6] = (-81.0 * aDelta) / 20.0 + (39.0 * pi) / 7.0
    delta_coeffs_vh[3,3][9] = -227827.0 / 3000.0 + (78.0 * pi * pi) / 7.0
    delta_coeffs[3,3][5] = -80897.0 * nu / 2430.0
    # (3,3) mode ends

    # (3,2) mode begins
    delta_coeffs_vh[3,2][3] = (10.0 + 33.0 * nu) / (-15.0 * m1Plus3nu)
    delta_coeffs_vh[3,2][4] = 4.0 * aDelta
    delta_coeffs_vh[3,2][6] = (-136.0 * aDelta) / 45.0 + (52.0 * pi) / 21.0
    delta_coeffs_vh[3,2][9] = -9112.0 / 405.0 + (208.0 * pi * pi) / 63.0
    # (3,2) mode ends

    # (3,1) mode begins
    delta_coeffs_vh[3,1][3] = 13.0 / 30.0
    delta_coeffs_vh[3,1][6] = (61.0 * aDelta) / 20.0 + (13.0 * pi) / 21.0
    delta_coeffs_vh[3,1][7] = (-24.0 * aDelta * aDelta) / 5.0
    delta_coeffs_vh[3,1][9] = -227827.0 / 81000.0 + (26.0 * pi * pi) / 63.0
    delta_coeffs[3,1][5] = -17.0 * nu / 10.0
    # (3,1) mode ends

    # (4,4) mode begins
    delta_coeffs_vh[4,4][3] = (112.0 + 219.0 * nu) / (-120.0 * m1Plus3nu)
    delta_coeffs_vh[4,4][6] = (-464.0 * aDelta) / 75.0 + (25136.0 * pi) / 3465.0

    # RC: This terms are in Eq.A15 in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.084028
    delta_coeffs_vh[4,4][9] = -55144.0 / 375.0 + 201088.0 * pi * pi / 10395.0

    # SO: what is going on with delta44v5: it's declared but never set in the LAL code!

    # (4,4) mode ends

    # (4,3) mode begins
    delta_coeffs_vh[4,3][3] = (486.0 + 4961.0 * nu) / (810.0 * (1.0 - 2.0 * nu))
    delta_coeffs_vh[4,3][4] = (11.0 * aDelta) / 4.0
    delta_coeffs_vh[4,3][6] = 1571.0 * pi / 385.0
    # (4,3) mode ends

    # (4,2) mode begins
    delta_coeffs_vh[4,2][3] = (7.0 * (1.0 + 6.0 * nu)) / (-15.0 * m1Plus3nu)
    delta_coeffs_vh[4,2][6] = (212.0 * aDelta) / 75.0 + (6284.0 * pi) / 3465.0
    # (4,2) mode ends

    # (4,1) mode begins
    delta_coeffs_vh[4,1][3] = (2.0 + 507.0 * nu) / (10.0 * (1.0 - 2.0 * nu))
    delta_coeffs_vh[4,1][4] = (11.0 * aDelta) / 12.0
    delta_coeffs_vh[4,1][6] = 1571.0 * pi / 3465.0
    # (4,1) mode ends

    # l=5 modes begin
    delta_coeffs_vh[5,5][3] = (96875.0 + 857528.0 * nu) / (131250.0 * (1 - 2 * nu))

    # RC: This terms are in Eq.A16 in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.084028
    delta_coeffs_vh[5,5][6] = 3865.0 / 429.0 * pi
    delta_coeffs_vh[5,5][9] = (
        -7686949127.0 + 954500400.0 * pi * pi
    ) / 31783752.0

    delta_coeffs_vh[5,4][3] = 8.0 / 15.0
    delta_coeffs_vh[5,4][4] = 12.0 * aDelta / 5.0

    delta_coeffs_vh[5,3][3] = 31.0 / 70.0

    delta_coeffs_vh[5,2][3] = 4.0 / 15.0
    delta_coeffs_vh[5,2][4] = 6.0 * aDelta / 5.0

    delta_coeffs_vh[5,1][3] = 31.0 / 210.0
    # ell = 5 modes end

    # l=6 modes begin
    delta_coeffs_vh[6,6][3] = 43.0 / 70.0
    delta_coeffs_vh[6,5][3] = 10.0 / 21.0
    delta_coeffs_vh[6,4][3] = 43.0 / 105.0
    delta_coeffs_vh[6,3][3] = 2.0 / 7.0
    delta_coeffs_vh[6,2][3] = 43.0 / 210.0
    delta_coeffs_vh[6,1][3] = 2.0 / 21.0
    # l=6 modes end

    # l=7 modes begin
    delta_coeffs_vh[7,7][3] = 19.0 / 36.0
    delta_coeffs_vh[7,5][3] = 95.0 / 252.0
    delta_coeffs_vh[7,3][3] = 19.0 / 84.0
    delta_coeffs_vh[7,1][3] = 19.0 / 252.0
    # l=7 modes end


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef double complex compute_deltalm_single(double[] vs, double[] vhs,int l, int m, FluxParams fl):
    """
    Compute the  full \delta_{\ell m} contribution for a given mode
    """
    cdef int j
    cdef double complex delta = 0.0
    for j in range(PN_limit):
        delta += fl.delta_coeffs[l,m,j]*vs[j] + fl.delta_coeffs_vh[l,m,j]*vhs[j]
    return delta

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cdef void compute_delta(double v,double vh,double nu, EOBParams eob_pars):
    """
    Compute the  full \delta_{\ell m} contribution for all modes
    """
    cdef int i,j,l,m
    cdef double vs[PN_limit]
    cdef double vhs[PN_limit]
    cdef double complex delta  = 0.0
    for i in range(PN_limit):
        vs[i] = v**i
        vhs[i] = vh**i
    cdef FluxParams fl = eob_pars.flux_params
    for l in range(2,ell_max+1):
        for m in range(1,l+1):
            fl.deltalm[l,m] = compute_deltalm_single(vs,vhs,l,m,fl)



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)

cdef double complex compute_extra_flm_terms(int l,int m,double vh,EOBParams eob_pars):
    """
    Compute the complex term in f_{33}. See last term in Eq(A10) of https://arxiv.org/pdf/1803.10701.pdf
    """
    cdef double vh3 = vh**3
    cdef double complex extra_term = 0.0
    if l==3 and m==3:
        extra_term = I*vh3 * vh3 * eob_pars.flux_params.f_coeffs_vh[3,3][6]
    return extra_term

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef double complex compute_rholm_single(double[] vs,double vh, int l, int m, EOBParams eob_pars):
    """
    Compute the full \rho_{\ell m}​ contribution for a given mode
    """
    cdef int j
    cdef double v = vs[1]
    cdef double nu = eob_pars.p_params.nu
    cdef double complex rho_final  = 0.0
    cdef double rho  = 1.0

    cdef double rho_log = 0.0
    cdef double f = 0.0
    cdef double complex f_final = 0.0
    cdef double eulerlogxabs =  euler_gamma + log(2.0 * m * v)
    cdef double complex extra_flm_term
    # This is just computing rho_coeffs \dot v

    if m%2:
        # For odd m modes we need to compute f
        for j in range(1,PN_limit):
            rho+= (eob_pars.flux_params.rho_coeffs[l,m,j]+eob_pars.flux_params.extra_coeffs[l,m,j]+ (eob_pars.flux_params.rho_coeffs_log[l,m,j]+eob_pars.flux_params.extra_coeffs_log[l,m,j])*eulerlogxabs)*vs[j]
            f += eob_pars.flux_params.f_coeffs[l,m,j]*vs[j]
    else:
        # For even m modes we only need rho
        for j in range(1,PN_limit):
            rho+= (eob_pars.flux_params.rho_coeffs[l,m,j]+eob_pars.flux_params.extra_coeffs[l,m,j]+(eob_pars.flux_params.rho_coeffs_log[l,m,j]+eob_pars.flux_params.extra_coeffs_log[l,m,j])*eulerlogxabs)*vs[j]

    # Deal with the complex amplitude term
    f_final = f
    if l==3 and m==3:
        extra_flm_term = compute_extra_flm_terms(l,m,vh,eob_pars)
        f_final += extra_flm_term


    rho_final = rho**l
    if fabs(nu-0.25)<1e-14 and m%2:
        rho_final = f_final
    else:
        rho_final += f_final
    return rho_final


'''
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cdef void compute_rholm(double v,double vh,double nu, EOBParams eob_pars):
    """
    Compute the full \rho_{\ell m}​ contribution for all modes.
    """
    cdef int i,l,m
    cdef double vs[PN_limit]
    for i in range(PN_limit):
        vs[i] = v**i

    for l in range(2,ell_max+1):
        for m in range(1,l+1):
            eob_pars.flux_params.rholm[l,m] = compute_rholm_single(vs,vh,l,m,eob_pars)
'''

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cdef void compute_rholm(double v,double vh,double nu, EOBParams eob_pars):
    cdef int i,l,m
    cdef double vs[PN_limit]
    vs[0] = 0
    vs[1] = v
    for i in range(2,PN_limit):
        vs[i] = v*vs[i-1]

    for l in range(2,ell_max+1):
        for m in range(1,l+1):
            eob_pars.flux_params.rholm[l,m] = compute_rholm_single(vs,vh,l,m,eob_pars)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.profile(True)
@cython.linetrace(True)
cdef double  EOBFluxCalculateNewtonianMultipoleAbs(
    double x, double phi, int l, int m, double [:,:] params
):
    """
    Compute the Newtonian multipole (optimised for the flux calculation).
    """
    cdef double param = params[l,m]
    cdef int epsilon = (l + m) % 2


    cdef double y = ylms[m][l-epsilon]
    cdef double multipole = param * x ** ((l + epsilon) / 2.0)
    multipole *= y
    return multipole


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cdef void update_rho_coeffs(double[:,:,:] rho_coeffs, double[:,:,:] extra_coeffs):
    cdef int l,m,i
    cdef double temp = 0.0
    for l in range(2,ell_max+1):
        for m in range(1,l+1):
            for i in range(PN_limit):
                temp = extra_coeffs[l,m,i]
                if fabs(temp)>1e-15:
                    rho_coeffs[l,m,i] += temp

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cdef double compute_flux(double r,double phi,double pr,double pphi,double omega,double omega_circ,double H,EOBParams eob_pars):
    """
    Compute the full flux. See Eq(43) in the .
    """
    cdef int l,m
    cdef double[:,:] Tlm = eob_pars.flux_params.Tlm
    # Note the "abs" in the prefixes.
    cdef double[:,:] prefixes = eob_pars.flux_params.prefixes_abs
    cdef double v = omega**(1./3)
    cdef double vh3 = H*omega
    cdef double vh = vh3**(1./3)
    cdef double omega2 = omega*omega
    cdef double nu = eob_pars.p_params.nu
    cdef double flux = 0.0
    cdef double hNewton

    # Precompute the tail
    compute_tail(omega, H, Tlm)
    cdef double tail

    # Precompute the source term
    cdef double Slm = 0.0
    cdef double source1 = (H * H - 1.0) / (2.0 * nu) + 1.0 # H_eff
    cdef double source2 = v*pphi
    cdef double v_phi = omega/omega_circ**(2./3)
    cdef double v_phi2 = v_phi*v_phi

    cdef double complex rholmPwrl  = 1.0

    # Assume that the spin params have already been updated appropriately
    compute_rho_coeffs(eob_pars.p_params.nu,eob_pars.p_params.delta,eob_pars.p_params.a,eob_pars.p_params.chi_S,
    eob_pars.p_params.chi_A, eob_pars.flux_params.rho_coeffs,eob_pars.flux_params.rho_coeffs_log,
    eob_pars.flux_params.f_coeffs,eob_pars.flux_params.f_coeffs_vh, eob_pars.flux_params.extra_PN_terms)



    compute_rholm(v,vh,eob_pars.p_params.nu,eob_pars)
    cdef double complex hlm

    # Deal with NQCs
    cdef double[:] nqc_coeffs
    cdef double correction = 1.0

    for l in range(2,ell_max+1):
        for m in range(1,l+1):
            # Assemble the waveform

            hNewton = EOBFluxCalculateNewtonianMultipoleAbs(
                v_phi2, M_PI_2, l, m, prefixes
            )
            if ((l + m) % 2) == 0:
                Slm = source1

            else:
                Slm = source2

            tail = Tlm[l,m]
            rholmPwrl = eob_pars.flux_params.rholm[l,m]
            hlm = tail*Slm*hNewton*rholmPwrl

            flux += m*m*omega2*cabs(hlm)**2
    return flux/(8*pi)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef double compute_refactorized_flux(
    double chi_1,
    double chi_2,
    double x,
    double x_phi,
    double nu,
    double H,
    bint beaming,
    int alpha_N,
    int beta_PN_N,
    int beta_TP_N,
    bint quasi_circ,
):
    cdef double CES21 = 1.
    cdef double CES22 = 1.
    cdef double CES31 = 1.
    cdef double CES32 = 1.

    # cdef double x
    cdef double h_Newt_22
    cdef double T_22
    cdef double Heff
    cdef double a
    cdef double rho_22
    cdef double dE_dt_22
    cdef double B
    cdef double flux

    cdef double expr1
    cdef double expr2
    cdef double expr3

    cdef double tanh_pia
    cdef double tanh_2pia
    cdef double sinh_2pia

    cdef double alpha
    cdef double alpha52
    cdef double alpha3
    cdef double alpha72
    cdef double alpha4
    cdef double alpha92
    cdef double alpha5

    cdef double beta
    cdef double beta12
    cdef double beta1
    cdef double beta32
    cdef double beta2
    cdef double beta52
    cdef double beta3
    cdef double beta72
    cdef double beta4
    cdef double beta92
    cdef double beta5
    cdef double beta112

    if quasi_circ == 1:
        h_Newt_22 = 6.341323676169617 * x_phi
    else:
        h_Newt_22 = 6.341323676169617 * x

    T_22 = sqrt(
        6.283185307179586 * H * x**(3./2)
        * (1. + 16. * H**2. * x**3.) * (4. + 16. * H**2. * x**3.)
        / (1. - np.exp(- 25.132741228718345 * H * x**(3./2)))
    )

    Heff = (H * H - 1.) / (2. * nu) + 1.

    expr1 = sqrt(1. - 4. * nu)
    expr2 = chi_1 + chi_2

    a = (
        (1. - 2. * nu) * expr2 / 2.
        + expr1 * (chi_1 - chi_2) / 2.
    )

    expr3 = sqrt(1. - a**2)

    rho_22 = (
        1.
        + (-1.0238095238095237 + 0.6547619047619048 * nu) * x
        - 0.6666666666666666 * (
            0.5 * expr2 + 0.5 * (chi_1 - chi_2) * expr1 - 0.5 * expr2 * nu
        ) * x**(3./2)
        + (
            - 1.94208238851096
            + 0.5 * (
                0.25 * expr2**2
                + 0.5 * (chi_1 - chi_2) * expr2 * expr1
                + 0.25 * (chi_1 - chi_2)**2 * (1. - 4.*nu)
            )
            - 1.5601379440665155 * nu
            + 0.4625614134542706 * nu**2
        ) * x**2
        - 1.619047619047619 * a * x**(5./2)
        + (1.1799571680524061 * a + 0.3333333333333333 * a**3) * x**(7./2)
        + (
            12.736034731834051 + 0.3531746031746032 * a**2
            - 2.902228713904598 * nu
            - 1.9301558466099282 * nu**2
            + 0.2715020968103451 * nu**3
            - 4.076190476190476 * (1.9635100260214235 + 0.5 * log(x))
        ) * x**3
        + (
            -2.4172313935587004 + 0.8670162509448224 * a**2
            - 0.125 * a**4
            + 4.173242630385488 * (1.9635100260214235 + 0.5 * log(x))
        ) * x**4
        + (
            -30.14143102836864
            + 7.916297736025627 * (1.9635100260214235 + 0.5 * log(x))
        ) * x**5
    )

    dE_dt_22 = (
        nu**2. * x**3. * h_Newt_22**2. * Heff**2. * T_22**2. * rho_22**4.
        / 6.283185307179586
    )

    alpha52 = (
        -0.25 - 0.75 * a**2
        + expr3 * (-0.25 - 0.75 * a**2)
    )

    alpha3 = 0.

    if beaming == 1:
        alpha72 = (
            (-2.052827380952381 - 5.220982142857142 * a**2) * (1 + expr3)
        )
    else:
        alpha72 = (
            (-1.927827380952381 - 4.845982142857142 * a**2) * (1 + expr3)
        )

    tanh_2pia = tanh(6.283185307179586 * a / expr3)

    if a != 0:
        alpha4 = (
            3.141592653589793 - 0.5 / a
            + 2.645833333333333 * a - 0.8125 * a**3
            + 3.141592653589793 / tanh_2pia
            + (
                9.42477796076938 + 9.42477796076938 / tanh_2pia
            ) * a**2
            + expr3 * (
                3.141592653589793 - 0.5 / a
                + 2.145833333333333 * a - 2.3125 * a**3
                + 3.141592653589793 / tanh_2pia
                + a**2 * (
                    9.42477796076938
                    + 9.42477796076938 / tanh_2pia
                )
            )
        )
    else:
        alpha4 = 6.283185307179586

    if beaming == 1:
        alpha92 = (
            -15.47802525234631 - 37.38403855465798 * a**2
            + 1.2433035714285716 * a**4
        ) * (1 + expr3)
    else:
        alpha92 = (
            -14.52973656187012 - 35.00792248322941 * a**2
            + 1.2433035714285716 * a**4
        ) * (1 + expr3)

    if beaming == 1:
        if a != 0:
            tanh_pia = tanh(3.141592653589793 * a / expr3)

            alpha5 = (
                - 4.105654761904762 / a + 16.37735615079365 * a
                - 20.34709821428571 * a**3
                + 0.0001240079365079365 * (
                    224846.92781007508 + 104011.84957505087 / tanh_pia
                    + 91344.94799577682 * tanh_pia
                )
                + 0.0001240079365079365 * a**2 * (
                    579539.02158567 + 264534.66780287493 / tanh_pia
                    + 274034.84398733045 * tanh_pia
                )
                + expr3 * (
                    -4.105654761904762 / a + 12.271701388888888 * a
                    - 30.7890625 * a**3
                    + 0.0001240079365079365 * (
                        224846.92781007508 + 104011.84957505087 / tanh_pia
                        + 91344.94799577682 * tanh_pia
                    )
                    + 0.0001240079365079365 * a**2 * (
                        579539.02158567 + 264534.66780287493 / tanh_pia
                        + 274034.84398733045 * tanh_pia
                    )
                )
            )
        else:
            alpha5 = 55.7656070957528
    else:
        if a != 0:
            sinh_2pia = sinh(6.283185307179586 * a / expr3)

            alpha5 = (
                -3.855654761904762 / a + 15.387772817460316 * a
                - 18.94084821428571 * a**3
                + 0.0001240079365079365 * a**2 * (
                    541538.3168478478 + 500568.80705238326 / tanh_2pia
                    - 9500.176184455535 / sinh_2pia
                    )
                + 0.0001240079365079365 * (
                    212180.02623080104 + 182689.89599155364 / tanh_2pia
                    + 12666.901579274047 / sinh_2pia
                )
                + expr3 * (
                    - 3.855654761904762 / a + 11.532118055555555 * a
                    - 28.6328125 * a**3
                    + 0.0001240079365079365 * a**2 * (
                        541538.3168478478 + 500568.80705238326 / tanh_2pia
                        - 9500.176184455535 / sinh_2pia
                    )
                    + 0.0001240079365079365 * (
                        212180.02623080104 + 182689.89599155364 / tanh_2pia
                        + 12666.901579274047 / sinh_2pia
                    )
                )
            )
        else:
            alpha5 = 52.6240144421630

    #alpha = 1. + (a / (1. + expr3) - 2. * x**(3./2)) * (
    #    alpha52 * x**(5./2)
    #    + alpha3 * x**3.
    #    + alpha72 * x**(7./2)
    #    + alpha4 * x**4.
    #    + alpha92 * x**(9./2)
    #    + alpha5 * x**5.
    #)

    alpha = 1.

    beta12 = 0.

    if beaming == 1:
        beta1 = 0.47098214285714285 - 1.3839285714285714 * nu
    else:
        beta1 = 0.34598214285714285 - 1.3839285714285714 * nu

    beta32 = (
        - 0.010416666666666666 * (chi_1 + chi_2)
        - 0.010416666666666666 * (chi_1 - chi_2) * expr1
        + 0.041666666666666664 * (chi_1 + chi_2) * nu
    )

    if beaming == 1:
        beta2 = (
            -0.5486301490358875
            - 0.2421875 * chi_1**2 + 0.25 * CES21 * chi_1**2
            - 0.2421875 * chi_2**2 + 0.25 * CES22 * chi_2**2
            - 0.2421875 * chi_1**2 * expr1
            + 0.25 * CES21 * chi_1**2 * expr1
            + 0.2421875 * chi_2**2 * expr1
            - 0.25 * CES22 * chi_2**2 * expr1
            + (
                2.005527949026833
                + 0.484375 * chi_1**2 - 0.5 * CES21 * chi_1**2
                - 0.03125 * chi_1 * chi_2
                + 0.484375 * chi_2**2 - 0.5 * CES22 * chi_2**2
            ) * nu
            + 0.5488340301398337 * nu**2
        )
    else:
        beta2 = (
            -0.5528154168930304
            - 0.2421875 * chi_1**2 + 0.25 * CES21 * chi_1**2
            - 0.2421875 * chi_2**2 + 0.25 * CES22 * chi_2**2
            - 0.2421875 * chi_1**2 * expr1
            + 0.25 * CES21 * chi_1**2 * expr1
            + 0.2421875 * chi_2**2 * expr1
            - 0.25 * CES22 * chi_2**2 * expr1
            + (
                2.220185687122071
                + 0.484375 * chi_1**2 - 0.5 * CES21 * chi_1**2
                - 0.03125 * chi_1 * chi_2
                + 0.484375 * chi_2**2 - 0.5 * CES22 * chi_2**2
            ) * nu
            + 0.5488340301398338 * nu**2
        )

    if beaming == 1:
        beta52 = (
            2.086213871524472
            - 0.09081256200396826 * chi_1 - 0.09081256200396826 * chi_2
            + (-0.09081256200396826 * chi_1 + 0.09081256200396826 * chi_2) * expr1
            + (
                - 8.344855486097888
                + 1.1473834325396826 * chi_1 + 1.1473834325396826 * chi_2
                + (1.2275855654761905 * chi_1 - 1.2275855654761905 * chi_2) * expr1
            ) * nu
            + (-0.9262152777777778 * chi_1 - 0.9262152777777778 * chi_2) * nu**2
        )
    else:
        beta52 = (
            2.086213871524472
            - 0.25617714533730157 * chi_1 - 0.25617714533730157 * chi_2
            + (-0.25617714533730157 * chi_1 + 0.25617714533730157 * chi_2) * expr1
            + (
                -8.344855486097888
                + 1.2255084325396826 * chi_1 + 1.2255084325396826 * chi_2
                + (1.2275855654761905 * chi_1 - 1.2275855654761905 * chi_2) * expr1
            ) * nu
            + (-0.9262152777777777 * chi_1 - 0.9262152777777777 * chi_2) * nu**2
        )

    if beaming == 1:
        beta3 = (
            - 2.128113388455193
            + 0.06544984694978735 * chi_1
            - 0.7453671409970238 * chi_1**2
            + 0.8759300595238095 * CES21 * chi_1**2
            + 0.06544984694978735 * chi_2
            - 0.7453671409970238 * chi_2**2
            + 0.8759300595238095 * CES22 * chi_2**2
            + (
                0.06544984694978735 * chi_1
                - 0.7453671409970238 * chi_1**2
                + 0.8759300595238095 * CES21 * chi_1**2
                - 0.06544984694978735 * chi_2
                + 0.7453671409970238 * chi_2**2
                - 0.8759300595238095 * CES22 * chi_2**2
            ) * expr1
            + (
                6.7228991308382025
                - 0.2617993877991494 * chi_1
                + 1.9449831039186507 * chi_1**2
                - 3.3530505952380953 * CES21 * chi_1**2
                - 0.2617993877991494 * chi_2
                - 0.9156048487103174 * chi_1 * chi_2
                + 1.9449831039186507 * chi_2**2
                - 3.3530505952380953 * CES22 * chi_2**2
                + (
                    0.4542488219246032 * chi_1**2
                    - 1.6011904761904763 * CES21 * chi_1**2
                    - 0.4542488219246032 * chi_2**2
                    + 1.6011904761904763 * CES22 * chi_2**2
                ) * expr1
            ) * nu
            + (
                - 1.4791500204153913
                - 1.020600818452381 * chi_1**2
                + 1.9211309523809523 * CES21 * chi_1**2
                - 1.8060825892857142 * chi_1 * chi_2
                - 1.020600818452381 * chi_2**2
                + 1.9211309523809523 * CES22 * chi_2**2
            ) * nu**2
            + 0.029224860174085567 * nu**3
        )
    else:
        beta3 = (
            - 1.5673064297215402
            + 0.06544984694978737 * chi_1
            - 0.7150937034970238 * chi_1**2
            + 0.9071800595238094 * CES21 * chi_1**2
            + 0.06544984694978737 * chi_2
            - 0.7150937034970238 * chi_2**2
            + 0.9071800595238094 * CES22 * chi_2**2
            + (
                0.06544984694978735 * chi_1
                - 0.7150937034970237 * chi_1**2
                + 0.9071800595238094 * CES21 * chi_1**2
                - 0.06544984694978735 * chi_2
                + 0.7150937034970237 * chi_2**2
                - 0.9071800595238094 * CES22 * chi_2**2
            ) * expr1
            + (
                6.0984404660788964
                - 0.26179938779914946 * chi_1
                + 1.884436228918651 * chi_1**2
                - 3.415550595238095 * CES21 * chi_1**2
                - 0.26179938779914946 * chi_2
                - 0.6616985987103176 * chi_1 * chi_2
                + 1.884436228918651 * chi_2**2
                - 3.415550595238095 * CES22 * chi_2**2
                + (
                    0.45424882192460314 * chi_1**2
                    - 1.601190476190476 * CES21 * chi_1**2
                    - 0.45424882192460314 * chi_2**2
                    + 1.601190476190476 * CES22 * chi_2**2
                ) * expr1
            ) * nu
            + (
                - 1.6054179646590612
                - 1.020600818452381 * chi_1**2
                + 1.9211309523809523 * CES21 * chi_1**2
                - 1.8060825892857144 * chi_1 * chi_2
                - 1.020600818452381 * chi_2**2
                + 1.9211309523809523 * CES22 * chi_2**2
            ) * nu**2
            + 0.02922486017408557 * nu**3
        )

    if beaming == 1:
        beta72 = (
            - 1.169026602968159
            + 0.06236510029705724 * chi_1 - 0.04908738521234052 * chi_1**2
            - 0.3328450520833333 * chi_1**3 + 0.8385416666666666 * CES21 * chi_1**3
            - 0.5 * CES31 * chi_1**3 + 0.06236510029705724 * chi_2
            - 0.04908738521234052 * chi_2**2
            - 0.3328450520833333 * chi_2**3 + 0.8385416666666666 * CES22 * chi_2**3
            - 0.5 * CES32 * chi_2**3
            + (
                0.06236510029705724 * chi_1 - 0.04908738521234052 * chi_1**2
                - 0.3328450520833333 * chi_1**3 + 0.8385416666666666 * CES21 * chi_1**3
                - 0.5 * CES31 * chi_1**3 - 0.06236510029705724 * chi_2
                + 0.04908738521234052 * chi_2**2 + 0.3328450520833333 * chi_2**3
                - 0.8385416666666666 * CES22 * chi_2**3 + 0.5 * CES32 * chi_2**3
            ) * expr1
            + (
                1.1044499346767318 + 10.925968357585004 * chi_1
                + 0.09817477042468103 * chi_1**2 + 1.5704752604166667 * chi_1**3
                - 2.3177083333333335 * CES21 * chi_1**3 + 0.020833333333333332 * CES22 * chi_1**3
                + 1.5 * CES31 * chi_1**3 + 10.925968357585004 * chi_2
                + 0.19634954084936207 * chi_1 * chi_2 + 1.24853515625 * chi_1**2 * chi_2
                - 0.5052083333333334 * CES21 * chi_1**2 * chi_2
                + 0.09817477042468103 * chi_2**2 + 1.24853515625 * chi_1 * chi_2**2
                - 0.5052083333333334 * CES22 * chi_1 * chi_2**2 + 1.5704752604166667 * chi_2**3
                + 0.020833333333333332 * CES21 * chi_2**3 - 2.3177083333333335 * CES22 * chi_2**3
                + 1.5 * CES32 * chi_2**3
                + (
                    8.134637611102745 * chi_1 + 0.90478515625 * chi_1**3
                    - 0.640625 * CES21 * chi_1**3 + 0.020833333333333332 * CES22 * chi_1**3
                    + 0.5 * CES31 * chi_1**3 - 8.134637611102745 * chi_2
                    + 1.24853515625 * chi_1**2 * chi_2 - 0.5052083333333334 * CES21 * chi_1**2 * chi_2
                    - 1.24853515625 * chi_1 * chi_2**2 + 0.5052083333333334 * CES22 * chi_1 * chi_2**2
                    - 0.90478515625 * chi_2**3 - 0.020833333333333332 * CES21 * chi_2**3
                    + 0.640625 * CES22 * chi_2**3 - 0.5 * CES32 * chi_2**3
                ) * expr1
            ) * nu
            + (
                21.378757472443116 - 8.71349019660671 * chi_1
                - 1.6438802083333333 * chi_1**3 - 0.4583333333333333 * CES21 * chi_1**3
                - 0.10416666666666667 * CES22 * chi_1**3 - 8.71349019660671 * chi_2
                + 0.3600260416666667 * chi_1**2 * chi_2 - 0.4166666666666667 * CES21 * chi_1**2 * chi_2
                - 0.0625 * CES22 * chi_1**2 * chi_2 + 0.3600260416666667 * chi_1 * chi_2**2
                - 0.0625 * CES21 * chi_1 * chi_2**2 - 0.4166666666666667 * CES22 * chi_1 * chi_2**2
                - 1.6438802083333333 * chi_2**3 - 0.10416666666666667 * CES21 * chi_2**3
                - 0.4583333333333333 * CES22 * chi_2**3
                + (
                    - 4.077088632581216 * chi_1 - 0.5 * chi_1**3
                    - 0.0625 * CES21 * chi_1**3 - 0.0625 * CES22 * chi_1**3
                    + 4.077088632581216 * chi_2 - 0.5 * chi_1**2 * chi_2
                    - 0.0625 * CES21 * chi_1**2 * chi_2 - 0.0625 * CES22 * chi_1**2 * chi_2
                    + 0.5 * chi_1 * chi_2**2 + 0.0625 * CES21 * chi_1 * chi_2**2
                    + 0.0625 * CES22 * chi_1 * chi_2**2 + 0.5 * chi_2**3
                    + 0.0625 * CES21 * chi_2**3 + 0.0625 * CES22 * chi_2**3
                ) * expr1
            ) * nu**2
            + (
                1.8834442884503337 * chi_1 + 0.3333333333333333 * chi_1**3
                + 0.08333333333333333 * CES21 * chi_1**3 + 0.08333333333333333 * CES22 * chi_1**3
                + 1.8834442884503337 * chi_2 + chi_1**2 * chi_2
                + 0.25 * CES21 * chi_1**2 * chi_2 + 0.25 * CES22 * chi_1**2 * chi_2
                + chi_1 * chi_2**2 + 0.25 * CES21 * chi_1 * chi_2**2
                + 0.25 * CES22 * chi_1 * chi_2**2 + 0.3333333333333333 * chi_2**3
                + 0.08333333333333333 * CES21 * chi_2**3 + 0.08333333333333333 * CES22 * chi_2**3
            ) * nu**3
        )
    else:
        beta72 = (
            - 1.429803336908718
            - 0.6095166813869705 * chi_1 - 0.04908738521234052 * chi_1**2
            + (-0.3328450520833333 + 0.8385416666666666 * CES21 - 0.49999999999999994 * CES31) * chi_1**3
            - 0.6095166813869705 * chi_2 - 0.04908738521234052 * chi_2**2
            + (-0.3328450520833333 + 0.8385416666666666 * CES22 - 0.49999999999999994 * CES32) * chi_2**3
            + (
                -0.6095166813869705 * chi_1 - 0.04908738521234052 * chi_1**2
                + (-0.3328450520833333 + 0.8385416666666666 * CES21 - 0.49999999999999994 * CES31) * chi_1**3
                + 0.6095166813869705 * chi_2 + 0.04908738521234052 * chi_2**2
                + (0.3328450520833333 - 0.8385416666666666 * CES22 + 0.49999999999999994 * CES32) * chi_2**3
            ) * expr1
            + (
                2.1475568704389674
                + (1.5704752604166665 - 2.317708333333333 * CES21 + 0.020833333333333332 * CES22 + 1.5 * CES31) * chi_1**3
                + 12.029987764827068 * chi_2 + 0.09817477042468103 * chi_2**2
                + (1.5704752604166665 + 0.020833333333333332 * CES21 - 2.317708333333333 * CES22 + 1.5 * CES32) * chi_2**3
                + chi_1**2 * (0.09817477042468103 + (1.24853515625 - 0.5052083333333333 * CES21) * chi_2)
                + chi_1 * (12.029987764827068 + 0.19634954084936207 * chi_2 + (1.24853515625 - 0.5052083333333333 * CES22) * chi_2**2)
                + (
                    (0.9047851562499999 - 0.640625 * CES21 + 0.020833333333333332 * CES22 + 0.49999999999999994 * CES31) * chi_1**3
                    - 8.510021260656316 * chi_2 + (1.24853515625 - 0.5052083333333333 * CES21) * chi_1**2 * chi_2
                    + (-0.9047851562499999 - 0.020833333333333332 * CES21 + 0.640625 * CES22 - 0.49999999999999994 * CES32) * chi_2**3
                    + chi_1 * (8.510021260656316 + (-1.24853515625 + 0.5052083333333333 * CES22) * chi_2**2)
                ) * expr1
            ) * nu +
            (
                21.378757472443112
                + (-1.6438802083333333 - 0.4583333333333333 * CES21 - 0.10416666666666666 * CES22) * chi_1**3
                - 8.766860112281313 * chi_2
                + (0.36002604166666663 - 0.41666666666666663 * CES21 - 0.0625 * CES22) * chi_1**2 * chi_2
                + (-1.6438802083333333 - 0.10416666666666666 * CES21 - 0.4583333333333333 * CES22) * chi_2**3
                + chi_1 * (-8.766860112281313 + (0.36002604166666663 - 0.0625 * CES21 - 0.41666666666666663 * CES22) * chi_2**2)
                + (
                    (-0.49999999999999994 - 0.0625 * CES21 - 0.0625 * CES22) * chi_1**3
                    + 4.077088632581215 * chi_2
                    + (-0.49999999999999994 - 0.0625 * CES21 - 0.0625 * CES22) * chi_1**2 * chi_2
                    + (0.49999999999999994 + 0.0625 * CES21 + 0.0625 * CES22) * chi_2**3
                    + chi_1 * (-4.077088632581215 + (0.49999999999999994 + 0.0625 * CES21 + 0.0625 * CES22) * chi_2**2)
                ) * expr1
            )* nu**2
            + (
                (0.3333333333333333 + 0.08333333333333333 * CES21 + 0.08333333333333333 * CES22) * chi_1**3
                + 1.8834442884503337 * chi_2
                + (0.9999999999999999 + 0.25 * CES21 + 0.25 * CES22) * chi_1**2 * chi_2
                + (0.3333333333333333 + 0.08333333333333333 * CES21 + 0.08333333333333333 * CES22) * chi_2**3
                + chi_1 * (1.8834442884503337 + (0.9999999999999999 + 0.25 * CES21 + 0.25 * CES22) * chi_2**2)
            ) * nu**3
        )

    if beaming == 1:
        beta4 = (
            8.072488677403342
            - 2.3141439374831623 * a
            - 0.18213016868508652 * a**2
            - 0.0159912109375 * a**4
            - 0.9711167800453515 * log(x)
        )
    else:
        beta4 = (
            10.114228370386105
            - 2.330506399220609 * a
            + 0.38141469506119324 * a**2
            - 0.0159912109375 * a**4
            - 0.9711167800453515 * log(x)
        )

    if beaming == 1:
        beta92 = (
            - 18.29435444994186
            + 4.546626318517944 * a
            + 1.7402829421802135 * a**2
            + 0.6351334237127287 * a**3
            + 3.898015873015873 * a * log(x)
        )
    else:
        beta92 = (
            -18.034136303471843
            + 1.7525547884832986 * a**2
            + 0.6258967700668954 * a**3
            + a * (0.16942839332227244 + 3.898015873015873 * log(x))
        )

    if beaming == 1:
        beta5 = (
            19.820637776303766
            - 9.921907618552808 * a
            - 1.025103686888767 * a**2
            - 0.07772169325287248 * a**3
            - 0.15794956873333643 * a**4
            + 0.4738176563005019 * log(x)
            - 0.9235119047619048 * a**2 * log(x)
        )
    else:
        beta5 = (
            23.820736971461926
            - 10.32088567319877 * a
            + 1.71620670347648 * a**2
            - 0.07772169325287248 * a**3
            - 0.15399754236614893 * a**4
            + (0.5952072538061708 - 0.9235119047619048 * a**2) * log(x)
        )

    if beaming == 1:
        beta112 = (
            1.8072480800519273
            - 20.122387510504154 * a
            + 5.501838571193589 * a**2
            + 2.3338507859384703 * a**3
            + 0.10277671278833796 * a**4
            - 0.28276824951171875 * a**5
            - 6.637972453723382 * log(x)
            + 7.909000909391534 * a * log(x)
        )
    else:
        beta112 = (
            5.003270253676121
            - 33.70682640102828 * a
            + 5.496077773304163 * a**2
            + 1.7064747240397795 * a**3
            + 0.10277671278833796 * a**4
            - 0.28276824951171875 * a**5
            + (-6.637972453723382 + 7.42174892526455 * a) * log(x)
        )

    beta = 1. + (
        beta12 * x**(1./2)
        + beta1 * x
        + beta32 * x**(3./2)
        + beta2 * x**(2)
        + beta52 * x**(5./2)
        + beta3 * x**3.
        + beta72 * x**(7./2)
        + beta4 * x**4.
        + beta92 * x**(9./2)
        + beta5 * x**5.
        + beta112 * x**(11./2)
    )

    if beaming == 1:
        B = sqrt(2) * (asinh(Heff) - asinh(1.)) + 1.
    else:
        B = 1.

    flux = dE_dt_22 * alpha * beta**4. * B

    return flux

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef  (double,double) RR_force(double[::1] q,double[::1] p,double omega,double omega_circ,double H,EOBParams eob_par):
    """
    Compute the RR force in polar coordinates, from the flux
    """
    cdef double r = q[0]
    cdef double phi = q[1]
    cdef double pr = p[0]
    cdef double pphi = p[1]
    cdef double nu = eob_par.p_params.nu
    # Note the multiplication of H by nu!
    cdef double flux = compute_flux(r,phi,pr,pphi,omega,omega_circ,nu*H,eob_par)
    flux /= nu
    cdef double f_over_om = flux/omega
    cdef double Fr = -pr / pphi * f_over_om
    cdef double Fphi = -f_over_om
    return Fr, Fphi



cdef class RadiationReactionForce:
    """
    Convenience wrappers around the RR_force function to enable typed calls
    """
    def __cinit__(self):
        pass
    cpdef (double,double) RR(self, double[::1] q,double[::1] p,double omega,double omega_circ,double H,EOBParams eob_par):
        pass


cdef class SEOBNRv5RRForce(RadiationReactionForce):
    """
    Convenience wrappers around the RR_force function to enable typed calls
    """
    def __cinit__(self):
        pass
    cpdef (double,double) RR(self, double[::1] q,double[::1] p,double omega,double omega_circ,double H,EOBParams eob_par):
        return RR_force(q,p,omega,omega_circ,H,eob_par)





@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.profile(True)
@cython.linetrace(True)
cdef double complex EOBFluxCalculateNewtonianMultipole(
    double x, double phi, int l, int m, double complex[:,:] params
):
    """
    Compute the Newtonian multipole
    """
    cdef double complex param = params[l,m]
    cdef int epsilon = (l + m) % 2


    cdef double complex y = 0.0
    # Calculate the necessary Ylm
    cdef double complex ylm = sph_harm(-m, l - epsilon, phi, pi / 2)


    cdef double complex multipole = param * x ** ((l + epsilon) / 2.0)
    multipole *= ylm
    return multipole


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cdef double complex compute_mode(double v_phi2,double phi, double Slm, double[] vs,double[] vhs,int l, int m, EOBParams eob_pars):
    """
    Compute the given (l,m) mode at one instant in time. See Eq(24) in https://dcc.ligo.org/LIGO-T2300060 (SEOBNRv5HM.pdf).
    """

    cdef double complex Tlm,hNewton,rholm,deltalm
    cdef double v = vs[1]
    cdef double vh = vhs[1]

    # Compute the tail
    cdef double k = m*v*v*v # this is just m*Omega
    cdef double hathatk = m*vh*vh*vh # this is just m*Omega*H
    # This is just l!
    cdef double z2 = tgamma(l+1)
    cdef double complex lnr1 = loggamma(l+1.0-2.0*hathatk*I)

    cdef double lnr1_abs = cabs(lnr1)
    cdef double lnr1_arg = carg(lnr1)

    Tlm = cexp((pi * hathatk) +
	  I * (2.0 * hathatk * log(4.0 * k / sqrt(M_E)))) * cexp(lnr1)
    Tlm /= z2

    # Calculate the newtonian multipole, 1st term in Eq. 17, given by Eq. A1
    hNewton = EOBFluxCalculateNewtonianMultipole(
        v_phi2, phi, l, m, eob_pars.flux_params.prefixes
    )

    # Compute rho^l
    rholm = compute_rholm_single(vs,vh,l,m,eob_pars)
    # Compute delta
    deltalm = compute_deltalm_single(vs,vhs,l,m,eob_pars.flux_params)
    # Put everything together
    cdef double complex hlm = Tlm * cexp (I * deltalm) * Slm * rholm
    hlm *= hNewton
    return hlm



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef compute_hlms(double[:,:] dynamics, EOBParams eob_pars):
    """
    Compute the inspiral modes for aligned *or* precessing binaries.
    Note that:
    - it is assumed that the special mode coefficients have already been
    computed and set correctly in eob_pars.flux_params
    - it is assumed that if the case is precessing, dynamics has been correctly
    augmented with the projections of the spins

    """

    cdef double r,pr,pphi,omega_circ,omega,v_phi,v_phi2,H,v,vh,phi,chi_1,chi_2
    cdef int l,m,i,j
    cdef double[:] row

    cdef double Slm = 0.0
    cdef double source1
    cdef double source2
    cdef (int,int) ell_m
    cdef double nu = eob_pars.p_params.nu
    cdef double complex tmp

    cdef double vs[PN_limit]
    cdef double vhs[PN_limit]
    cdef double complex rho
    compute_rho_coeffs(eob_pars.p_params.nu,eob_pars.p_params.delta,eob_pars.p_params.a,eob_pars.p_params.chi_S,
                eob_pars.p_params.chi_A, eob_pars.flux_params.rho_coeffs,eob_pars.flux_params.rho_coeffs_log,
                eob_pars.flux_params.f_coeffs,eob_pars.flux_params.f_coeffs_vh, eob_pars.flux_params.extra_PN_terms)

    compute_delta_coeffs(eob_pars.p_params.nu,eob_pars.p_params.delta,eob_pars.p_params.a,eob_pars.p_params.chi_S,
        eob_pars.p_params.chi_A, eob_pars.flux_params.delta_coeffs,eob_pars.flux_params.delta_coeffs_vh)

    cdef int N = dynamics.shape[0]
    cdef dict modes = {(ell_m[0],ell_m[1]):np.zeros(N,dtype=np.complex128) for ell_m in eob_pars.mode_array}
    cdef double complex[:,:,:] temp_modes = np.zeros((ell_max,ell_max,N),dtype=np.complex128)
    # Compute all the modes
    for i in range(N):
        row = dynamics[i]
        r = row[0]
        phi = row[1]
        pr = row[2]
        pphi = row[3]
        H = nu*row[4]
        omega = row[5]
        omega_circ = row[6]
        v = omega**(1./3)
        vh = (H*omega)**(1./3)
        # Various powers of v that enter the computation
        # of rholm and deltalm
        for j in range(PN_limit):
            vs[j] = v**j
            vhs[j] = vh**j

        v_phi = omega/omega_circ**(2./3)
        v_phi2 = v_phi*v_phi
        source1 = (H * H - 1.0) / (2.0 * nu) + 1.0 # H_eff
        source2 = v*pphi
        if not eob_pars.aligned:
            # For precession, we have to recompute the coefficients that enter
            # the waveform at every step, since the spins change.
            # Note that we only need to recompute rho and delta coefficients
            # since the Newtonian prefixes are independent of the spins.
            # Assume the dynamics has been augmented appropriately with the spins.
            chi_1 = row[7]
            chi_2 = row[8]
            eob_pars.p_params.update_spins(chi_1,chi_2)
            # Recompute rho coeffs
            compute_rho_coeffs(eob_pars.p_params.nu,eob_pars.p_params.delta,eob_pars.p_params.a,eob_pars.p_params.chi_S,
                eob_pars.p_params.chi_A, eob_pars.flux_params.rho_coeffs,eob_pars.flux_params.rho_coeffs_log,
                eob_pars.flux_params.f_coeffs,eob_pars.flux_params.f_coeffs_vh,eob_pars.flux_params.extra_PN_terms)
            # Recompute delta coeffs
            compute_delta_coeffs(eob_pars.p_params.nu,eob_pars.p_params.delta,eob_pars.p_params.a,eob_pars.p_params.chi_S,
                eob_pars.p_params.chi_A, eob_pars.flux_params.delta_coeffs,eob_pars.flux_params.delta_coeffs_vh)

        for j in range(len(eob_pars.mode_array)):
            ell_m = eob_pars.mode_array[j]
            l = ell_m[0]
            m = ell_m[1]
            if ((l + m) % 2) == 0:
                Slm = source1
            else:
                Slm = source2


            #modes[l,m][i]  = compute_mode(v_phi2,phi, Slm, vs,vhs,l, m, eob_pars)
            temp_modes[l,m,i] = compute_mode(v_phi2,phi, Slm, vs,vhs,l, m, eob_pars)

    for j in range(len(eob_pars.mode_array)):
            ell_m = eob_pars.mode_array[j]
            l = ell_m[0]
            m = ell_m[1]
            modes[l,m] = temp_modes[l,m]
    return modes


cdef double min_threshold(int l,int m):
    """
    The values for (2,1) and (5,5) are taken from SEOBNRv4HM. The
    value for (4,3) is determined empirically
    """
    if l==2 and m==1:
        return 90
    elif l==4 and m==3:
        return 100
    elif l==5 and m==5:
        return 2000


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef compute_special_coeffs(double[:,:] dynamics, double t_attach, EOBParams eob_pars,
    dict amp_fits, dict amp_thresholds, dict modes={(2,1):7,(4,3):7,(5,5):5}):
    """
    Compute the "special" amplitude coefficients. See discussion after Eq. (33) in https://dcc.ligo.org/LIGO-T2300060 (SEOBNRv5HM.pdf).
    """
    # Step 0: spline the dynamics to the attachment point


    cdef int i,j,l,m,power
    cdef double phi,pphi,omega,omega_circ,H,v,vh,vphi,vphi2,source1,source2,Slm
    cdef double rholm,hlm,K,amp,clm,min_amp
    cdef double vs[PN_limit]
    cdef double vhs[PN_limit]

    cdef np.ndarray[DTYPE_T,ndim=1] dynamics_55 = np.zeros(dynamics.shape[1]-1)
    cdef np.ndarray[DTYPE_T,ndim=1]  dynamics_all = np.zeros(dynamics.shape[1]-1)


    for i in range(1,dynamics.shape[1]):
        spline = CubicSpline(dynamics[:,0],dynamics[:,i])
        dynamics_all[i-1] = spline(t_attach)
        dynamics_55[i-1] = spline(t_attach-10)

    cdef double m_1 = eob_pars.p_params.m_1
    cdef double m_2 = eob_pars.p_params.m_2
    cdef double chi_1 = eob_pars.p_params.chi_1
    cdef double chi_2 = eob_pars.p_params.chi_2

    compute_rho_coeffs(eob_pars.p_params.nu,eob_pars.p_params.delta,eob_pars.p_params.a,eob_pars.p_params.chi_S,
                eob_pars.p_params.chi_A, eob_pars.flux_params.rho_coeffs,eob_pars.flux_params.rho_coeffs_log,
                eob_pars.flux_params.f_coeffs,eob_pars.flux_params.f_coeffs_vh, eob_pars.flux_params.extra_PN_terms)

    compute_delta_coeffs(eob_pars.p_params.nu,eob_pars.p_params.delta,eob_pars.p_params.a,eob_pars.p_params.chi_S,
        eob_pars.p_params.chi_A, eob_pars.flux_params.delta_coeffs,eob_pars.flux_params.delta_coeffs_vh)
    # Now, loop over every mode of interest
    for mode in modes.keys():
        l,m = mode
        if l==5 and m==5:
            dynamics_interp = 1*dynamics_55
        else:
            dynamics_interp = 1*dynamics_all

        phi = dynamics_interp[1]
        pphi = dynamics_interp[3]
        omega_circ = dynamics_interp[6]
        omega = dynamics_interp[5]
        H = eob_pars.p_params.nu*dynamics_interp[4]
        v = omega**(1./3)
        vh = (H*omega)**(1./3)
        vphi = omega/omega_circ**(2./3)
        vphi2 = vphi*vphi

        for j in range(PN_limit):
            vs[j] = v**j
            vhs[j] = vh**j
        source1 = (H * H - 1.0) / (2.0 * eob_pars.p_params.nu) + 1.0 # H_eff
        source2 = v*pphi

        if ((l + m) % 2) == 0:
            Slm = source1
        else:
            Slm = source2
        # Step 1: compute the mode with the calibration coeff set to 0
        power = modes[mode]

        eob_pars.flux_params.f_coeffs[l,m,power] = 0.0
        hlm = cabs(compute_mode(vphi2,phi, Slm, vs,vhs,l, m,eob_pars))
        # Step 2: compute rho_lm for this mode
        rholm = creal(compute_rholm_single(vs,vh,l,m,eob_pars))

        # Step 3: compute K = |h_{lm}|/|rho_{lm}|
        K = hlm/fabs(rholm)

        # Step 4: compute |h_{lm}^{NR}| from fit

        amp =  amp_fits[(l, m)]
        amp22 = amp_fits[(2,2)]

        # when the amplitude at merger is too small a positive sign is better
        if np.abs(amp)<1e-4:
            amp = np.abs(amp)

        min_amp = amp_thresholds[(l,m)]
        if np.abs(amp)<amp22/min_amp:
            amp = np.sign(amp)*amp22/min_amp


        amp*=eob_pars.p_params.nu


        # Step 5: compute c_lm = (|h_{lm}^{NR}|/K - rho_{lm}(clm=0))/v**power
        clm1 = (amp/K - rholm)*1/v**power
        clm2 = (-amp/K - rholm)*1/v**power

        # We always pick the positive solution
        eob_pars.flux_params.f_coeffs[l,m,power] = clm1



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef compute_factors(double[::1] phi_orb,int m_max, double complex[:,:]result):
    """
    Trivial helper function that computes iterative e^{im\phi} which is used in interpolation, see `interpolate_modes_fast` in `compute_hlms.py`
    """
    cdef int N = phi_orb.shape[0]
    cdef int i,m
    cdef double complex factor
    for i in range(N):
        factor = cexp(-1*I*phi_orb[i])
        result[0][i]  = factor
        for m in range(1,m_max):
                result[m][i]=result[m-1][i]*factor



@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.profile(True)
@cython.linetrace(True)
cpdef unrotate_leading_pn(double[::1]re_part,double[::1]im_part, double complex[:] factor,double complex[:] result):
    """
    Helper function used to multiply the interpolated re and im parts by the leading order PN scaling
    """
    cdef int N = re_part.shape[0]
    cdef int i
    #print(re_part.shape[0],im_part.shape[0],phi.shape[0],result.shape[0])
    for i in range(N):
        result[i] = (re_part[i]+I*im_part[i])*factor[i]