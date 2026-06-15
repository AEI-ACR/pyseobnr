
# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
# cython: profile=False, linetrace=False, binding=True, cpow=True

import numpy as np
cimport numpy as np

from libc.math cimport log, sqrt

from ..utils.containers cimport qp_param_t
from .Hamiltonian_C cimport (
  Hamiltonian_C,
  Hamiltonian_C_call_return_t,
  Hamiltonian_C_grad_return_t,
  Hamiltonian_C_dynamics_return_t,
  Hamiltonian_C_auxderivs_return_t
)

from pyseobnr.eob.utils.containers cimport EOBParams
from pyseobnr.eob.hamiltonian.Hamiltonian_C cimport Hamiltonian_C

from pyseobnr.eob.utils.utils_tidal_opt_C cimport (
  tidal_contribution,
  tidal_and_d_tidal_contribution,
  tidal_and_d_tidal_and_d2_tidal_contribution
)

cpdef (double, double) evaluate_H(
  double r,
  double prst,
  double pphi,
  double X_1,
  double X_2,
  double chi_1,
  double chi_2,
  double nu,
  EOBParams EOBpars,
  double Tidal1,
  double Tidal2,
  double CES21,
  double CES22,
  double CBS31,
  double CBS32,
  double CES41,
  double CES42
):
    """
    Evaluate the Hamiltonian and xi
    """
    # Non-spinning calibration coefficient
    cdef double a6 = EOBpars.c_coeffs.a6
    # Spin-orbit calibration coefficient
    cdef double dSO = EOBpars.c_coeffs.dSO

    cdef double d5 = 0
    cdef double x49 = 7680.0*nu
    cdef double x48 = 315.0*d5
    cdef double x33 = log(r)
    cdef double x47 = 49152.0*x33
    cdef double x7 = nu**2
    cdef double x46 = 102574080.0*x7 - 2119671837.36038
    cdef double x44 = 14700.0*nu + 42911.0
    cdef double x45 = 5760.0*x44
    cdef double x37 = x33**2
    cdef double x43 = 5787938193408.0*x37
    cdef double x42 = 22295347200.0*d5
    cdef double x41 = nu*r
    cdef double x38 = nu**3
    cdef double x8 = r**3
    cdef double x4 = r**2
    cdef double Dbpm = (
        r*(
            -193226342400.0*d5*nu
            + 2589101062873.81*nu*x4
            - 12049908701745.2*nu
            - 326837426.241486*r*x44
            + 1822680546449.21*r*x7
            - 39476764256925.6*r
            + 6730497718123.02*x38
            + 133772083200.0*x4*x7
            + 5107745331375.71*x4
            + x41*x42
            + 10611661054566.2*x41
            + x42*x7
            + x43
            - x47*(-516620178.136075*nu - r*x45 - 17902080.0*x4 - x46)
            + 80059249540278.2*x7
            + 275059053208689.0
        )/(
            55296.0*nu*(
                4331361844.61149*nu
                + 14515200.0*x38
                - x49*(x48 + 890888.810272497)
                - 42636451.6032331*x7
                + 1002013764.01019
            )
            + r*x43
            + r*(
                -4718592.0*nu*(40950.0*d5 + 86207832.4415642)
                + 450172889755120.0*nu
                + 5927865218923.02*x38
                + 70778880.0*x7*(x48 + 2561145.80918574)
                - 138141470005001.0*x7
                + 86618264430493.3*(1 - 0.496948781616935*nu)**2
                + 188440788778196.0
            )
            - 9216.0*x4*(
                2481453539.84635*nu - x49*(x48 + 405152.309729121) - 197773496.793534*x7 + 5805304367.87913
            )
            + x47*(
                -34560.0*nu*(11592.0*nu + 69847.0) + r*(409207698.136075*nu + x46) + x4*x45 + 17902080.0*x8
            )
            - 967680.0*x8*(-2675575.66847905*nu - 138240.0*x7 - 5278341.3229329)
        )
    )
    cdef double x40 = 8.0*r + 4.0*x4 + 2.0*x8 + 16.0
    cdef double x34 = 756.0*nu
    cdef double x39 = x34 + 1079.0
    cdef double x36 = r**5
    cdef double x35 = nu**4
    cdef double x11 = r**4
    cdef double Apm = (
        Tidal1
        + Tidal2
        + 7680.0*x11*(
            2048.0*nu*x33*(336.0*r + x34 + 407.0)
            + 28.0*nu*(1920.0*a6 + 733955.307463037)
            - 7.0*r*(938918.400156317*nu - 185763.092693281*x7 - 245760.0)
            - 5416406.59541186*x7
            - 3440640.0
        )/(
            32768.0*nu*x33*(
                -38842241.4769507*nu
                + 240.0*r*(-7466.27061066206*nu - 3024.0*x7 + 17264.0)
                + 480.0*x11*x39
                + 161280.0*x36
                + 960.0*x39*x8
                + 1920.0*x4*(588.0*nu + 1079.0)
                - 1882456.23663972*x7
                + 13447680.0
            )
            + 53760.0*nu*(
                7680.0*a6*(x11 + x40)
                + 113485.217444961*r*(-x11 + x40)
                + 148.04406601634*r*(7704.0*r + 349.0*x11 + 3852.0*x4 + 1926.0*x8 + 36400.0)
                + 128.0*r*(
                    13218.7851094412*r
                    - 6852.34813868015*x11
                    + 8529.39255472061*x4
                    + 4264.6962773603*x8
                    - 33722.4297811176
                )
            )
            + 241555486248.807*x35
            + 13212057600.0*x36
            + 67645734912.0*x37*x7
            + 1120.0*x38*(-163683964.822551*r - 17833256.898555*x4 - 1188987459.03162)
            + 7.0*x7*(
                -39321600.0*a6*(3.0*r + 59.0)
                + 745857848.115604*a6
                + 122635399361.987*r
                - 3089250703.76879*x11
                + 1426660551.8844*x36
                + 2064783811.32587*x4
                - 6178501407.53758*x8
                + 276057889687.011
            )
        )
    )
    cdef double x1 = X_2*chi_2
    cdef double x0 = X_1*chi_1
    cdef double ap = x0 + x1
    cdef double ap2 = ap**2
    cdef double x56 = ap2 + x4
    cdef double x5 = 1/x4
    cdef double x54 = ap2*x5
    cdef double x55 = Apm + x54
    cdef double xi = sqrt(Dbpm)*x4*x55/x56
    cdef double pr = prst/xi
    cdef double flagNLOSS2 = 1.00000000000000
    cdef double am = x0 - x1
    cdef double apam = am*ap
    cdef double am_2 = am**2
    cdef double x81 = 14.0*CES21 + 14.0*CES22
    cdef double x80 = 8.0*x7
    cdef double x9 = 1/x8
    cdef double QSalign2 = (
        flagNLOSS2*pr**4*x9*(
            -0.15625*am_2*(nu*(13.0 - x81) + x7*(8.0*CES21 + 8.0*CES22 - 4.0) + 3.0)
            + 0.15625*ap2*(nu*(x81 + 5.0) - x80*(CES21 + CES22 + 2.0) + 5.0)
            - 0.3125*apam*(
                X_2*(36.0*nu - 2.0) - 2.0*nu*(7.0*CES21 - 7.0*CES22 + 9.0) + x80*(CES21 - CES22) + 1.0
            )
        )
    )
    cdef double flagQPN55 = 1.00000000000000
    cdef double flagQPN5 = 1.00000000000000
    cdef double flagQPN4 = 1.00000000000000
    cdef double x52 = prst**8
    cdef double x53 = nu*x52
    cdef double x51 = prst**6
    cdef double x50 = prst**4
    cdef double x12 = 1/x11
    cdef double x2 = 1/r
    cdef double Qpm = (
        flagQPN4*(
            0.121954868780449*x2*x53
            + x5*x51*(-2.78300763695006*nu + 6.0*x38 - 5.4*x7)
            + x50*x9*(92.7110442849544*nu + 10.0*x38 - 131.0*x7)
        )
        + flagQPN5*(
            x12*x50*(
                nu*(452.542166996693 - 51.6952380952381*x33)
                + 602.318540416564*x38
                + x7*(118.4*x33 - 1796.13660498019)
            )
            + x5*x52*(1.38977750996128*nu - 6.0*x35 + 3.42857142857143*x38 + 3.33842023648322*x7)
            + x51*x9*(-33.9782122170436*nu - 14.0*x35 + 188.0*x38 - 89.5298327361234*x7)
        )
        + flagQPN55*(
            147.443752990146*nu*r**(-4.5)*x50
            - 11.3175085791863*nu*r**(-3.5)*x51
            + 1.48275342024365*r**(-2.5)*x53
        )
        + x5*x50*(8.0*nu - 6.0*x7)
    )
    cdef double Qq = QSalign2 + Qpm
    cdef double x82 = r*(r + 2.0)
    cdef double Bnpa = -x82/(ap2*x82 + x11)
    cdef double flagNLOSS = 1.00000000000000
    cdef double delta = X_1 - X_2
    cdef double x76 = 48.0*nu
    cdef double x58 = 4.0*CES22
    cdef double x78 = x58*(X_2*x76 - 38.0*X_2 + 87.0*nu + 16.0)
    cdef double x64 = 4.0*CES21
    cdef double x77 = x64*(X_2*(x76 - 38.0) - 135.0*nu + 22.0)
    cdef double x79 = x77 - x78
    cdef double x72 = nu - 1.0
    cdef double x74 = x58*x72
    cdef double x73 = x64*x72
    cdef double x75 = x73 + x74
    cdef double x67 = 0.1875*am_2
    cdef double x60 = 4.0*nu
    cdef double BnpSalign2 = (
        flagNLOSS*x9*(
            0.1875*ap2*(8.0*nu + x75 + 23.0)
            + 0.375*apam*(14.0*X_2 + x73 - x74 - 7.0)
            + x67*(-x60 + x75 + 7.0)
        )
        + flagNLOSS2*x12*(
            0.015625*am_2*(-773.0*nu + 4.0*x7 - x79 - 13.0)
            + 0.015625*ap2*(-2059.0*nu - x79 - 837.0)
            + 0.03125*apam*(-delta*(166.0*nu - 601.0) - x77 - x78)
        )
    )
    cdef double Bnp = Apm*Dbpm + BnpSalign2 + x54 - 1.0
    cdef double x70 = 3.0*CES42
    cdef double x69 = 3.0*CES41
    cdef double x71 = x69 + x70
    cdef double x68 = CES41 + CES42
    cdef double x61 = 1/x36
    cdef double x30 = 6.0*CES22
    cdef double x28 = 0.0625*apam
    cdef double x26 = -4.0*CBS31
    cdef double x17 = 4.0*CBS32
    cdef double x27 = -x17 + x26
    cdef double x25 = 0.03125*am_2
    cdef double x20 = 12.0*CBS32
    cdef double x19 = 12.0*CBS31
    cdef double ASalign4 = (
        x61*(
            0.046875*am_2**2*(-CES21*x30 - x27 - x68)
            + 0.015625*ap2**2*(-18.0*CES21*CES22 + 8.0*CES21 + 8.0*CES22 - x19 - x20 - x71 + 32.0)
            + ap2*x25*(CES21*(x30 + 4.0) + x58 - x71 - 8.0)
            + ap2*x28*(-6.0*CBS31 + 6.0*CBS32 + 4.0*CES21 - x58 - x69 + x70)
            + 0.1875*apam**2*(2.0*CES21*CES22 - x68)
            + apam*x67*(2.0*CBS31 - 2.0*CBS32 - CES41 + CES42)
        )
    )
    cdef double x66 = 182.0*nu
    cdef double x62 = 34.0*X_2 + 69.0*nu - 34.0
    cdef double x65 = x62*x64
    cdef double x63 = 276.0*CES22*nu
    cdef double x57 = CES21*(X_2 - 1.0)
    cdef double x59 = -X_2*x58 + 4.0*x57
    cdef double x31 = CES22*X_2
    cdef double ASalign2 = (
        flagNLOSS*x12*(
            0.125*am_2*(x59 + x60 + 5.0) + 0.125*ap2*(x59 + 13.0) + apam*(X_2*(CES22 + 0.5) + x57 - 0.25)
        )
        + flagNLOSS2*x61*(
            0.00669642857142857*am_2*(-741.0*nu - 136.0*x31 + x63 + x65 + 196.0*x7 + 115.0)
            + 0.00223214285714286*ap2*(12.0*CES21*x62 + 828.0*CES22*nu - 2881.0*nu - 408.0*x31 - 1167.0)
            + 0.0133928571428571*apam*(2.0*X_2*(68.0*CES22 + x66 - 409.0) - x63 + x65 - x66 + 409.0)
        )
        + x9*(0.25*(1.0 - CES21)*(am + ap)**2 + 0.25*(1.0 - CES22)*(am - ap)**2)
    )
    cdef double A = (ASalign2 + ASalign4 + x55)/(x54*(2.0*x2 + 1.0) + 1.0)
    cdef double lap = ap
    cdef double x3 = pphi**2
    cdef double x6 = x3*x5
    cdef double Heven = sqrt(A*(Bnpa*lap**2*x6 + Qq + prst**2*(Bnp + 1.0)/xi**2 + x6 + 1.0))
    cdef double lam = am
    cdef double x32 = lam*pphi
    cdef double x29 = CES21*X_2
    cdef double x24 = 12.0*X_2
    cdef double x23 = -15.0*CES22
    cdef double x22 = 3.0*CES21
    cdef double x21 = x19 - x20
    cdef double x18 = 4.0*CBS31 + x17
    cdef double x15 = 9.0*CES21
    cdef double x16 = 9.0*CES22 + x15
    cdef double x14 = lap*pphi
    cdef double Ga3 = (
        x14*x5*(
            0.03125*ap2*(x16 + x18 - 34.0)
            + 0.0208333333333333*apam*(delta*(x16 - 8.0) + x21)
            + x25*(-CES21*x24 + CES22*x24 + x18 - x22 + x23 + 10.0)
        )
        + x32*x5*(
            0.0104166666666667*ap2*(-27.0*CES22 - 44.0*X_2 + x15 + x21 + 18.0*x29 + 18.0*x31 + 22.0)
            + x25*(-X_2*x30 + 12.0*X_2 - x15 - x17 - x23 - x26 - 6.0*x29 - 6.0)
            + x28*(-3.0*CES22 - x22 - x27 - 2.0)
        )
    )
    cdef double SOcalib = dSO*nu*x14*x9
    cdef double flagNLOSO2 = 1.00000000000000
    cdef double flagNLOSO = 1.00000000000000
    cdef double x13 = pphi**4*x12
    cdef double x10 = x3*x9
    cdef double gam = (
        flagNLOSO*(x2*(0.34375*nu + 0.09375) + x6*(0.46875 - 0.28125*nu))
        + flagNLOSO2*(
            x10*(-0.2734375*nu - 0.798177083333333*x7 - 0.23046875)
            + x13*(-0.3515625*nu + 0.29296875*x7 - 0.41015625)
            + x5*(-0.03125*nu + 0.536458333333333*x7 + 0.078125)
        )
        + 0.25
    )
    cdef double gap = (
        flagNLOSO*(x2*(0.71875*nu - 0.09375) + x6*(-1.40625*nu - 0.46875))
        + flagNLOSO2*(
            x10*(-2.0859375*nu - 2.07161458333333*x7 + 0.23046875)
            + x13*(0.5859375*nu + 1.34765625*x7 + 0.41015625)
            + x5*(-5.53125*nu + 0.567708333333333*x7 - 0.078125)
        )
        + 1.75
    )
    cdef double Hodd = (Ga3 + SOcalib + delta*gam*x32 + gap*x14)/(2.0*ap2 + r*(-2.0*r + x56) + 2.0*x4)
    cdef double Heff = Heven + Hodd
    cdef double H = 1.4142135623731*sqrt(nu*(Heff - 1.0) + 0.5)/nu
    cdef double dSS = 0
    cdef double _ASalignCal2 = ap2*dSS*nu/r**6
    return H, xi

cdef class Ham_align_DT_C(Hamiltonian_C):

    def __call__(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
        bint verbose=False
    ):

        cdef:
            double H
            double xi
            double A
            double Bnp
            double Bnpa
            double Qq
            double Heven
            double Hodd

        H, xi, A, Bnp, Bnpa, Qq, Heven, Hodd = self._call(q, p, chi_1, chi_2, m_1, m_2)
        if not verbose:
            return H, xi

        return H, xi, A, Bnp, Bnpa, Qq, Heven, Hodd

    cpdef Hamiltonian_C_call_return_t _call(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        """
        Hamiltonian of spinning SEOBNRv5T, including spin effects
        """
        cdef double r = q[0]
        # cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double pphi = p[1]

        # Extra quantities used in the Hamiltonian
        cdef double M = self.EOBpars.p_params.M
        cdef double _M2 = M * M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2

        cdef double Tidal1 = tidal_contribution(self.EOBpars, r, 1)
        cdef double Tidal2 = tidal_contribution(self.EOBpars, r, 2)

        # Non-spinning calibration coefficient
        cdef double a6 = self.EOBpars.c_coeffs.a6
        # Spin-orbit calibration coefficient
        cdef double dSO = self.EOBpars.c_coeffs.dSO
        # Spin-induced multipole moments
        cdef double CES21 = self.EOBpars.tidal_params.CES21
        cdef double CES22 = self.EOBpars.tidal_params.CES22
        cdef double CES41 = self.EOBpars.tidal_params.CES41
        cdef double CES42 = self.EOBpars.tidal_params.CES42
        cdef double CBS31 = self.EOBpars.tidal_params.CBS31
        cdef double CBS32 = self.EOBpars.tidal_params.CBS32

        cdef double d5 = 0
        cdef double x49 = 7680.0*nu
        cdef double x48 = 315.0*d5
        cdef double x33 = log(r)
        cdef double x47 = 49152.0*x33
        cdef double x7 = nu**2
        cdef double x46 = 102574080.0*x7 - 2119671837.36038
        cdef double x44 = 14700.0*nu + 42911.0
        cdef double x45 = 5760.0*x44
        cdef double x37 = x33**2
        cdef double x43 = 5787938193408.0*x37
        cdef double x42 = 22295347200.0*d5
        cdef double x41 = nu*r
        cdef double x38 = nu**3
        cdef double x8 = r**3
        cdef double x4 = r**2
        cdef double Dbpm = (
            r*(
                -193226342400.0*d5*nu
                + 2589101062873.81*nu*x4
                - 12049908701745.2*nu
                - 326837426.241486*r*x44
                + 1822680546449.21*r*x7
                - 39476764256925.6*r
                + 6730497718123.02*x38
                + 133772083200.0*x4*x7
                + 5107745331375.71*x4
                + x41*x42
                + 10611661054566.2*x41
                + x42*x7
                + x43
                - x47*(-516620178.136075*nu - r*x45 - 17902080.0*x4 - x46)
                + 80059249540278.2*x7
                + 275059053208689.0
            )/(
                55296.0*nu*(
                    4331361844.61149*nu
                    + 14515200.0*x38
                    - x49*(x48 + 890888.810272497)
                    - 42636451.6032331*x7
                    + 1002013764.01019
                )
                + r*x43
                + r*(
                    -4718592.0*nu*(40950.0*d5 + 86207832.4415642)
                    + 450172889755120.0*nu
                    + 5927865218923.02*x38
                    + 70778880.0*x7*(x48 + 2561145.80918574)
                    - 138141470005001.0*x7
                    + 86618264430493.3*(1 - 0.496948781616935*nu)**2
                    + 188440788778196.0
                )
                - 9216.0*x4*(
                    2481453539.84635*nu
                    - x49*(x48 + 405152.309729121)
                    - 197773496.793534*x7
                    + 5805304367.87913
                )
                + x47*(
                    -34560.0*nu*(11592.0*nu + 69847.0)
                    + r*(409207698.136075*nu + x46)
                    + x4*x45
                    + 17902080.0*x8
                )
                - 967680.0*x8*(-2675575.66847905*nu - 138240.0*x7 - 5278341.3229329)
            )
        )
        cdef double x40 = 8.0*r + 4.0*x4 + 2.0*x8 + 16.0
        cdef double x34 = 756.0*nu
        cdef double x39 = x34 + 1079.0
        cdef double x36 = r**5
        cdef double x35 = nu**4
        cdef double x11 = r**4
        cdef double Apm = (
            Tidal1
            + Tidal2
            + 7680.0*x11*(
                2048.0*nu*x33*(336.0*r + x34 + 407.0)
                + 28.0*nu*(1920.0*a6 + 733955.307463037)
                - 7.0*r*(938918.400156317*nu - 185763.092693281*x7 - 245760.0)
                - 5416406.59541186*x7
                - 3440640.0
            )/(
                32768.0*nu*x33*(
                    -38842241.4769507*nu
                    + 240.0*r*(-7466.27061066206*nu - 3024.0*x7 + 17264.0)
                    + 480.0*x11*x39
                    + 161280.0*x36
                    + 960.0*x39*x8
                    + 1920.0*x4*(588.0*nu + 1079.0)
                    - 1882456.23663972*x7
                    + 13447680.0
                )
                + 53760.0*nu*(
                    7680.0*a6*(x11 + x40)
                    + 113485.217444961*r*(-x11 + x40)
                    + 148.04406601634*r*(7704.0*r + 349.0*x11 + 3852.0*x4 + 1926.0*x8 + 36400.0)
                    + 128.0*r*(
                        13218.7851094412*r
                        - 6852.34813868015*x11
                        + 8529.39255472061*x4
                        + 4264.6962773603*x8
                        - 33722.4297811176
                    )
                )
                + 241555486248.807*x35
                + 13212057600.0*x36
                + 67645734912.0*x37*x7
                + 1120.0*x38*(-163683964.822551*r - 17833256.898555*x4 - 1188987459.03162)
                + 7.0*x7*(
                    -39321600.0*a6*(3.0*r + 59.0)
                    + 745857848.115604*a6
                    + 122635399361.987*r
                    - 3089250703.76879*x11
                    + 1426660551.8844*x36
                    + 2064783811.32587*x4
                    - 6178501407.53758*x8
                    + 276057889687.011
                )
            )
        )
        cdef double x1 = X_2*chi_2
        cdef double x0 = X_1*chi_1
        cdef double ap = x0 + x1
        cdef double ap2 = ap**2
        cdef double x56 = ap2 + x4
        cdef double x5 = 1/x4
        cdef double x54 = ap2*x5
        cdef double x55 = Apm + x54
        cdef double xi = sqrt(Dbpm)*x4*x55/x56
        cdef double pr = prst/xi
        cdef double flagNLOSS2 = 1.00000000000000
        cdef double am = x0 - x1
        cdef double apam = am*ap
        cdef double am_2 = am**2
        cdef double x81 = 14.0*CES21 + 14.0*CES22
        cdef double x80 = 8.0*x7
        cdef double x9 = 1/x8
        cdef double QSalign2 = (
            flagNLOSS2*pr**4*x9*(
                -0.15625*am_2*(nu*(13.0 - x81) + x7*(8.0*CES21 + 8.0*CES22 - 4.0) + 3.0)
                + 0.15625*ap2*(nu*(x81 + 5.0) - x80*(CES21 + CES22 + 2.0) + 5.0)
                - 0.3125*apam*(
                    X_2*(36.0*nu - 2.0) - 2.0*nu*(7.0*CES21 - 7.0*CES22 + 9.0) + x80*(CES21 - CES22) + 1.0
                )
            )
        )
        cdef double flagQPN55 = 1.00000000000000
        cdef double flagQPN5 = 1.00000000000000
        cdef double flagQPN4 = 1.00000000000000
        cdef double x52 = prst**8
        cdef double x53 = nu*x52
        cdef double x51 = prst**6
        cdef double x50 = prst**4
        cdef double x12 = 1/x11
        cdef double x2 = 1/r
        cdef double Qpm = (
            flagQPN4*(
                0.121954868780449*x2*x53
                + x5*x51*(-2.78300763695006*nu + 6.0*x38 - 5.4*x7)
                + x50*x9*(92.7110442849544*nu + 10.0*x38 - 131.0*x7)
            )
            + flagQPN5*(
                x12*x50*(
                    nu*(452.542166996693 - 51.6952380952381*x33)
                    + 602.318540416564*x38
                    + x7*(118.4*x33 - 1796.13660498019)
                )
                + x5*x52*(1.38977750996128*nu - 6.0*x35 + 3.42857142857143*x38 + 3.33842023648322*x7)
                + x51*x9*(-33.9782122170436*nu - 14.0*x35 + 188.0*x38 - 89.5298327361234*x7)
            )
            + flagQPN55*(
                147.443752990146*nu*r**(-4.5)*x50
                - 11.3175085791863*nu*r**(-3.5)*x51
                + 1.48275342024365*r**(-2.5)*x53
            )
            + x5*x50*(8.0*nu - 6.0*x7)
        )
        cdef double Qq = QSalign2 + Qpm
        cdef double x82 = r*(r + 2.0)
        cdef double Bnpa = -x82/(ap2*x82 + x11)
        cdef double flagNLOSS = 1.00000000000000
        cdef double delta = X_1 - X_2
        cdef double x76 = 48.0*nu
        cdef double x58 = 4.0*CES22
        cdef double x78 = x58*(X_2*x76 - 38.0*X_2 + 87.0*nu + 16.0)
        cdef double x64 = 4.0*CES21
        cdef double x77 = x64*(X_2*(x76 - 38.0) - 135.0*nu + 22.0)
        cdef double x79 = x77 - x78
        cdef double x72 = nu - 1.0
        cdef double x74 = x58*x72
        cdef double x73 = x64*x72
        cdef double x75 = x73 + x74
        cdef double x67 = 0.1875*am_2
        cdef double x60 = 4.0*nu
        cdef double BnpSalign2 = (
            flagNLOSS*x9*(
                0.1875*ap2*(8.0*nu + x75 + 23.0)
                + 0.375*apam*(14.0*X_2 + x73 - x74 - 7.0)
                + x67*(-x60 + x75 + 7.0)
            )
            + flagNLOSS2*x12*(
                0.015625*am_2*(-773.0*nu + 4.0*x7 - x79 - 13.0)
                + 0.015625*ap2*(-2059.0*nu - x79 - 837.0)
                + 0.03125*apam*(-delta*(166.0*nu - 601.0) - x77 - x78)
            )
        )
        cdef double Bnp = Apm*Dbpm + BnpSalign2 + x54 - 1.0
        cdef double x70 = 3.0*CES42
        cdef double x69 = 3.0*CES41
        cdef double x71 = x69 + x70
        cdef double x68 = CES41 + CES42
        cdef double x61 = 1/x36
        cdef double x30 = 6.0*CES22
        cdef double x28 = 0.0625*apam
        cdef double x26 = -4.0*CBS31
        cdef double x17 = 4.0*CBS32
        cdef double x27 = -x17 + x26
        cdef double x25 = 0.03125*am_2
        cdef double x20 = 12.0*CBS32
        cdef double x19 = 12.0*CBS31
        cdef double ASalign4 = (
            x61*(
                0.046875*am_2**2*(-CES21*x30 - x27 - x68)
                + 0.015625*ap2**2*(-18.0*CES21*CES22 + 8.0*CES21 + 8.0*CES22 - x19 - x20 - x71 + 32.0)
                + ap2*x25*(CES21*(x30 + 4.0) + x58 - x71 - 8.0)
                + ap2*x28*(-6.0*CBS31 + 6.0*CBS32 + 4.0*CES21 - x58 - x69 + x70)
                + 0.1875*apam**2*(2.0*CES21*CES22 - x68)
                + apam*x67*(2.0*CBS31 - 2.0*CBS32 - CES41 + CES42)
            )
        )
        cdef double x66 = 182.0*nu
        cdef double x62 = 34.0*X_2 + 69.0*nu - 34.0
        cdef double x65 = x62*x64
        cdef double x63 = 276.0*CES22*nu
        cdef double x57 = CES21*(X_2 - 1.0)
        cdef double x59 = -X_2*x58 + 4.0*x57
        cdef double x31 = CES22*X_2
        cdef double ASalign2 = (
            flagNLOSS*x12*(
                0.125*am_2*(x59 + x60 + 5.0) + 0.125*ap2*(x59 + 13.0) + apam*(X_2*(CES22 + 0.5) + x57 - 0.25)
            )
            + flagNLOSS2*x61*(
                0.00669642857142857*am_2*(-741.0*nu - 136.0*x31 + x63 + x65 + 196.0*x7 + 115.0)
                + 0.00223214285714286*ap2*(12.0*CES21*x62 + 828.0*CES22*nu - 2881.0*nu - 408.0*x31 - 1167.0)
                + 0.0133928571428571*apam*(2.0*X_2*(68.0*CES22 + x66 - 409.0) - x63 + x65 - x66 + 409.0)
            )
            + x9*(0.25*(1.0 - CES21)*(am + ap)**2 + 0.25*(1.0 - CES22)*(am - ap)**2)
        )
        cdef double A = (ASalign2 + ASalign4 + x55)/(x54*(2.0*x2 + 1.0) + 1.0)
        cdef double lap = ap
        cdef double x3 = pphi**2
        cdef double x6 = x3*x5
        cdef double Heven = sqrt(A*(Bnpa*lap**2*x6 + Qq + prst**2*(Bnp + 1.0)/xi**2 + x6 + 1.0))
        cdef double lam = am
        cdef double x32 = lam*pphi
        cdef double x29 = CES21*X_2
        cdef double x24 = 12.0*X_2
        cdef double x23 = -15.0*CES22
        cdef double x22 = 3.0*CES21
        cdef double x21 = x19 - x20
        cdef double x18 = 4.0*CBS31 + x17
        cdef double x15 = 9.0*CES21
        cdef double x16 = 9.0*CES22 + x15
        cdef double x14 = lap*pphi
        cdef double Ga3 = (
            x14*x5*(
                0.03125*ap2*(x16 + x18 - 34.0)
                + 0.0208333333333333*apam*(delta*(x16 - 8.0) + x21)
                + x25*(-CES21*x24 + CES22*x24 + x18 - x22 + x23 + 10.0)
            )
            + x32*x5*(
                0.0104166666666667*ap2*(-27.0*CES22 - 44.0*X_2 + x15 + x21 + 18.0*x29 + 18.0*x31 + 22.0)
                + x25*(-X_2*x30 + 12.0*X_2 - x15 - x17 - x23 - x26 - 6.0*x29 - 6.0)
                + x28*(-3.0*CES22 - x22 - x27 - 2.0)
            )
        )
        cdef double SOcalib = dSO*nu*x14*x9
        cdef double flagNLOSO2 = 1.00000000000000
        cdef double flagNLOSO = 1.00000000000000
        cdef double x13 = pphi**4*x12
        cdef double x10 = x3*x9
        cdef double gam = (
            flagNLOSO*(x2*(0.34375*nu + 0.09375) + x6*(0.46875 - 0.28125*nu))
            + flagNLOSO2*(
                x10*(-0.2734375*nu - 0.798177083333333*x7 - 0.23046875)
                + x13*(-0.3515625*nu + 0.29296875*x7 - 0.41015625)
                + x5*(-0.03125*nu + 0.536458333333333*x7 + 0.078125)
            )
            + 0.25
        )
        cdef double gap = (
            flagNLOSO*(x2*(0.71875*nu - 0.09375) + x6*(-1.40625*nu - 0.46875))
            + flagNLOSO2*(
                x10*(-2.0859375*nu - 2.07161458333333*x7 + 0.23046875)
                + x13*(0.5859375*nu + 1.34765625*x7 + 0.41015625)
                + x5*(-5.53125*nu + 0.567708333333333*x7 - 0.078125)
            )
            + 1.75
        )
        cdef double Hodd = (Ga3 + SOcalib + delta*gam*x32 + gap*x14)/(2.0*ap2 + r*(-2.0*r + x56) + 2.0*x4)
        cdef double Heff = Heven + Hodd
        cdef double H = 1.4142135623731*sqrt(nu*(Heff - 1.0) + 0.5)/nu
        cdef double dSS = 0
        cdef double _ASalignCal2 = ap2*dSS*nu/r**6

        return H, xi, A, Bnp, Bnpa, Qq, Heven, Hodd

    cpdef Hamiltonian_C_grad_return_t grad(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2
    ):
        """
        Jacobian of the SEOBNRv5HM Hamiltonian
        """
        cdef double r = q[0]
        # cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double pphi = p[1]

        # Extra quantities used in the Hamiltonian
        cdef double M = self.EOBpars.p_params.M
        cdef double M2 = M * M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2

        cdef (double, double) tides1 = tidal_and_d_tidal_contribution(self.EOBpars, r, 1)
        cdef double Tidal1 = tides1[0]
        cdef double dTidal1 = tides1[1]
        cdef (double, double) tides2 = tidal_and_d_tidal_contribution(self.EOBpars, r, 2)
        cdef double Tidal2 = tides2[0]
        cdef double dTidal2 = tides2[1]

        # Non-spinning calibration coefficient
        cdef double a6 = self.EOBpars.c_coeffs.a6
        # Spin-orbit calibration coefficient
        cdef double dSO = self.EOBpars.c_coeffs.dSO
        # Spin-induced multipole moments
        cdef double CES21 = self.EOBpars.tidal_params.CES21
        cdef double CES22 = self.EOBpars.tidal_params.CES22
        cdef double CES41 = self.EOBpars.tidal_params.CES41
        cdef double CES42 = self.EOBpars.tidal_params.CES42
        cdef double CBS31 = self.EOBpars.tidal_params.CBS31
        cdef double CBS32 = self.EOBpars.tidal_params.CBS32
        cdef double x11 = r**4
        cdef double x12 = 1/x11
        cdef double x242 = 4.0*pphi**3*x12
        cdef double x60 = 2.0*pphi
        cdef double x0 = r**2
        cdef double x15 = 1/x0
        cdef double x241 = x15*x60
        cdef double x2 = X_2*chi_2
        cdef double x1 = X_1*chi_1
        cdef double x3 = x1 + x2
        cdef double x4 = x3**2
        cdef double x72 = x15*x4
        cdef double x63 = 1/r
        cdef double x71 = 2.0*x63 + 1.0
        cdef double x73 = x71*x72 + 1.0
        cdef double x218 = 1/x73
        cdef double x124 = 113485.217444961*r
        cdef double x123 = 148.04406601634*r
        cdef double x122 = 7704.0*r
        cdef double x121 = 128.0*r
        cdef double x120 = 7680.0*a6
        cdef double x118 = 8.0*r
        cdef double x18 = r**3
        cdef double x119 = 4.0*x0 + x118 + 2.0*x18 + 16.0
        cdef double x125 = (
            nu*(
                x120*(x11 + x119)
                + x121*(
                    13218.7851094412*r
                    + 8529.39255472061*x0
                    - 6852.34813868015*x11
                    + 4264.6962773603*x18
                    - 33722.4297811176
                )
                + x123*(3852.0*x0 + 349.0*x11 + x122 + 1926.0*x18 + 36400.0)
                + x124*(-x11 + x119)
            )
        )
        cdef double x102 = log(r)
        cdef double x116 = nu*x102
        cdef double x103 = 756.0*nu
        cdef double x113 = x103 + 1079.0
        cdef double x114 = x113*x18
        cdef double x112 = 588.0*nu + 1079.0
        cdef double x28 = r**5
        cdef double x23 = nu**2
        cdef double x115 = (
            -38842241.4769507*nu
            + 240.0*r*(-7466.27061066206*nu - 3024.0*x23 + 17264.0)
            + 1920.0*x0*x112
            + 480.0*x11*x113
            + 960.0*x114
            - 1882456.23663972*x23
            + 161280.0*x28
            + 13447680.0
        )
        cdef double x117 = x115*x116
        cdef double x111 = (
            x23*(
                -39321600.0*a6*(3.0*r + 59.0)
                + 745857848.115604*a6
                + 122635399361.987*r
                + 2064783811.32587*x0
                - 3089250703.76879*x11
                - 6178501407.53758*x18
                + 1426660551.8844*x28
                + 276057889687.011
            )
        )
        cdef double x109 = nu**3
        cdef double x110 = x109*(-163683964.822551*r - 17833256.898555*x0 - 1188987459.03162)
        cdef double x107 = x102**2
        cdef double x108 = x107*x23
        cdef double x106 = nu**4
        cdef double x126 = (
            1/(
                241555486248.807*x106
                + 67645734912.0*x108
                + 1120.0*x110
                + 7.0*x111
                + 32768.0*x117
                + 53760.0*x125
                + 13212057600.0*x28
            )
        )
        cdef double x127 = x11*x126
        cdef double x104 = 336.0*r + x103 + 407.0
        cdef double x105 = (
            2048.0*nu*x102*x104
            + 28.0*nu*(1920.0*a6 + 733955.307463037)
            - 7.0*r*(938918.400156317*nu - 185763.092693281*x23 - 245760.0)
            - 5416406.59541186*x23
            - 3440640.0
        )
        cdef double x128 = x105*x127
        cdef double x129 = Tidal1 + Tidal2 + 7680.0*x128
        cdef double x101 = X_2**2*chi_2**2*(1.0 - CES22)
        cdef double x100 = X_1**2*chi_1**2*(1.0 - CES21)
        cdef double x94 = X_2 - 1.0
        cdef double x33 = x1 - x2
        cdef double x51 = x3*x33
        cdef double x99 = x51*(CES21*x94 + X_2*(CES22 + 0.5) - 0.25)
        cdef double x97 = 4.0*nu
        cdef double x91 = 4.0*CES22
        cdef double x78 = 4.0*CES21
        cdef double x95 = -X_2*x91 + x78*x94
        cdef double x46 = x33**2
        cdef double x98 = x46*(x95 + x97 + 5.0)
        cdef double x96 = x4*(x95 + 13.0)
        cdef double x87 = 3.0*CES42
        cdef double x86 = 3.0*CES41
        cdef double x88 = x86 + x87
        cdef double x58 = 6.0*CES22
        cdef double x93 = x46*(CES21*(x58 + 4.0) - x88 + x91 - 8.0)
        cdef double x92 = x3**3*x33*(-6.0*CBS31 + 6.0*CBS32 + 4.0*CES21 - x86 + x87 - x91)
        cdef double x83 = CES41 + CES42
        cdef double x90 = x4*x46*(2.0*CES21*CES22 - x83)
        cdef double x49 = 12.0*CBS32
        cdef double x48 = 12.0*CBS31
        cdef double x89 = x3**4*(-18.0*CES21*CES22 + 8.0*CES21 + 8.0*CES22 - x48 - x49 - x88 + 32.0)
        cdef double x85 = x3*x33**3*(2.0*CBS31 - 2.0*CBS32 - CES41 + CES42)
        cdef double x54 = -4.0*CBS31
        cdef double x40 = 4.0*CBS32
        cdef double x55 = -x40 + x54
        cdef double x84 = x33**4*(-CES21*x58 - x55 - x83)
        cdef double x81 = 182.0*nu
        cdef double x75 = 34.0*X_2 + 69.0*nu - 34.0
        cdef double x79 = x75*x78
        cdef double x77 = 276.0*CES22*nu
        cdef double x82 = x51*(2.0*X_2*(68.0*CES22 + x81 - 409.0) - x77 + x79 - x81 + 409.0)
        cdef double x57 = CES22*X_2
        cdef double x80 = x46*(-741.0*nu + 196.0*x23 - 136.0*x57 + x77 + x79 + 115.0)
        cdef double x76 = x4*(12.0*CES21*x75 + 828.0*CES22*nu - 2881.0*nu - 408.0*x57 - 1167.0)
        cdef double x42 = 0.03125*x4
        cdef double x29 = 1/x28
        cdef double x19 = 1/x18
        cdef double x130 = (
            x12*(0.125*x96 + 0.125*x98 + x99)
            + x129
            + x19*(x100 + x101)
            + x29*(0.00223214285714286*x76 + 0.00669642857142857*x80 + 0.0133928571428571*x82)
            + x29*(x42*x93 + 0.046875*x84 + 0.1875*x85 + 0.015625*x89 + 0.1875*x90 + 0.0625*x92)
            + x72
        )
        cdef double x233 = x130*x218
        cdef double x175 = 102574080.0*x23 - 2119671837.36038
        cdef double x176 = 409207698.136075*nu + x175
        cdef double x177 = r*x176
        cdef double x173 = 14700.0*nu + 42911.0
        cdef double x174 = 5760.0*x173
        cdef double x172 = nu*(11592.0*nu + 69847.0)
        cdef double x178 = x102*(x0*x174 - 34560.0*x172 + x177 + 17902080.0*x18)
        cdef double x170 = (
            43393301259014.8*nu
            + 5927865218923.02*x109
            + 43133561885859.3*x23
            + 86618264430493.3*(1 - 0.496948781616935*nu)**2
            + 188440788778196.0
        )
        cdef double x171 = r*x170
        cdef double x169 = (
            nu*(-2510664218.28128*nu + 14515200.0*x109 - 42636451.6032331*x23 + 1002013764.01019)
        )
        cdef double x167 = -2675575.66847905*nu - 138240.0*x23 - 5278341.3229329
        cdef double x168 = x167*x18
        cdef double x165 = -630116198.873299*nu - 197773496.793534*x23 + 5805304367.87913
        cdef double x166 = x0*x165
        cdef double x164 = r*x107
        cdef double x213 = (
            5787938193408.0*x164 - 9216.0*x166 - 967680.0*x168 + 55296.0*x169 + x171 + 49152.0*x178
        )
        cdef double x214 = 1/x213
        cdef double x206 = 48.0*nu
        cdef double x208 = x91*(X_2*x206 - 38.0*X_2 + 87.0*nu + 16.0)
        cdef double x207 = x78*(X_2*(x206 - 38.0) - 135.0*nu + 22.0)
        cdef double x32 = X_1 - X_2
        cdef double x212 = x51*(-x207 - x208 - x32*(166.0*nu - 601.0))
        cdef double x209 = x207 - x208
        cdef double x211 = x46*(-773.0*nu - x209 + 4.0*x23 - 13.0)
        cdef double x210 = x4*(-2059.0*nu - x209 - 837.0)
        cdef double x199 = nu - 1.0
        cdef double x201 = x199*x91
        cdef double x200 = x199*x78
        cdef double x205 = x51*(14.0*X_2 + x200 - x201 - 7.0)
        cdef double x202 = x200 + x201
        cdef double x204 = x46*(x202 - x97 + 7.0)
        cdef double x203 = x4*(8.0*nu + x202 + 23.0)
        cdef double x196 = 5787938193408.0*x107
        cdef double x195 = 1822680546449.21*x23
        cdef double x185 = x102*(-516620178.136075*nu - r*x174 - 17902080.0*x0 - x175)
        cdef double x184 = x0*x23
        cdef double x183 = r*x173
        cdef double x181 = nu*x0
        cdef double x180 = nu*r
        cdef double x197 = (
            -12049908701745.2*nu
            + r*x195
            - 39476764256925.6*r
            + 5107745331375.71*x0
            + 6730497718123.02*x109
            + 10611661054566.2*x180
            + 2589101062873.81*x181
            - 326837426.241486*x183
            + 133772083200.0*x184
            - 49152.0*x185
            + x196
            + 80059249540278.2*x23
            + 275059053208689.0
        )
        cdef double x215 = (
            r*x129*x197*x214
            + x12*(0.015625*x210 + 0.015625*x211 + 0.03125*x212)
            + x19*(0.1875*x203 + 0.1875*x204 + 0.375*x205)
            + x72
        )
        cdef double x198 = 1/x197
        cdef double x162 = (
            0.000130208333333333*Tidal1 + 0.000130208333333333*Tidal2 + x128 + 0.000130208333333333*x72
        )
        cdef double x194 = x162**(-2)
        cdef double x193 = prst**2
        cdef double x7 = x0 + x4
        cdef double x192 = x7**2
        cdef double x216 = x192*x193*x194*x198*x213*x215
        cdef double x189 = 14.0*CES21 + 14.0*CES22
        cdef double x188 = 8.0*x23
        cdef double x190 = (
            0.15625*x4*(nu*(x189 + 5.0) - x188*(CES21 + CES22 + 2.0) + 5.0)
            - 0.15625*x46*(nu*(13.0 - x189) + x23*(8.0*CES21 + 8.0*CES22 - 4.0) + 3.0)
            - 0.3125*x51*(
                X_2*(36.0*nu - 2.0) - 2.0*nu*(7.0*CES21 - 7.0*CES22 + 9.0) + x188*(CES21 - CES22) + 1.0
            )
        )
        cdef double x182 = r*x23
        cdef double x186 = (
            -0.0438084424460039*nu
            - 0.143521050466841*r
            + 0.0185696317637669*x0
            + 0.0210425293255724*x107
            + 0.0244692826489756*x109
            + 0.0385795738434214*x180
            + 0.00941289164152486*x181
            + 0.00662650629087394*x182
            - 1.18824456940711e-6*x183
            + 0.000486339502879429*x184
            - 1.78696172427774e-10*x185
            + 0.291062041428379*x23
            + 1
        )
        cdef double x187 = x186**(-2)
        cdef double x179 = (
            (
                x164
                - 1.59227685093395e-9*x166
                - 1.67189069348064e-7*x168
                + 9.55366110560367e-9*x169
                + 1.72773095804465e-13*x171
                + 8.49214320498104e-9*x178
            )**2
        )
        cdef double x163 = x162**(-4)
        cdef double x161 = x7**4
        cdef double x131 = prst**4
        cdef double x191 = x131*x161*x163*x179*x187*x190
        cdef double x160 = r**(-13)
        cdef double x156 = r*x4
        cdef double x154 = r + 2.0
        cdef double x157 = x11 + x154*x156
        cdef double x158 = 1/x157
        cdef double x20 = pphi**2
        cdef double x159 = x158*x20*x63
        cdef double x155 = x154*x4
        cdef double x153 = x12*x131
        cdef double x152 = (
            nu*(452.542166996693 - 51.6952380952381*x102)
            + 602.318540416564*x109
            + x23*(118.4*x102 - 1796.13660498019)
        )
        cdef double x150 = 1.38977750996128*nu - 6.0*x106 + 3.42857142857143*x109 + 3.33842023648322*x23
        cdef double x138 = prst**8
        cdef double x151 = x138*x150
        cdef double x134 = prst**6
        cdef double x149 = x134*x19
        cdef double x148 = -33.9782122170436*nu - 14.0*x106 + 188.0*x109 - 89.5298327361234*x23
        cdef double x146 = -2.78300763695006*nu + 6.0*x109 - 5.4*x23
        cdef double x147 = x146*x15
        cdef double x145 = x131*x19
        cdef double x144 = 92.7110442849544*nu + 10.0*x109 - 131.0*x23
        cdef double x142 = 8.0*nu - 6.0*x23
        cdef double x143 = x142*x15
        cdef double x141 = 0.121954868780449*x138
        cdef double x140 = nu*x63
        cdef double x139 = nu*x138
        cdef double x137 = r**(-2.5)
        cdef double x135 = r**(-3.5)
        cdef double x136 = nu*x135
        cdef double x132 = r**(-4.5)
        cdef double x133 = nu*x132
        cdef double x64 = x15*x20
        cdef double x217 = (
            147.443752990146*x131*x133
            + x131*x143
            - 11.3175085791863*x134*x136
            + x134*x147
            + 1.48275342024365*x137*x139
            + x140*x141
            + x144*x145
            + x148*x149
            + x15*x151
            + x152*x153
            - x155*x159
            + 1.27277314139085e-19*x160*x191
            + 1.69542100694444e-8*x216*x29
            + x64
            + 1.0
        )
        cdef double x234 = 0.5*(x217*x233)**(-0.5)
        cdef double x240 = x233*x234
        cdef double x56 = CES21*X_2
        cdef double x50 = x48 - x49
        cdef double x47 = 0.03125*x46
        cdef double x44 = -15.0*CES22
        cdef double x43 = 3.0*CES21
        cdef double x38 = 9.0*CES21
        cdef double x59 = (
            x33*(
                0.0104166666666667*x4*(-27.0*CES22 - 44.0*X_2 + x38 + x50 + 18.0*x56 + 18.0*x57 + 22.0)
                + x47*(-X_2*x58 + 12.0*X_2 - x38 - x40 - x44 - x54 - 6.0*x56 - 6.0)
                + 0.0625*x51*(-3.0*CES22 - x43 - x55 - 2.0)
            )
        )
        cdef double x70 = x15*x59
        cdef double x45 = 12.0*X_2
        cdef double x41 = 4.0*CBS31 + x40
        cdef double x39 = 9.0*CES22 + x38
        cdef double x52 = (
            x42*(x39 + x41 - 34.0)
            + x47*(-CES21*x45 + CES22*x45 + x41 - x43 + x44 + 10.0)
            + 0.0208333333333333*x51*(x32*(x39 - 8.0) + x50)
        )
        cdef double x69 = x15*x52
        cdef double x67 = x32*x33
        cdef double x27 = pphi**4
        cdef double x65 = x12*x27
        cdef double x37 = -0.3515625*nu + 0.29296875*x23 - 0.41015625
        cdef double x36 = -0.2734375*nu - 0.798177083333333*x23 - 0.23046875
        cdef double x35 = 0.46875 - 0.28125*nu
        cdef double x34 = 0.34375*nu + 0.09375
        cdef double x21 = x19*x20
        cdef double x68 = (
            x67*(
                x15*(-0.03125*nu + 0.536458333333333*x23 + 0.078125)
                + x21*x36
                + x34*x63
                + x35*x64
                + x37*x65
                + 0.25
            )
        )
        cdef double x26 = 0.5859375*nu + 1.34765625*x23 + 0.41015625
        cdef double x24 = -2.0859375*nu - 2.07161458333333*x23 + 0.23046875
        cdef double x17 = -1.40625*nu - 0.46875
        cdef double x16 = 0.71875*nu - 0.09375
        cdef double x66 = (
            x3*(
                x15*(-5.53125*nu + 0.567708333333333*x23 - 0.078125)
                + x16*x63
                + x17*x64
                + x21*x24
                + x26*x65
                + 1.75
            )
        )
        cdef double x14 = dSO*nu
        cdef double x62 = x14*x19
        cdef double x61 = x19*x60
        cdef double x10 = pphi*x3
        cdef double x6 = 2.0*r
        cdef double x8 = r*(-x6 + x7)
        cdef double x5 = 2.0*x4
        cdef double x9 = 1/(2.0*x0 + x5 + x8)
        cdef double dHeffdpphi = (
            x240*(2.0*pphi*x15 - pphi*x154*x158*x5*x63)
            + x9*(
                pphi*x67*(x241*x35 + x242*x37 + x36*x61)
                + x10*(x17*x241 + x24*x61 + x242*x26)
                + x3*x62
                + x3*x69
                + x66
                + x68
                + x70
            )
        )
        cdef double x236 = prst**5
        cdef double x239 = 6.0*x236
        cdef double x235 = prst**3
        cdef double x238 = 4.0*x235
        cdef double x237 = prst**7
        cdef double x232 = x213*x29
        cdef double dHeffdpr = (
            x240*(
                11.8620273619492*nu*x137*x237
                + 3.39084201388889e-8*prst*x192*x194*x198*x215*x232
                + x12*x152*x238
                + 589.775011960583*x133*x235
                - 67.9050514751178*x136*x236
                + 0.975638950243592*x140*x237
                + x143*x238
                + x144*x19*x238
                + x147*x239
                + x148*x19*x239
                + 8.0*x15*x150*x237
                + 5.09109256556341e-19*x160*x161*x163*x179*x187*x190*x235
            )
        )
        cdef double dHeffdphi = 0
        cdef double x223 = 6.0*x0 + x118 + 8.0
        cdef double x222 = 4.0*x18
        cdef double x221 = x23*x63
        cdef double x224 = (
            1.31621673590926e-19*x105*x11*(
                53760.0*nu*(
                    3740417.71815805*r
                    + 2115968.85907902*x0
                    - 938918.400156317*x11
                    + x120*(x222 + x223)
                    + x121*(
                        17058.7851094412*r + 12794.0888320809*x0 - 27409.3925547206*x18 + 13218.7851094412
                    )
                    + x123*(5778.0*x0 + x122 + 1396.0*x18 + 7704.0)
                    + x124*(-x222 + x223)
                    + 1057984.42953951*x18
                    + 2888096.47013111
                )
                + 135291469824.0*x102*x221
                + 1120.0*x109*(-35666513.7971099*r - 163683964.822551)
                + 66060288000.0*x11
                + 32768.0*x115*x140
                + 32768.0*x116*(
                    -1791904.9465589*nu
                    + 3840.0*r*x112
                    + 2880.0*x0*x113
                    + 806400.0*x11
                    + 1920.0*x114
                    - 725760.0*x23
                    + 4143360.0
                )
                + 7.0*x23*(
                    -117964800.0*a6
                    + 4129567622.65173*r
                    - 18535504222.6128*x0
                    + 7133302759.42198*x11
                    - 12357002815.0752*x18
                    + 122635399361.987
                )
            )/(
                x106
                + 0.28004222119933*x108
                + 4.63661586574928e-9*x110
                + 2.8978849160933e-11*x111
                + 1.35654132757922e-7*x117
                + 2.22557561555966e-7*x125
                + 0.0546957463279941*x28
            )**2
        )
        cdef double x220 = (
            -6572428.80109422*nu + 2048.0*x104*x140 + 688128.0*x116 + 1300341.64885296*x23 + 1720320.0
        )
        cdef double x230 = dTidal1 + dTidal2 + 30720.0*x105*x126*x18 + 7680.0*x127*x220 - x224
        cdef double x74 = x19*x5
        cdef double x231 = x230 - x74
        cdef double x229 = x193*x215
        cdef double x225 = 11575876386816.0*x102
        cdef double x228 = (
            -18432.0*r*x165
            - 2903040.0*x0*x167
            + 49152.0*x102*(53706240.0*x0 + x176 + 11520.0*x183)
            + x170
            + x196
            + x225
            + x63*(283115520.0*x0*x173 - 1698693120.0*x172 + 49152.0*x177 + 879923036160.0*x18)
        )
        cdef double x227 = x131*x160*x161*x179*x190
        cdef double x226 = (
            5807150888816.34*nu
            + 10215490662751.4*r
            + 6291456.0*x102*(661500.0*nu + 279720.0*r + 1930995.0)
            + 5178202125747.62*x180
            + 267544166400.0*x182
            + x195
            + x225*x63
            - x63*(
                -25392914995744.3*nu
                - 879923036160.0*x0
                - 283115520.0*x183
                - 5041721180160.0*x23
                + 104186110149937.0
            )
            - 53501685054374.1
        )
        cdef double x219 = r**(-6)
        cdef double x53 = 2.0*x19
        cdef double x30 = 4.0*x29
        cdef double x31 = x27*x30
        cdef double x13 = 3.0*x12
        cdef double x25 = x13*x20
        cdef double x22 = 2.0*x21
        cdef double dHeffdr = (
            x234*(
                -x130*x217*(-x12*x5 - x71*x74)/x73**2
                + x217*x218*(
                    dTidal1
                    + dTidal2
                    + 30720.0*x105*x126*x18
                    + 7680.0*x11*x126*x220
                    - x12*(3.0*x100 + 3.0*x101)
                    - x219*(0.0111607142857143*x76 + 0.0334821428571429*x80 + 0.0669642857142857*x82)
                    - x219*(
                        0.15625*x4*x93 + 0.234375*x84 + 0.9375*x85 + 0.078125*x89 + 0.9375*x90 + 0.3125*x92
                    )
                    - x224
                    - x29*(0.5*x96 + 0.5*x98 + 4.0*x99)
                    - x74
                )
                + x233*(
                    -663.496888455656*nu*r**(-5.5)*x131
                    + 39.6112800271521*nu*x132*x134
                    - nu*x141*x15
                    + x12*x131*(-51.6952380952381*x140 + 118.4*x221)
                    + 6.78168402777778e-8*x12*x193*x194*x198*x213*x215*x7
                    - x13*x134*x148
                    - x131*x152*x30
                    + 7.59859378406358e-45*x131*x160*x161*x163*x187*x190*x213*x228
                    - 3.70688355060912*x135*x139
                    - 2.0*x142*x145
                    - 3.0*x144*x153
                    - 2.0*x146*x149
                    + x15*x154*x158*x20*x4
                    - x151*x53
                    + x154*x20*x4*x63*(x155 + x156 + x222)/x157**2
                    - x159*x4
                    - 9.25454462627843e-34*x163*x226*x227/x186**3
                    - 2.24091649004576e-37*x187*x192*x194*x213*x226*x229*x29
                    + 1.69542100694444e-8*x192*x193*x194*x198*x213*x29*(
                        r*x129*x214*x226
                        - 2.98505426338587e-26*r*x129*x197*x228/x179
                        + r*x197*x214*x230
                        - x12*(0.5625*x203 + 0.5625*x204 + 1.125*x205)
                        + x129*x197*x214
                        - x29*(0.0625*x210 + 0.0625*x211 + 0.125*x212)
                        - x74
                    )
                    + 1.69542100694444e-8*x192*x193*x194*x198*x215*x228*x29
                    - 8.47710503472222e-8*x216*x219
                    - x22
                    - 4.41515887225116e-12*x192*x198*x229*x231*x232/x162**3
                    - 6.62902677807736e-23*x187*x227*x231/x162**5
                    + 1.01821851311268e-18*x131*x163*x179*x187*x190*x7**3/r**12
                    - 1.65460508380811e-18*x191/r**14
                )
            )
            + x9*(
                pphi*x3*(
                    -x15*x16
                    - x17*x22
                    - x19*(-11.0625*nu + 1.13541666666667*x23 - 0.15625)
                    - x24*x25
                    - x26*x31
                )
                + pphi*x32*x33*(
                    -x15*x34 - x19*(-0.0625*nu + 1.07291666666667*x23 + 0.15625) - x22*x35 - x25*x36 - x31*x37
                )
                - x10*x13*x14
                - x10*x52*x53
                - x59*x61
            )
            - 0.25*(
                r*(x6 - 2.0) + x6 + x7
            )*(
                pphi*x66 + pphi*x68 + pphi*x70 + x10*x62 + x10*x69
            )/(
                x7 + 0.5*x8
            )**2
        )
        # Evaluate Hamiltonian
        cdef double H, _xi
        H, _xi = self.__call__(q, p, chi_1, chi_2, m_1, m_2, verbose=False)
        cdef double nuH = nu * H

        # Compute H Jacobian
        cdef double dHdr = M2 * dHeffdr / nuH
        cdef double dHdphi = M2 * dHeffdphi / nuH
        cdef double dHdpr = M2 * dHeffdpr / nuH
        cdef double dHdpphi = M2 * dHeffdpphi / nuH

        # Return the gradient of H
        return [dHdr, dHdphi, dHdpr, dHdpphi]

    cpdef hessian(self, qp_param_t q, qp_param_t p, double chi_1, double chi_2, double m_1, double m_2):
        """
        Hessian of the SEOBNRv5HM Hamiltonian
        """
        cdef double r = q[0]
        # cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double pphi = p[1]

        # Extra quantities used in the Hamiltonian
        cdef double M = self.EOBpars.p_params.M
        cdef double M2 = M * M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2

        cdef (double, double, double) tides1 = tidal_and_d_tidal_and_d2_tidal_contribution(self.EOBpars, r, 1)
        cdef double Tidal1 = tides1[0]
        cdef double dTidal1 = tides1[1]
        cdef double ddTidal1 = tides1[2]
        cdef (double, double, double) tides2 = tidal_and_d_tidal_and_d2_tidal_contribution(self.EOBpars, r, 2)
        cdef double Tidal2 = tides2[0]
        cdef double dTidal2 = tides2[1]
        cdef double ddTidal2 = tides2[2]

        # Non-spinning calibration coefficient
        cdef double a6 = self.EOBpars.c_coeffs.a6
        # Spin-orbit calibration coefficient
        cdef double dSO = self.EOBpars.c_coeffs.dSO
        # Spin-induced multipole moments
        cdef double CES21 = self.EOBpars.tidal_params.CES21
        cdef double CES22 = self.EOBpars.tidal_params.CES22
        cdef double CES41 = self.EOBpars.tidal_params.CES41
        cdef double CES42 = self.EOBpars.tidal_params.CES42
        cdef double CBS31 = self.EOBpars.tidal_params.CBS31
        cdef double CBS32 = self.EOBpars.tidal_params.CBS32
        cdef double x11 = r**4
        cdef double x12 = 1/x11
        cdef double x242 = 4.0*pphi**3*x12
        cdef double x60 = 2.0*pphi
        cdef double x0 = r**2
        cdef double x15 = 1/x0
        cdef double x241 = x15*x60
        cdef double x2 = X_2*chi_2
        cdef double x1 = X_1*chi_1
        cdef double x3 = x1 + x2
        cdef double x4 = x3**2
        cdef double x72 = x15*x4
        cdef double x63 = 1/r
        cdef double x71 = 2.0*x63 + 1.0
        cdef double x73 = x71*x72 + 1.0
        cdef double x218 = 1/x73
        cdef double x124 = 113485.217444961*r
        cdef double x123 = 148.04406601634*r
        cdef double x122 = 7704.0*r
        cdef double x121 = 128.0*r
        cdef double x120 = 7680.0*a6
        cdef double x118 = 8.0*r
        cdef double x18 = r**3
        cdef double x119 = 4.0*x0 + x118 + 2.0*x18 + 16.0
        cdef double x125 = (
            nu*(
                x120*(x11 + x119)
                + x121*(
                    13218.7851094412*r
                    + 8529.39255472061*x0
                    - 6852.34813868015*x11
                    + 4264.6962773603*x18
                    - 33722.4297811176
                )
                + x123*(3852.0*x0 + 349.0*x11 + x122 + 1926.0*x18 + 36400.0)
                + x124*(-x11 + x119)
            )
        )
        cdef double x102 = log(r)
        cdef double x116 = nu*x102
        cdef double x103 = 756.0*nu
        cdef double x113 = x103 + 1079.0
        cdef double x114 = x113*x18
        cdef double x112 = 588.0*nu + 1079.0
        cdef double x28 = r**5
        cdef double x23 = nu**2
        cdef double x115 = (
            -38842241.4769507*nu
            + 240.0*r*(-7466.27061066206*nu - 3024.0*x23 + 17264.0)
            + 1920.0*x0*x112
            + 480.0*x11*x113
            + 960.0*x114
            - 1882456.23663972*x23
            + 161280.0*x28
            + 13447680.0
        )
        cdef double x117 = x115*x116
        cdef double x111 = (
            x23*(
                -39321600.0*a6*(3.0*r + 59.0)
                + 745857848.115604*a6
                + 122635399361.987*r
                + 2064783811.32587*x0
                - 3089250703.76879*x11
                - 6178501407.53758*x18
                + 1426660551.8844*x28
                + 276057889687.011
            )
        )
        cdef double x109 = nu**3
        cdef double x110 = x109*(-163683964.822551*r - 17833256.898555*x0 - 1188987459.03162)
        cdef double x107 = x102**2
        cdef double x108 = x107*x23
        cdef double x106 = nu**4
        cdef double x126 = (
            1/(
                241555486248.807*x106
                + 67645734912.0*x108
                + 1120.0*x110
                + 7.0*x111
                + 32768.0*x117
                + 53760.0*x125
                + 13212057600.0*x28
            )
        )
        cdef double x127 = x11*x126
        cdef double x104 = 336.0*r + x103 + 407.0
        cdef double x105 = (
            2048.0*nu*x102*x104
            + 28.0*nu*(1920.0*a6 + 733955.307463037)
            - 7.0*r*(938918.400156317*nu - 185763.092693281*x23 - 245760.0)
            - 5416406.59541186*x23
            - 3440640.0
        )
        cdef double x128 = x105*x127
        cdef double x129 = Tidal1 + Tidal2 + 7680.0*x128
        cdef double x101 = X_2**2*chi_2**2*(1.0 - CES22)
        cdef double x100 = X_1**2*chi_1**2*(1.0 - CES21)
        cdef double x94 = X_2 - 1.0
        cdef double x33 = x1 - x2
        cdef double x51 = x3*x33
        cdef double x99 = x51*(CES21*x94 + X_2*(CES22 + 0.5) - 0.25)
        cdef double x97 = 4.0*nu
        cdef double x91 = 4.0*CES22
        cdef double x78 = 4.0*CES21
        cdef double x95 = -X_2*x91 + x78*x94
        cdef double x46 = x33**2
        cdef double x98 = x46*(x95 + x97 + 5.0)
        cdef double x96 = x4*(x95 + 13.0)
        cdef double x87 = 3.0*CES42
        cdef double x86 = 3.0*CES41
        cdef double x88 = x86 + x87
        cdef double x58 = 6.0*CES22
        cdef double x93 = x46*(CES21*(x58 + 4.0) - x88 + x91 - 8.0)
        cdef double x92 = x3**3*x33*(-6.0*CBS31 + 6.0*CBS32 + 4.0*CES21 - x86 + x87 - x91)
        cdef double x83 = CES41 + CES42
        cdef double x90 = x4*x46*(2.0*CES21*CES22 - x83)
        cdef double x49 = 12.0*CBS32
        cdef double x48 = 12.0*CBS31
        cdef double x89 = x3**4*(-18.0*CES21*CES22 + 8.0*CES21 + 8.0*CES22 - x48 - x49 - x88 + 32.0)
        cdef double x85 = x3*x33**3*(2.0*CBS31 - 2.0*CBS32 - CES41 + CES42)
        cdef double x54 = -4.0*CBS31
        cdef double x40 = 4.0*CBS32
        cdef double x55 = -x40 + x54
        cdef double x84 = x33**4*(-CES21*x58 - x55 - x83)
        cdef double x81 = 182.0*nu
        cdef double x75 = 34.0*X_2 + 69.0*nu - 34.0
        cdef double x79 = x75*x78
        cdef double x77 = 276.0*CES22*nu
        cdef double x82 = x51*(2.0*X_2*(68.0*CES22 + x81 - 409.0) - x77 + x79 - x81 + 409.0)
        cdef double x57 = CES22*X_2
        cdef double x80 = x46*(-741.0*nu + 196.0*x23 - 136.0*x57 + x77 + x79 + 115.0)
        cdef double x76 = x4*(12.0*CES21*x75 + 828.0*CES22*nu - 2881.0*nu - 408.0*x57 - 1167.0)
        cdef double x42 = 0.03125*x4
        cdef double x29 = 1/x28
        cdef double x19 = 1/x18
        cdef double x130 = (
            x12*(0.125*x96 + 0.125*x98 + x99)
            + x129
            + x19*(x100 + x101)
            + x29*(0.00223214285714286*x76 + 0.00669642857142857*x80 + 0.0133928571428571*x82)
            + x29*(x42*x93 + 0.046875*x84 + 0.1875*x85 + 0.015625*x89 + 0.1875*x90 + 0.0625*x92)
            + x72
        )
        cdef double x233 = x130*x218
        cdef double x175 = 102574080.0*x23 - 2119671837.36038
        cdef double x176 = 409207698.136075*nu + x175
        cdef double x177 = r*x176
        cdef double x173 = 14700.0*nu + 42911.0
        cdef double x174 = 5760.0*x173
        cdef double x172 = nu*(11592.0*nu + 69847.0)
        cdef double x178 = x102*(x0*x174 - 34560.0*x172 + x177 + 17902080.0*x18)
        cdef double x170 = (
            43393301259014.8*nu
            + 5927865218923.02*x109
            + 43133561885859.3*x23
            + 86618264430493.3*(1 - 0.496948781616935*nu)**2
            + 188440788778196.0
        )
        cdef double x171 = r*x170
        cdef double x169 = (
            nu*(-2510664218.28128*nu + 14515200.0*x109 - 42636451.6032331*x23 + 1002013764.01019)
        )
        cdef double x167 = -2675575.66847905*nu - 138240.0*x23 - 5278341.3229329
        cdef double x168 = x167*x18
        cdef double x165 = -630116198.873299*nu - 197773496.793534*x23 + 5805304367.87913
        cdef double x166 = x0*x165
        cdef double x164 = r*x107
        cdef double x213 = (
            5787938193408.0*x164 - 9216.0*x166 - 967680.0*x168 + 55296.0*x169 + x171 + 49152.0*x178
        )
        cdef double x214 = 1/x213
        cdef double x206 = 48.0*nu
        cdef double x208 = x91*(X_2*x206 - 38.0*X_2 + 87.0*nu + 16.0)
        cdef double x207 = x78*(X_2*(x206 - 38.0) - 135.0*nu + 22.0)
        cdef double x32 = X_1 - X_2
        cdef double x212 = x51*(-x207 - x208 - x32*(166.0*nu - 601.0))
        cdef double x209 = x207 - x208
        cdef double x211 = x46*(-773.0*nu - x209 + 4.0*x23 - 13.0)
        cdef double x210 = x4*(-2059.0*nu - x209 - 837.0)
        cdef double x199 = nu - 1.0
        cdef double x201 = x199*x91
        cdef double x200 = x199*x78
        cdef double x205 = x51*(14.0*X_2 + x200 - x201 - 7.0)
        cdef double x202 = x200 + x201
        cdef double x204 = x46*(x202 - x97 + 7.0)
        cdef double x203 = x4*(8.0*nu + x202 + 23.0)
        cdef double x196 = 5787938193408.0*x107
        cdef double x195 = 1822680546449.21*x23
        cdef double x185 = x102*(-516620178.136075*nu - r*x174 - 17902080.0*x0 - x175)
        cdef double x184 = x0*x23
        cdef double x183 = r*x173
        cdef double x181 = nu*x0
        cdef double x180 = nu*r
        cdef double x197 = (
            -12049908701745.2*nu
            + r*x195
            - 39476764256925.6*r
            + 5107745331375.71*x0
            + 6730497718123.02*x109
            + 10611661054566.2*x180
            + 2589101062873.81*x181
            - 326837426.241486*x183
            + 133772083200.0*x184
            - 49152.0*x185
            + x196
            + 80059249540278.2*x23
            + 275059053208689.0
        )
        cdef double x215 = (
            r*x129*x197*x214
            + x12*(0.015625*x210 + 0.015625*x211 + 0.03125*x212)
            + x19*(0.1875*x203 + 0.1875*x204 + 0.375*x205)
            + x72
        )
        cdef double x198 = 1/x197
        cdef double x162 = (
            0.000130208333333333*Tidal1 + 0.000130208333333333*Tidal2 + x128 + 0.000130208333333333*x72
        )
        cdef double x194 = x162**(-2)
        cdef double x193 = prst**2
        cdef double x7 = x0 + x4
        cdef double x192 = x7**2
        cdef double x216 = x192*x193*x194*x198*x213*x215
        cdef double x189 = 14.0*CES21 + 14.0*CES22
        cdef double x188 = 8.0*x23
        cdef double x190 = (
            0.15625*x4*(nu*(x189 + 5.0) - x188*(CES21 + CES22 + 2.0) + 5.0)
            - 0.15625*x46*(nu*(13.0 - x189) + x23*(8.0*CES21 + 8.0*CES22 - 4.0) + 3.0)
            - 0.3125*x51*(
                X_2*(36.0*nu - 2.0) - 2.0*nu*(7.0*CES21 - 7.0*CES22 + 9.0) + x188*(CES21 - CES22) + 1.0
            )
        )
        cdef double x182 = r*x23
        cdef double x186 = (
            -0.0438084424460039*nu
            - 0.143521050466841*r
            + 0.0185696317637669*x0
            + 0.0210425293255724*x107
            + 0.0244692826489756*x109
            + 0.0385795738434214*x180
            + 0.00941289164152486*x181
            + 0.00662650629087394*x182
            - 1.18824456940711e-6*x183
            + 0.000486339502879429*x184
            - 1.78696172427774e-10*x185
            + 0.291062041428379*x23
            + 1
        )
        cdef double x187 = x186**(-2)
        cdef double x179 = (
            (
                x164
                - 1.59227685093395e-9*x166
                - 1.67189069348064e-7*x168
                + 9.55366110560367e-9*x169
                + 1.72773095804465e-13*x171
                + 8.49214320498104e-9*x178
            )**2
        )
        cdef double x163 = x162**(-4)
        cdef double x161 = x7**4
        cdef double x131 = prst**4
        cdef double x191 = x131*x161*x163*x179*x187*x190
        cdef double x160 = r**(-13)
        cdef double x156 = r*x4
        cdef double x154 = r + 2.0
        cdef double x157 = x11 + x154*x156
        cdef double x158 = 1/x157
        cdef double x20 = pphi**2
        cdef double x159 = x158*x20*x63
        cdef double x155 = x154*x4
        cdef double x153 = x12*x131
        cdef double x152 = (
            nu*(452.542166996693 - 51.6952380952381*x102)
            + 602.318540416564*x109
            + x23*(118.4*x102 - 1796.13660498019)
        )
        cdef double x150 = 1.38977750996128*nu - 6.0*x106 + 3.42857142857143*x109 + 3.33842023648322*x23
        cdef double x138 = prst**8
        cdef double x151 = x138*x150
        cdef double x134 = prst**6
        cdef double x149 = x134*x19
        cdef double x148 = -33.9782122170436*nu - 14.0*x106 + 188.0*x109 - 89.5298327361234*x23
        cdef double x146 = -2.78300763695006*nu + 6.0*x109 - 5.4*x23
        cdef double x147 = x146*x15
        cdef double x145 = x131*x19
        cdef double x144 = 92.7110442849544*nu + 10.0*x109 - 131.0*x23
        cdef double x142 = 8.0*nu - 6.0*x23
        cdef double x143 = x142*x15
        cdef double x141 = 0.121954868780449*x138
        cdef double x140 = nu*x63
        cdef double x139 = nu*x138
        cdef double x137 = r**(-2.5)
        cdef double x135 = r**(-3.5)
        cdef double x136 = nu*x135
        cdef double x132 = r**(-4.5)
        cdef double x133 = nu*x132
        cdef double x64 = x15*x20
        cdef double x217 = (
            147.443752990146*x131*x133
            + x131*x143
            - 11.3175085791863*x134*x136
            + x134*x147
            + 1.48275342024365*x137*x139
            + x140*x141
            + x144*x145
            + x148*x149
            + x15*x151
            + x152*x153
            - x155*x159
            + 1.27277314139085e-19*x160*x191
            + 1.69542100694444e-8*x216*x29
            + x64
            + 1.0
        )
        cdef double x234 = 0.5*(x217*x233)**(-0.5)
        cdef double x240 = x233*x234
        cdef double x56 = CES21*X_2
        cdef double x50 = x48 - x49
        cdef double x47 = 0.03125*x46
        cdef double x44 = -15.0*CES22
        cdef double x43 = 3.0*CES21
        cdef double x38 = 9.0*CES21
        cdef double x59 = (
            x33*(
                0.0104166666666667*x4*(-27.0*CES22 - 44.0*X_2 + x38 + x50 + 18.0*x56 + 18.0*x57 + 22.0)
                + x47*(-X_2*x58 + 12.0*X_2 - x38 - x40 - x44 - x54 - 6.0*x56 - 6.0)
                + 0.0625*x51*(-3.0*CES22 - x43 - x55 - 2.0)
            )
        )
        cdef double x70 = x15*x59
        cdef double x45 = 12.0*X_2
        cdef double x41 = 4.0*CBS31 + x40
        cdef double x39 = 9.0*CES22 + x38
        cdef double x52 = (
            x42*(x39 + x41 - 34.0)
            + x47*(-CES21*x45 + CES22*x45 + x41 - x43 + x44 + 10.0)
            + 0.0208333333333333*x51*(x32*(x39 - 8.0) + x50)
        )
        cdef double x69 = x15*x52
        cdef double x67 = x32*x33
        cdef double x27 = pphi**4
        cdef double x65 = x12*x27
        cdef double x37 = -0.3515625*nu + 0.29296875*x23 - 0.41015625
        cdef double x36 = -0.2734375*nu - 0.798177083333333*x23 - 0.23046875
        cdef double x35 = 0.46875 - 0.28125*nu
        cdef double x34 = 0.34375*nu + 0.09375
        cdef double x21 = x19*x20
        cdef double x68 = (
            x67*(
                x15*(-0.03125*nu + 0.536458333333333*x23 + 0.078125)
                + x21*x36
                + x34*x63
                + x35*x64
                + x37*x65
                + 0.25
            )
        )
        cdef double x26 = 0.5859375*nu + 1.34765625*x23 + 0.41015625
        cdef double x24 = -2.0859375*nu - 2.07161458333333*x23 + 0.23046875
        cdef double x17 = -1.40625*nu - 0.46875
        cdef double x16 = 0.71875*nu - 0.09375
        cdef double x66 = (
            x3*(
                x15*(-5.53125*nu + 0.567708333333333*x23 - 0.078125)
                + x16*x63
                + x17*x64
                + x21*x24
                + x26*x65
                + 1.75
            )
        )
        cdef double x14 = dSO*nu
        cdef double x62 = x14*x19
        cdef double x61 = x19*x60
        cdef double x10 = pphi*x3
        cdef double x6 = 2.0*r
        cdef double x8 = r*(-x6 + x7)
        cdef double x5 = 2.0*x4
        cdef double x9 = 1/(2.0*x0 + x5 + x8)
        cdef double dHeffdpphi = (
            x240*(2.0*pphi*x15 - pphi*x154*x158*x5*x63)
            + x9*(
                pphi*x67*(x241*x35 + x242*x37 + x36*x61)
                + x10*(x17*x241 + x24*x61 + x242*x26)
                + x3*x62
                + x3*x69
                + x66
                + x68
                + x70
            )
        )
        cdef double x236 = prst**5
        cdef double x239 = 6.0*x236
        cdef double x235 = prst**3
        cdef double x238 = 4.0*x235
        cdef double x237 = prst**7
        cdef double x232 = x213*x29
        cdef double dHeffdpr = (
            x240*(
                11.8620273619492*nu*x137*x237
                + 3.39084201388889e-8*prst*x192*x194*x198*x215*x232
                + x12*x152*x238
                + 589.775011960583*x133*x235
                - 67.9050514751178*x136*x236
                + 0.975638950243592*x140*x237
                + x143*x238
                + x144*x19*x238
                + x147*x239
                + x148*x19*x239
                + 8.0*x15*x150*x237
                + 5.09109256556341e-19*x160*x161*x163*x179*x187*x190*x235
            )
        )
        cdef double dHeffdphi = 0
        cdef double x223 = 6.0*x0 + x118 + 8.0
        cdef double x222 = 4.0*x18
        cdef double x221 = x23*x63
        cdef double x224 = (
            1.31621673590926e-19*x105*x11*(
                53760.0*nu*(
                    3740417.71815805*r
                    + 2115968.85907902*x0
                    - 938918.400156317*x11
                    + x120*(x222 + x223)
                    + x121*(
                        17058.7851094412*r + 12794.0888320809*x0 - 27409.3925547206*x18 + 13218.7851094412
                    )
                    + x123*(5778.0*x0 + x122 + 1396.0*x18 + 7704.0)
                    + x124*(-x222 + x223)
                    + 1057984.42953951*x18
                    + 2888096.47013111
                )
                + 135291469824.0*x102*x221
                + 1120.0*x109*(-35666513.7971099*r - 163683964.822551)
                + 66060288000.0*x11
                + 32768.0*x115*x140
                + 32768.0*x116*(
                    -1791904.9465589*nu
                    + 3840.0*r*x112
                    + 2880.0*x0*x113
                    + 806400.0*x11
                    + 1920.0*x114
                    - 725760.0*x23
                    + 4143360.0
                )
                + 7.0*x23*(
                    -117964800.0*a6
                    + 4129567622.65173*r
                    - 18535504222.6128*x0
                    + 7133302759.42198*x11
                    - 12357002815.0752*x18
                    + 122635399361.987
                )
            )/(
                x106
                + 0.28004222119933*x108
                + 4.63661586574928e-9*x110
                + 2.8978849160933e-11*x111
                + 1.35654132757922e-7*x117
                + 2.22557561555966e-7*x125
                + 0.0546957463279941*x28
            )**2
        )
        cdef double x220 = (
            -6572428.80109422*nu + 2048.0*x104*x140 + 688128.0*x116 + 1300341.64885296*x23 + 1720320.0
        )
        cdef double x230 = dTidal1 + dTidal2 + 30720.0*x105*x126*x18 + 7680.0*x127*x220 - x224
        cdef double x74 = x19*x5
        cdef double x231 = x230 - x74
        cdef double x229 = x193*x215
        cdef double x225 = 11575876386816.0*x102
        cdef double x228 = (
            -18432.0*r*x165
            - 2903040.0*x0*x167
            + 49152.0*x102*(53706240.0*x0 + x176 + 11520.0*x183)
            + x170
            + x196
            + x225
            + x63*(283115520.0*x0*x173 - 1698693120.0*x172 + 49152.0*x177 + 879923036160.0*x18)
        )
        cdef double x227 = x131*x160*x161*x179*x190
        cdef double x226 = (
            5807150888816.34*nu
            + 10215490662751.4*r
            + 6291456.0*x102*(661500.0*nu + 279720.0*r + 1930995.0)
            + 5178202125747.62*x180
            + 267544166400.0*x182
            + x195
            + x225*x63
            - x63*(
                -25392914995744.3*nu
                - 879923036160.0*x0
                - 283115520.0*x183
                - 5041721180160.0*x23
                + 104186110149937.0
            )
            - 53501685054374.1
        )
        cdef double x219 = r**(-6)
        cdef double x53 = 2.0*x19
        cdef double x30 = 4.0*x29
        cdef double x31 = x27*x30
        cdef double x13 = 3.0*x12
        cdef double x25 = x13*x20
        cdef double x22 = 2.0*x21
        cdef double dHeffdr = (
            x234*(
                -x130*x217*(-x12*x5 - x71*x74)/x73**2
                + x217*x218*(
                    dTidal1
                    + dTidal2
                    + 30720.0*x105*x126*x18
                    + 7680.0*x11*x126*x220
                    - x12*(3.0*x100 + 3.0*x101)
                    - x219*(0.0111607142857143*x76 + 0.0334821428571429*x80 + 0.0669642857142857*x82)
                    - x219*(
                        0.15625*x4*x93 + 0.234375*x84 + 0.9375*x85 + 0.078125*x89 + 0.9375*x90 + 0.3125*x92
                    )
                    - x224
                    - x29*(0.5*x96 + 0.5*x98 + 4.0*x99)
                    - x74
                )
                + x233*(
                    -663.496888455656*nu*r**(-5.5)*x131
                    + 39.6112800271521*nu*x132*x134
                    - nu*x141*x15
                    + x12*x131*(-51.6952380952381*x140 + 118.4*x221)
                    + 6.78168402777778e-8*x12*x193*x194*x198*x213*x215*x7
                    - x13*x134*x148
                    - x131*x152*x30
                    + 7.59859378406358e-45*x131*x160*x161*x163*x187*x190*x213*x228
                    - 3.70688355060912*x135*x139
                    - 2.0*x142*x145
                    - 3.0*x144*x153
                    - 2.0*x146*x149
                    + x15*x154*x158*x20*x4
                    - x151*x53
                    + x154*x20*x4*x63*(x155 + x156 + x222)/x157**2
                    - x159*x4
                    - 9.25454462627843e-34*x163*x226*x227/x186**3
                    - 2.24091649004576e-37*x187*x192*x194*x213*x226*x229*x29
                    + 1.69542100694444e-8*x192*x193*x194*x198*x213*x29*(
                        r*x129*x214*x226
                        - 2.98505426338587e-26*r*x129*x197*x228/x179
                        + r*x197*x214*x230
                        - x12*(0.5625*x203 + 0.5625*x204 + 1.125*x205)
                        + x129*x197*x214
                        - x29*(0.0625*x210 + 0.0625*x211 + 0.125*x212)
                        - x74
                    )
                    + 1.69542100694444e-8*x192*x193*x194*x198*x215*x228*x29
                    - 8.47710503472222e-8*x216*x219
                    - x22
                    - 4.41515887225116e-12*x192*x198*x229*x231*x232/x162**3
                    - 6.62902677807736e-23*x187*x227*x231/x162**5
                    + 1.01821851311268e-18*x131*x163*x179*x187*x190*x7**3/r**12
                    - 1.65460508380811e-18*x191/r**14
                )
            )
            + x9*(
                pphi*x3*(
                    -x15*x16
                    - x17*x22
                    - x19*(-11.0625*nu + 1.13541666666667*x23 - 0.15625)
                    - x24*x25
                    - x26*x31
                )
                + pphi*x32*x33*(
                    -x15*x34 - x19*(-0.0625*nu + 1.07291666666667*x23 + 0.15625) - x22*x35 - x25*x36 - x31*x37
                )
                - x10*x13*x14
                - x10*x52*x53
                - x59*x61
            )
            - 0.25*(
                r*(x6 - 2.0) + x6 + x7
            )*(
                pphi*x66 + pphi*x68 + pphi*x70 + x10*x62 + x10*x69
            )/(
                x7 + 0.5*x8
            )**2
        )
        cdef double d2Heffdprdpphi = 0
        cdef double d2Heffdphidpphi = 0
        cdef double d2Heffdphidpr = 0
        cdef double y82 = 1/r
        cdef double y392 = pphi*y82
        cdef double y2 = X_2*chi_2
        cdef double y1 = X_1*chi_1
        cdef double y3 = y1 + y2
        cdef double y4 = y3**2
        cdef double y116 = r*y4
        cdef double y114 = r + 2.0
        cdef double y19 = r**4
        cdef double y117 = y114*y116 + y19
        cdef double y118 = 1/y117
        cdef double y5 = 2.0*y4
        cdef double y342 = y118*y5
        cdef double y393 = y342*y392
        cdef double y0 = r**2
        cdef double y63 = 1/y0
        cdef double y394 = 2.0*pphi*y63 - y114*y393
        cdef double y20 = 1/y19
        cdef double y391 = 4.0*y20
        cdef double y88 = pphi*y63
        cdef double y390 = 2.0*y88
        cdef double y386 = pphi**3
        cdef double y23 = nu**2
        cdef double y33 = -0.3515625*nu + 0.29296875*y23 - 0.41015625
        cdef double y389 = y33*y386
        cdef double y11 = r**5
        cdef double y12 = 1/y11
        cdef double y388 = 16.0*y12
        cdef double y29 = 0.5859375*nu + 1.34765625*y23 + 0.41015625
        cdef double y387 = y29*y386
        cdef double y15 = r**3
        cdef double y16 = 1/y15
        cdef double y74 = pphi*y16
        cdef double y385 = 4.0*y74
        cdef double y139 = y4*y63
        cdef double y89 = 2.0*y82 + 1.0
        cdef double y242 = y139*y89 + 1.0
        cdef double y266 = 1/y242
        cdef double y240 = X_2**2*chi_2**2*(1.0 - CES22)
        cdef double y239 = X_1**2*chi_1**2*(1.0 - CES21)
        cdef double y233 = X_2 - 1.0
        cdef double y36 = y1 - y2
        cdef double y52 = y3*y36
        cdef double y238 = y52*(CES21*y233 + X_2*(CES22 + 0.5) - 0.25)
        cdef double y192 = 4.0*CES22
        cdef double y190 = 4.0*CES21
        cdef double y234 = -X_2*y192 + y190*y233
        cdef double y196 = 4.0*nu
        cdef double y47 = y36**2
        cdef double y237 = y47*(y196 + y234 + 5.0)
        cdef double y235 = y234 + 13.0
        cdef double y236 = y235*y4
        cdef double y227 = 3.0*CES42
        cdef double y226 = 3.0*CES41
        cdef double y228 = y226 + y227
        cdef double y58 = 6.0*CES22
        cdef double y232 = y47*(CES21*(y58 + 4.0) + y192 - y228 - 8.0)
        cdef double y231 = y3**3*y36*(-6.0*CBS31 + 6.0*CBS32 + 4.0*CES21 - y192 - y226 + y227)
        cdef double y223 = CES41 + CES42
        cdef double y230 = y4*y47*(2.0*CES21*CES22 - y223)
        cdef double y50 = 12.0*CBS32
        cdef double y49 = 12.0*CBS31
        cdef double y229 = y3**4*(-18.0*CES21*CES22 + 8.0*CES21 + 8.0*CES22 - y228 - y49 - y50 + 32.0)
        cdef double y225 = y3*y36**3*(2.0*CBS31 - 2.0*CBS32 - CES41 + CES42)
        cdef double y54 = -4.0*CBS31
        cdef double y41 = 4.0*CBS32
        cdef double y55 = -y41 + y54
        cdef double y224 = y36**4*(-CES21*y58 - y223 - y55)
        cdef double y221 = 182.0*nu
        cdef double y216 = 34.0*X_2 + 69.0*nu - 34.0
        cdef double y219 = y190*y216
        cdef double y218 = 276.0*CES22*nu
        cdef double y222 = y52*(2.0*X_2*(68.0*CES22 + y221 - 409.0) - y218 + y219 - y221 + 409.0)
        cdef double y57 = CES22*X_2
        cdef double y220 = y47*(-741.0*nu + y218 + y219 + 196.0*y23 - 136.0*y57 + 115.0)
        cdef double y217 = y4*(12.0*CES21*y216 + 828.0*CES22*nu - 2881.0*nu - 408.0*y57 - 1167.0)
        cdef double y144 = 756.0*nu
        cdef double y159 = 336.0*r + y144 + 407.0
        cdef double y111 = log(r)
        cdef double y160 = (
            2048.0*nu*y111*y159
            + 28.0*nu*(1920.0*a6 + 733955.307463037)
            - 7.0*r*(938918.400156317*nu - 185763.092693281*y23 - 245760.0)
            - 5416406.59541186*y23
            - 3440640.0
        )
        cdef double y156 = 113485.217444961*r
        cdef double y155 = 148.04406601634*r
        cdef double y154 = 7704.0*r
        cdef double y153 = 128.0*r
        cdef double y152 = 7680.0*a6
        cdef double y150 = 8.0*r
        cdef double y151 = 4.0*y0 + 2.0*y15 + y150 + 16.0
        cdef double y157 = (
            nu*(
                y152*(y151 + y19)
                + y153*(
                    13218.7851094412*r
                    + 8529.39255472061*y0
                    + 4264.6962773603*y15
                    - 6852.34813868015*y19
                    - 33722.4297811176
                )
                + y155*(3852.0*y0 + 1926.0*y15 + y154 + 349.0*y19 + 36400.0)
                + y156*(y151 - y19)
            )
        )
        cdef double y148 = nu*y111
        cdef double y145 = y144 + 1079.0
        cdef double y146 = y145*y15
        cdef double y143 = 588.0*nu + 1079.0
        cdef double y147 = (
            -38842241.4769507*nu
            + 240.0*r*(-7466.27061066206*nu - 3024.0*y23 + 17264.0)
            + 1920.0*y0*y143
            + 161280.0*y11
            + 480.0*y145*y19
            + 960.0*y146
            - 1882456.23663972*y23
            + 13447680.0
        )
        cdef double y149 = y147*y148
        cdef double y142 = (
            y23*(
                -39321600.0*a6*(3.0*r + 59.0)
                + 745857848.115604*a6
                + 122635399361.987*r
                + 2064783811.32587*y0
                + 1426660551.8844*y11
                - 6178501407.53758*y15
                - 3089250703.76879*y19
                + 276057889687.011
            )
        )
        cdef double y104 = nu**3
        cdef double y141 = y104*(-163683964.822551*r - 17833256.898555*y0 - 1188987459.03162)
        cdef double y120 = y111**2
        cdef double y140 = y120*y23
        cdef double y108 = nu**4
        cdef double y158 = (
            1/(
                241555486248.807*y108
                + 13212057600.0*y11
                + 67645734912.0*y140
                + 1120.0*y141
                + 7.0*y142
                + 32768.0*y149
                + 53760.0*y157
            )
        )
        cdef double y161 = y158*y160
        cdef double y162 = y161*y19
        cdef double y208 = Tidal1 + Tidal2 + 7680.0*y162
        cdef double y43 = 0.03125*y4
        cdef double y241 = (
            y12*(0.00223214285714286*y217 + 0.00669642857142857*y220 + 0.0133928571428571*y222)
            + y12*(0.046875*y224 + 0.1875*y225 + 0.015625*y229 + 0.1875*y230 + 0.0625*y231 + y232*y43)
            + y139
            + y16*(y239 + y240)
            + y20*(0.125*y236 + 0.125*y237 + y238)
            + y208
        )
        cdef double y326 = y241*y266
        cdef double y214 = 1.69542100694444e-8*y12
        cdef double y133 = 102574080.0*y23 - 2119671837.36038
        cdef double y134 = 409207698.136075*nu + y133
        cdef double y135 = r*y134
        cdef double y131 = 14700.0*nu + 42911.0
        cdef double y132 = 5760.0*y131
        cdef double y130 = nu*(11592.0*nu + 69847.0)
        cdef double y136 = y111*(y0*y132 - 34560.0*y130 + y135 + 17902080.0*y15)
        cdef double y127 = (1 - 0.496948781616935*nu)**2
        cdef double y128 = (
            43393301259014.8*nu
            + 5927865218923.02*y104
            + 86618264430493.3*y127
            + 43133561885859.3*y23
            + 188440788778196.0
        )
        cdef double y129 = r*y128
        cdef double y126 = (
            nu*(-2510664218.28128*nu + 14515200.0*y104 - 42636451.6032331*y23 + 1002013764.01019)
        )
        cdef double y124 = -2675575.66847905*nu - 138240.0*y23 - 5278341.3229329
        cdef double y125 = y124*y15
        cdef double y122 = -630116198.873299*nu - 197773496.793534*y23 + 5805304367.87913
        cdef double y123 = y0*y122
        cdef double y121 = r*y120
        cdef double y206 = (
            5787938193408.0*y121 - 9216.0*y123 - 967680.0*y125 + 55296.0*y126 + y129 + 49152.0*y136
        )
        cdef double y207 = 1/y206
        cdef double y209 = y207*y208
        cdef double y199 = 48.0*nu
        cdef double y201 = y192*(X_2*y199 - 38.0*X_2 + 87.0*nu + 16.0)
        cdef double y200 = y190*(X_2*(y199 - 38.0) - 135.0*nu + 22.0)
        cdef double y35 = X_1 - X_2
        cdef double y205 = y52*(-y200 - y201 - y35*(166.0*nu - 601.0))
        cdef double y202 = y200 - y201
        cdef double y204 = y47*(-773.0*nu - y202 + 4.0*y23 - 13.0)
        cdef double y203 = y4*(-2059.0*nu - y202 - 837.0)
        cdef double y189 = nu - 1.0
        cdef double y193 = y189*y192
        cdef double y191 = y189*y190
        cdef double y198 = y52*(14.0*X_2 + y191 - y193 - 7.0)
        cdef double y194 = y191 + y193
        cdef double y197 = y47*(y194 - y196 + 7.0)
        cdef double y195 = y4*(8.0*nu + y194 + 23.0)
        cdef double y185 = 5787938193408.0*y120
        cdef double y184 = 1822680546449.21*y23
        cdef double y170 = y111*(-516620178.136075*nu - r*y132 - 17902080.0*y0 - y133)
        cdef double y169 = y0*y23
        cdef double y168 = r*y131
        cdef double y166 = nu*y0
        cdef double y165 = nu*r
        cdef double y186 = (
            -12049908701745.2*nu
            + r*y184
            - 39476764256925.6*r
            + 5107745331375.71*y0
            + 6730497718123.02*y104
            + 10611661054566.2*y165
            + 2589101062873.81*y166
            - 326837426.241486*y168
            + 133772083200.0*y169
            - 49152.0*y170
            + y185
            + 80059249540278.2*y23
            + 275059053208689.0
        )
        cdef double y210 = (
            r*y186*y209
            + y139
            + y16*(0.1875*y195 + 0.1875*y197 + 0.375*y198)
            + y20*(0.015625*y203 + 0.015625*y204 + 0.03125*y205)
        )
        cdef double y211 = y206*y210
        cdef double y187 = 1/y186
        cdef double y163 = (
            0.000130208333333333*Tidal1 + 0.000130208333333333*Tidal2 + 0.000130208333333333*y139 + y162
        )
        cdef double y183 = y163**(-2)
        cdef double y182 = prst**2
        cdef double y188 = y182*y183*y187
        cdef double y212 = y188*y211
        cdef double y7 = y0 + y4
        cdef double y181 = y7**2
        cdef double y213 = y181*y212
        cdef double y179 = y7**4
        cdef double y178 = r**(-13)
        cdef double y180 = y178*y179
        cdef double y174 = 14.0*CES21 + 14.0*CES22
        cdef double y173 = 8.0*y23
        cdef double y175 = (
            0.15625*y4*(nu*(y174 + 5.0) - y173*(CES21 + CES22 + 2.0) + 5.0)
            - 0.15625*y47*(nu*(13.0 - y174) + y23*(8.0*CES21 + 8.0*CES22 - 4.0) + 3.0)
            - 0.3125*y52*(
                X_2*(36.0*nu - 2.0) - 2.0*nu*(7.0*CES21 - 7.0*CES22 + 9.0) + y173*(CES21 - CES22) + 1.0
            )
        )
        cdef double y167 = r*y23
        cdef double y171 = (
            -0.0438084424460039*nu
            - 0.143521050466841*r
            + 0.0185696317637669*y0
            + 0.0244692826489756*y104
            + 0.0210425293255724*y120
            + 0.0385795738434214*y165
            + 0.00941289164152486*y166
            + 0.00662650629087394*y167
            - 1.18824456940711e-6*y168
            + 0.000486339502879429*y169
            - 1.78696172427774e-10*y170
            + 0.291062041428379*y23
            + 1
        )
        cdef double y172 = y171**(-2)
        cdef double y164 = y163**(-4)
        cdef double y93 = prst**4
        cdef double y176 = y164*y172*y175*y93
        cdef double y137 = (
            y121
            - 1.59227685093395e-9*y123
            - 1.67189069348064e-7*y125
            + 9.55366110560367e-9*y126
            + 1.72773095804465e-13*y129
            + 8.49214320498104e-9*y136
        )
        cdef double y138 = y137**2
        cdef double y177 = y138*y176
        cdef double y18 = pphi**2
        cdef double y119 = y118*y18*y82
        cdef double y115 = y114*y4
        cdef double y113 = y20*y93
        cdef double y112 = (
            nu*(452.542166996693 - 51.6952380952381*y111)
            + 602.318540416564*y104
            + y23*(118.4*y111 - 1796.13660498019)
        )
        cdef double y98 = prst**8
        cdef double y110 = y98*(1.38977750996128*nu + 3.42857142857143*y104 - 6.0*y108 + 3.33842023648322*y23)
        cdef double y96 = prst**6
        cdef double y109 = y96*(-33.9782122170436*nu + 188.0*y104 - 14.0*y108 - 89.5298327361234*y23)
        cdef double y107 = y96*(-2.78300763695006*nu + 6.0*y104 - 5.4*y23)
        cdef double y105 = 92.7110442849544*nu + 10.0*y104 - 131.0*y23
        cdef double y106 = y105*y93
        cdef double y102 = 8.0*nu - 6.0*y23
        cdef double y103 = y102*y93
        cdef double y101 = 0.121954868780449*y98
        cdef double y100 = nu*y82
        cdef double y99 = nu*y98
        cdef double y97 = nu*y96
        cdef double y95 = r**(-3.5)
        cdef double y94 = nu*y93
        cdef double y92 = r**(-4.5)
        cdef double y83 = y18*y63
        cdef double y215 = (
            1.48275342024365*r**(-2.5)*y99
            + y100*y101
            + y103*y63
            + y106*y16
            + y107*y63
            + y109*y16
            + y110*y63
            + y112*y113
            - y115*y119
            + 1.27277314139085e-19*y177*y180
            + y213*y214
            + y83
            + 147.443752990146*y92*y94
            - 11.3175085791863*y95*y97
            + 1.0
        )
        cdef double y328 = y215*y326
        cdef double y384 = y328**(-0.5)
        cdef double y258 = 4.0*y15
        cdef double y274 = y115 + y116 + y258
        cdef double y346 = y274*y5
        cdef double y273 = y117**(-2)
        cdef double y347 = y114*y273*y346
        cdef double y343 = y114*y342
        cdef double y262 = y160*y19
        cdef double y263 = 1.31621673590926e-19*y262
        cdef double y259 = 6.0*y0 + y150 + 8.0
        cdef double y260 = (
            nu*(
                3740417.71815805*r
                + 2115968.85907902*y0
                + 1057984.42953951*y15
                + y152*(y258 + y259)
                + y153*(17058.7851094412*r + 12794.0888320809*y0 - 27409.3925547206*y15 + 13218.7851094412)
                + y155*(5778.0*y0 + 1396.0*y15 + y154 + 7704.0)
                + y156*(-y258 + y259)
                - 938918.400156317*y19
                + 2888096.47013111
            )
        )
        cdef double y257 = y100*y147
        cdef double y254 = y0*y145
        cdef double y255 = (
            -1791904.9465589*nu
            + 3840.0*r*y143
            + 1920.0*y146
            + 806400.0*y19
            - 725760.0*y23
            + 2880.0*y254
            + 4143360.0
        )
        cdef double y256 = y148*y255
        cdef double y253 = (
            y23*(
                -117964800.0*a6
                + 4129567622.65173*r
                - 18535504222.6128*y0
                - 12357002815.0752*y15
                + 7133302759.42198*y19
                + 122635399361.987
            )
        )
        cdef double y251 = y111*y82
        cdef double y252 = y23*y251
        cdef double y250 = y104*(-35666513.7971099*r - 163683964.822551)
        cdef double y248 = (
            y108
            + 0.0546957463279941*y11
            + 0.28004222119933*y140
            + 4.63661586574928e-9*y141
            + 2.8978849160933e-11*y142
            + 1.35654132757922e-7*y149
            + 2.22557561555966e-7*y157
        )
        cdef double y249 = y248**(-2)
        cdef double y261 = (
            y249*(
                66060288000.0*y19
                + 1120.0*y250
                + 135291469824.0*y252
                + 7.0*y253
                + 32768.0*y256
                + 32768.0*y257
                + 53760.0*y260
            )
        )
        cdef double y264 = y261*y263
        cdef double y246 = 2048.0*y159
        cdef double y247 = -6572428.80109422*nu + y100*y246 + 688128.0*y148 + 1300341.64885296*y23 + 1720320.0
        cdef double y90 = y16*y5
        cdef double y80 = 0.5*y4
        cdef double y26 = r**(-6)
        cdef double y265 = (
            dTidal1
            + dTidal2
            - y12*(y235*y80 + 0.5*y237 + 4.0*y238)
            + 30720.0*y15*y158*y160
            + 7680.0*y158*y19*y247
            - y20*(3.0*y239 + 3.0*y240)
            - y26*(0.0111607142857143*y217 + 0.0334821428571429*y220 + 0.0669642857142857*y222)
            - y26*(0.234375*y224 + 0.9375*y225 + 0.078125*y229 + 0.9375*y230 + 0.3125*y231 + 0.15625*y232*y4)
            - y264
            - y90
        )
        cdef double y341 = y265*y266
        cdef double y243 = y242**(-2)
        cdef double y244 = y241*y243
        cdef double y91 = -y20*y5 - y89*y90
        cdef double y340 = y244*y91
        cdef double y329 = 0.25*y328**(-1.5)
        cdef double y321 = y186*y208
        cdef double y320 = 1/y138
        cdef double y322 = y320*y321
        cdef double y323 = 2.98505426338587e-26*r*y322
        cdef double y309 = y15*y161
        cdef double y308 = 7680.0*y19
        cdef double y307 = y158*y247
        cdef double y310 = dTidal1 + dTidal2 - y264 + y307*y308 + 30720.0*y309
        cdef double y298 = y0*y124
        cdef double y297 = r*y122
        cdef double y296 = 49152.0*y111
        cdef double y295 = 53706240.0*y0 + y134 + 11520.0*y168
        cdef double y293 = 283115520.0*y0*y131 - 1698693120.0*y130 + 49152.0*y135 + 879923036160.0*y15
        cdef double y294 = y293*y82
        cdef double y299 = (
            11575876386816.0*y111 + y128 + y185 + y294 + y295*y296 - 18432.0*y297 - 2903040.0*y298
        )
        cdef double y284 = (
            -25392914995744.3*nu
            - 879923036160.0*y0
            - 283115520.0*y168
            - 5041721180160.0*y23
            + 104186110149937.0
        )
        cdef double y285 = y284*y82
        cdef double y283 = y111*(661500.0*nu + 279720.0*r + 1930995.0)
        cdef double y281 = 11575876386816.0*y82
        cdef double y282 = y111*y281
        cdef double y280 = 267544166400.0*y23
        cdef double y279 = 5178202125747.62*nu
        cdef double y286 = (
            5807150888816.34*nu
            + r*y279
            + r*y280
            + 10215490662751.4*r
            + y184
            + y282
            + 6291456.0*y283
            - y285
            - 53501685054374.1
        )
        cdef double y324 = (
            r*y186*y207*y310
            + r*y207*y208*y286
            - y12*(0.0625*y203 + 0.0625*y204 + 0.125*y205)
            + y186*y207*y208
            - y20*(0.5625*y195 + 0.5625*y197 + 1.125*y198)
            - y299*y323
            - y90
        )
        cdef double y319 = 4.41515887225116e-12*y12
        cdef double y300 = y181*y182
        cdef double y301 = y211*y300
        cdef double y316 = y187*y301
        cdef double y315 = y163**(-3)
        cdef double y317 = y315*y316
        cdef double y311 = y310 - y90
        cdef double y318 = y311*y317
        cdef double y289 = y138*y175*y93
        cdef double y290 = y180*y289
        cdef double y313 = y172*y290
        cdef double y314 = 6.62902677807736e-23*y313
        cdef double y306 = y163**(-5)
        cdef double y312 = y306*y311
        cdef double y305 = 2.24091649004576e-37*y12
        cdef double y302 = y183*y301
        cdef double y303 = y172*y302
        cdef double y304 = y286*y303
        cdef double y291 = y164*y290
        cdef double y292 = 9.25454462627843e-34*y291
        cdef double y287 = y171**(-3)
        cdef double y288 = y286*y287
        cdef double y278 = y7**3
        cdef double y277 = r**(-12)
        cdef double y276 = y177*y179
        cdef double y275 = r**(-14)
        cdef double y272 = y112*y93
        cdef double y270 = 118.4*y23
        cdef double y271 = -51.6952380952381*nu*y82 + y270*y82
        cdef double y269 = nu*y63
        cdef double y268 = r**(-5.5)
        cdef double y72 = 2.0*y16
        cdef double y68 = 4.0*y12
        cdef double y65 = y16*y18
        cdef double y66 = 2.0*y65
        cdef double y61 = 3.0*y20
        cdef double y325 = (
            39.6112800271521*nu*y92*y96
            - y101*y269
            - y103*y72
            - 3.0*y105*y113
            - y107*y72
            - y109*y61
            - y110*y72
            + y114*y118*y18*y4*y63
            + y114*y18*y273*y274*y4*y82
            - y119*y4
            + 1.69542100694444e-8*y12*y181*y182*y183*y187*y206*y324
            + 1.69542100694444e-8*y12*y181*y182*y183*y187*y210*y299
            + 1.01821851311268e-18*y138*y164*y172*y175*y277*y278*y93
            + 7.59859378406358e-45*y164*y172*y175*y178*y179*y206*y299*y93
            + 6.78168402777778e-8*y182*y183*y187*y20*y206*y210*y7
            + y20*y271*y93
            - 8.47710503472222e-8*y213*y26
            - 663.496888455656*y268*y94
            - y272*y68
            - 1.65460508380811e-18*y275*y276
            - y288*y292
            - y304*y305
            - y312*y314
            - y318*y319
            - y66
            - 3.70688355060912*y95*y99
        )
        cdef double y267 = y215*y266
        cdef double y245 = y215*y244
        cdef double y327 = -y245*y91 + y265*y267 + y325*y326
        cdef double y51 = y49 - y50
        cdef double y48 = 0.03125*y47
        cdef double y46 = 12.0*X_2
        cdef double y45 = -15.0*CES22
        cdef double y44 = 3.0*CES21
        cdef double y42 = 4.0*CBS31 + y41
        cdef double y39 = 9.0*CES21
        cdef double y40 = 9.0*CES22 + y39
        cdef double y53 = (
            y43*(y40 + y42 - 34.0)
            + y48*(-CES21*y46 + CES22*y46 + y42 - y44 + y45 + 10.0)
            + 0.0208333333333333*y52*(y35*(y40 - 8.0) + y51)
        )
        cdef double y87 = y53*y63
        cdef double y28 = pphi**4
        cdef double y84 = y20*y28
        cdef double y70 = 0.34375*nu + 0.09375
        cdef double y37 = y35*y36
        cdef double y32 = -0.2734375*nu - 0.798177083333333*y23 - 0.23046875
        cdef double y31 = 0.46875 - 0.28125*nu
        cdef double y86 = (
            y37*(
                y31*y83
                + y32*y65
                + y33*y84
                + y63*(-0.03125*nu + 0.536458333333333*y23 + 0.078125)
                + y70*y82
                + 0.25
            )
        )
        cdef double y64 = 0.71875*nu - 0.09375
        cdef double y24 = -2.0859375*nu - 2.07161458333333*y23 + 0.23046875
        cdef double y17 = -1.40625*nu - 0.46875
        cdef double y85 = (
            y3*(
                y17*y83
                + y24*y65
                + y29*y84
                + y63*(-5.53125*nu + 0.567708333333333*y23 - 0.078125)
                + y64*y82
                + 1.75
            )
        )
        cdef double y14 = dSO*nu
        cdef double y81 = y14*y16
        cdef double y6 = 2.0*r
        cdef double y78 = r*(y6 - 2.0)
        cdef double y8 = r*(-y6 + y7)
        cdef double y76 = y7 + 0.5*y8
        cdef double y77 = y76**(-2)
        cdef double y79 = y77*(y6 + y7 + y78)
        cdef double y75 = 2.0*y74
        cdef double y73 = y53*y72
        cdef double y67 = y18*y61
        cdef double y34 = y28*y33
        cdef double y71 = (
            -y16*(-0.0625*nu + 1.07291666666667*y23 + 0.15625) - y31*y66 - y32*y67 - y34*y68 - y63*y70
        )
        cdef double y30 = y28*y29
        cdef double y69 = (
            -y16*(-11.0625*nu + 1.13541666666667*y23 - 0.15625) - y17*y66 - y24*y67 - y30*y68 - y63*y64
        )
        cdef double y62 = y14*y61
        cdef double y21 = 6.0*y20
        cdef double y60 = pphi*y21
        cdef double y56 = CES21*X_2
        cdef double y59 = (
            y36*(
                0.0104166666666667*y4*(-27.0*CES22 - 44.0*X_2 + y39 + y51 + 18.0*y56 + 18.0*y57 + 22.0)
                + y48*(-X_2*y58 + 12.0*X_2 - y39 - y41 - y45 - y54 - 6.0*y56 - 6.0)
                + 0.0625*y52*(-3.0*CES22 - y44 - y55 - 2.0)
            )
        )
        cdef double y38 = pphi*y37
        cdef double y10 = pphi*y3
        cdef double y9 = 1/(2.0*y0 + y5 + y8)
        cdef double d2Heffdrdpphi = (
            -y326*y327*y329*y394
            + 0.5*y384*(y326*(y343*y88 + y347*y392 - y385 - y393) - y340*y394 + y341*y394)
            - 0.25*y79*(
                y10*(y17*y390 + y24*y75 + y387*y391)
                + y3*y81
                + y3*y87
                + y38*(y31*y390 + y32*y75 + y389*y391)
                + y59*y63
                + y85
                + y86
            )
            + y9*(
                y10*(-y17*y385 - y24*y60 - y387*y388)
                - y3*y62
                + y3*y69
                - y3*y73
                + y37*y71
                + y38*(-y31*y385 - y32*y60 - y388*y389)
                - y59*y72
            )
        )
        cdef double d2Heffdrdpr = 0
        cdef double d2Heffdrdphi = 0
        cdef double d2Heffdpphi2 = 0
        cdef double d2Heffdpr2 = 0
        cdef double d2Heffdphi2 = 0
        cdef double y383 = y186*y207
        cdef double y381 = 5.97010852677174e-26*y299
        cdef double y382 = r*y320*y381
        cdef double y379 = y12*y324
        cdef double y380 = y206*y300*y379
        cdef double y378 = y206*y324
        cdef double y375 = y311*y315
        cdef double y376 = y187*y375
        cdef double y377 = 8.83031774450231e-12*y376
        cdef double y332 = y16*y4
        cdef double y374 = (
            (
                3.25520833333333e-5*dTidal1
                + 3.25520833333333e-5*dTidal2
                + 0.25*y19*y307
                - 4.28455968720463e-24*y261*y262
                + y309
                - 6.51041666666667e-5*y332
            )**2
        )
        cdef double y373 = y172*y312
        cdef double y369 = y172*y183*y286
        cdef double y372 = 4.48183298009152e-37*y369
        cdef double y364 = y210*y299
        cdef double y371 = y12*y300*y364
        cdef double y365 = y20*y7
        cdef double y370 = y182*y211*y365
        cdef double y368 = 1.69542100694444e-7*y26
        cdef double y362 = y181*y188
        cdef double y367 = y299*y362
        cdef double y366 = 1.35633680555556e-7*y188*y365
        cdef double y363 = y214*y362
        cdef double y359 = y206*y299
        cdef double y361 = y175*y180*y359*y93
        cdef double y360 = y176*y359
        cdef double y358 = (
            11614301777632.7*nu
            - 5806080.0*r*y124
            + 3645361092898.41*y23
            + y281
            + y282
            - y293*y63
            + y296*(169344000.0*nu + 107412480.0*r + 494334720.0)
            + y82*(
                40226753557568.7*nu
                + 5279538216960.0*y0
                + 1132462080.0*y168
                + 10083442360320.0*y23
                - 208372220299875.0
            )
            - 107003370108748.0
        )
        cdef double y357 = (
            (
                0.108541457767442*nu
                + 0.190937736865098*r
                + 0.0967857763822762*y165
                + 0.00500066803742898*y167
                + 0.0340677222520525*y23
                + 0.216364706551791*y251
                + 1.17593604642657e-7*y283
                - 1.86910000868887e-14*y285
                - 1
            )**2
        )
        cdef double y355 = y277*y278
        cdef double y356 = y289*y355
        cdef double y354 = y164*y288
        cdef double y352 = y179*y275
        cdef double y353 = y289*y352
        cdef double y350 = 11575876386816.0*y63
        cdef double y351 = (
            -y111*y350
            + 1759846072320.0*y111
            + y279
            + y280
            + y284*y63
            + y350
            + y82*(8323596288000.0*nu + 3519692144640.0*r + 24297540157440.0)
            + 10215490662751.4
        )
        cdef double y349 = y176*y180
        cdef double y348 = (
            (
                0.230275523363951*nu
                + 0.0314574421883811*y104
                + 2.60835248667179e-10*y111*y295
                + 0.0614297810037369*y111
                + 0.0307148905018684*y120
                + 0.459657725867659*y127
                + 0.228897162687159*y23
                + 5.30670671930296e-15*y294
                - 9.78132182501921e-11*y297
                - 1.54055818744053e-8*y298
                + 1
            )**2
        )
        cdef double y344 = y18*y82
        cdef double y345 = y273*y344
        cdef double y339 = 2.0*y325
        cdef double y336 = 12.0*r + 8.0
        cdef double y335 = 12.0*y0
        cdef double y334 = 135291469824.0*y23*y63
        cdef double y337 = (
            ddTidal1
            + ddTidal2
            + 92160.0*y0*y161
            - 1.05297338872741e-18*y15*y160*y261
            + 61440.0*y15*y307
            + y158*y308*(1376256.0*y100 - y246*y269)
            - 2.63243347181853e-19*y19*y247*y261
            - y249*y263*(
                53760.0*nu*(
                    8463875.43631609*r
                    + 6347906.57723707*y0
                    - 7511347.20125054*y15
                    + y152*(y335 + y336)
                    + y153*(25588.1776641618*r - 82228.1776641618*y0 + 17058.7851094412)
                    + y155*(11556.0*r + 4188.0*y0 + 7704.0)
                    + y156*(-y335 + y336)
                    + 7480835.43631609
                )
                + 65536.0*y100*y255
                - 39946495452.7631*y104
                - y111*y334
                - 32768.0*y147*y269
                + 32768.0*y148*(2257920.0*nu + 5760.0*r*y145 + 3225600.0*y15 + 5760.0*y254 + 4143360.0)
                + 264241152000.0*y15
                + 7.0*y23*(
                    -37071008445.2255*r - 37071008445.2255*y0 + 28533211037.6879*y15 + 4129567622.65173
                )
                + y334
            )
            + 1.99471718230171e-8*y262*(
                0.48828125*y19
                + 8.27842288547092e-9*y250
                + y252
                + 5.17401430341932e-11*y253
                + 2.42203000992063e-7*y256
                + 2.42203000992063e-7*y257
                + 3.97364298502604e-7*y260
            )**2/y248**3
        )
        cdef double y330 = y20*y4
        cdef double y331 = 6.0*y330
        cdef double y338 = y331 + y337
        cdef double y333 = r**(-7)
        cdef double y27 = 20.0*y26
        cdef double y13 = 12.0*y12
        cdef double y25 = y13*y18
        cdef double y22 = y18*y21
        cdef double d2Heffdr2 = (
            -y327**2*y329
            + 0.5*y384*(
                8.0*y215*y241*(-y330 - y332*y89)**2/y242**3
                - 2.0*y215*y243*y265*y91
                - y245*(y13*y4 + y331*y89)
                + y267*(
                    y12*(12.0*y239 + 12.0*y240)
                    + y26*(2.5*y236 + 2.5*y237 + 20.0*y238)
                    + y333*(0.0669642857142857*y217 + 0.200892857142857*y220 + 0.401785714285714*y222)
                    + y333*(
                        1.40625*y224 + 5.625*y225 + 0.46875*y229 + 5.625*y230 + 1.875*y231 + 0.9375*y232*y4
                    )
                    + y338
                )
                + y326*(
                    3649.23288650611*r**(-6.5)*y94
                    + 6.0*y102*y113
                    + y106*y13
                    + y107*y21
                    + y109*y13
                    + y110*y21
                    + y113*(51.6952380952381*nu*y63 - y270*y63)
                    + y115*y345*(y335 + y5)
                    - 32.0*y115*y344*(0.25*y115 + 0.25*y116 + y15)**2/y117**3
                    + 1.62760416666667e-6*y12*y164*y316*y374
                    + 1.16714400523217e-40*y12*y172*y286*y301*y375
                    - 6.103515625e-7*y12*y212*y7
                    - 8.0*y12*y271*y93
                    + 4.66406554828496e-24*y12*y287*y302*y357
                    + 1.35633680555556e-7*y16*y212
                    + 0.243909737560898*y16*y99
                    - 2.5455462827817e-17*y177*y178*y278
                    + 7.59859378406358e-45*y206*y349*y358
                    + y206*y363*(
                        r*y209*y351
                        + r*y337*y383
                        + 3.66275751433287e-10*r*y321*y348/y137**3
                        + y12*(2.25*y195 + 2.25*y197 + 4.5*y198)
                        - y186*y310*y382
                        + y207*y286*y310*y6
                        - y208*y286*y382
                        + 2.0*y209*y286
                        + y26*(0.3125*y203 + 0.3125*y204 + 0.625*y205)
                        + 2.0*y310*y383
                        - y322*y381
                        - y323*y358
                        + y331
                    )
                    + y210*y358*y363
                    - y210*y367*y368
                    + 5.08626302083333e-7*y213*y333
                    + y22
                    + 2.24091649004576e-36*y26*y304
                    + 4.41515887225116e-11*y26*y318
                    - 178.250760122184*y268*y97
                    + y27*y272
                    - y287*y292*y351
                    + 9.64015065237337e-37*y288*y290*y312
                    - y303*y305*y351
                    - y306*y314*y338
                    - y317*y319*y338
                    + y342*y83
                    - y343*y65
                    + y345*y346
                    - y347*y83
                    + 2.69825540021951e-16*y348*y349
                    - 1.97563438385653e-43*y352*y360
                    + 2.40618160283239e-32*y353*y354
                    + 1.72354696230011e-21*y353*y373
                    - 1.48072714020455e-32*y354*y356
                    - 1.10501271569469e-58*y354*y361
                    + 1.21577500545017e-43*y355*y360
                    - 1.06064428449238e-21*y356*y373
                    - 7.91520185839956e-48*y361*y373
                    - y362*y368*y378
                    + y364*y366
                    + y366*y378
                    + 3.39084201388889e-8*y367*y379
                    - 1.79273319203661e-36*y369*y370
                    - 3.53212709780093e-11*y370*y376
                    - y371*y372
                    - y371*y377
                    - y372*y380
                    - y377*y380
                    + 12.9740924271319*y92*y99
                    + 2.88925109089694e-20*y291*y357/y171**4
                    + 4.07287405245073e-17*y313*y374/y163**6
                    + 6.10931107867609e-18*y177*y181/r**11
                    + 2.31644711733135e-17*y276/r**15
                )
                - y339*y340
                + y339*y341
            )
            - 0.5*y79*(pphi*y3*y69 + pphi*y35*y36*y71 - y10*y62 - y10*y73 - y59*y75)
            + y9*(
                y10*y13*y14
                + y10*y21*y53
                + y10*(
                    y16*(1.4375*nu - 0.1875)
                    + y17*y22
                    + y20*(-33.1875*nu + 3.40625*y23 - 0.46875)
                    + y24*y25
                    + y27*y30
                )
                + y38*(
                    y16*(0.6875*nu + 0.1875)
                    + y20*(-0.1875*nu + 3.21875*y23 + 0.46875)
                    + y22*y31
                    + y25*y32
                    + y27*y34
                )
                + y59*y60
            )
            + (
                -1.5*r*y77 + 1.0*(r + 0.5*y0 + 0.5*y78 + y80)**2/y76**3
            )*(
                pphi*y85 + pphi*y86 + y10*y81 + y10*y87 + y59*y88
            )
        )
        # Evaluate Hamiltonian
        cdef double H, _xi
        H, _xi = self.__call__(q, p, chi_1, chi_2, m_1, m_2, verbose=False)
        cdef double _nuH = nu * H

        # Compute H Hessian
        cdef double d2Hdr2 = (-(dHeffdr**2/H**3)*(M2/nu) + d2Heffdr2/H) * M2 / nu
        cdef double d2Hdphi2 = (-(dHeffdphi**2/H**3)*(M2/nu) + d2Heffdphi2/H) * M2 / nu
        cdef double d2Hdpr2 = (-(dHeffdpr**2/H**3)*(M2/nu) + d2Heffdpr2/H) * M2 / nu
        cdef double d2Hdpphi2 = (-(dHeffdpphi**2/H**3)*(M2/nu) + d2Heffdpphi2/H) * M2 / nu
        cdef double d2Hdrdphi = (-(dHeffdr*dHeffdphi/H**3)*(M2/nu) + d2Heffdrdphi/H) * M2 / nu
        cdef double d2Hdrdpr = (-(dHeffdr*dHeffdpr/H**3)*(M2/nu) + d2Heffdrdpr/H) * M2 / nu
        cdef double d2Hdrdpphi = (-(dHeffdr*dHeffdpphi/H**3)*(M2/nu) + d2Heffdrdpphi/H) * M2 / nu
        cdef double d2Hdphidpr = (-(dHeffdphi*dHeffdpr/H**3)*(M2/nu) + d2Heffdphidpr/H) * M2 / nu
        cdef double d2Hdphidpphi = (-(dHeffdphi*dHeffdpphi/H**3)*(M2/nu) + d2Heffdphidpphi/H) * M2 / nu
        cdef double d2Hdprdpphi = (-(dHeffdpr*dHeffdpphi/H**3)*(M2/nu) + d2Heffdprdpphi/H) * M2 / nu

        # Return the hessian of H
        return (
            np.array(
                [
                    d2Hdr2,
                    d2Hdrdphi,
                    d2Hdrdpr,
                    d2Hdrdpphi,
                    d2Hdrdphi,
                    d2Hdphi2,
                    d2Hdphidpr,
                    d2Hdphidpphi,
                    d2Hdrdpr,
                    d2Hdphidpr,
                    d2Hdpr2,
                    d2Hdprdpphi,
                    d2Hdrdpphi,
                    d2Hdphidpphi,
                    d2Hdprdpphi,
                    d2Hdpphi2
                ]
            ).reshape(
                4, 4
            )
        )

    cpdef double csi(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2
    ):
        r"""
        Compute the tortoise factor \csi to convert between pr and prst.
        """
        cdef double r = q[0]
        # cdef double phi = q[1]

        cdef double _prst = p[0]
        cdef double _pphi = p[1]

        # Extra quantities used in the Hamiltonian
        cdef double M = self.EOBpars.p_params.M
        cdef double _M2 = M * M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2

        cdef double Tidal1 = tidal_contribution(self.EOBpars, r, 1)
        cdef double Tidal2 = tidal_contribution(self.EOBpars, r, 2)

        # Non-spinning calibration coefficient
        cdef double a6 = self.EOBpars.c_coeffs.a6
        # Spin-orbit calibration coefficient
        cdef double _dSO = self.EOBpars.c_coeffs.dSO
        # Spin-induced multipole moments
        cdef double _CES21 = self.EOBpars.tidal_params.CES21
        cdef double _CES22 = self.EOBpars.tidal_params.CES22
        cdef double _CES41 = self.EOBpars.tidal_params.CES41
        cdef double _CES42 = self.EOBpars.tidal_params.CES42
        cdef double _CBS31 = self.EOBpars.tidal_params.CBS31
        cdef double _CBS32 = self.EOBpars.tidal_params.CBS32

        cdef double x11 = r**3
        cdef double x0 = r**2
        cdef double x16 = 8.0*r + 4.0*x0 + 2.0*x11 + 16.0
        cdef double x13 = 756.0*nu
        cdef double x15 = x13 + 1079.0
        cdef double x14 = r**5
        cdef double x12 = r**4
        cdef double x4 = log(r)
        cdef double x10 = 49152.0*x4
        cdef double x2 = nu**2
        cdef double x9 = 102574080.0*x2 - 2119671837.36038
        cdef double x7 = 14700.0*nu + 42911.0
        cdef double x8 = 5760.0*x7
        cdef double x5 = x4**2
        cdef double x6 = 5787938193408.0*x5
        cdef double x3 = nu**3
        cdef double x1 = (X_1*chi_1 + X_2*chi_2)**2
        cdef double xi = (
            x0*sqrt(
                r*(
                    10611661054566.2*nu*r
                    + 2589101062873.81*nu*x0
                    - 12049908701745.2*nu
                    + 1822680546449.21*r*x2
                    - 326837426.241486*r*x7
                    - 39476764256925.6*r
                    + 133772083200.0*x0*x2
                    + 5107745331375.71*x0
                    - x10*(-516620178.136075*nu - r*x8 - 17902080.0*x0 - x9)
                    + 80059249540278.2*x2
                    + 6730497718123.02*x3
                    + x6
                    + 275059053208689.0
                )/(
                    55296.0*nu*(-2510664218.28128*nu - 42636451.6032331*x2 + 14515200.0*x3 + 1002013764.01019)
                    + r*x6
                    + r*(
                        43393301259014.8*nu
                        + 43133561885859.3*x2
                        + 5927865218923.02*x3
                        + 86618264430493.3*(1 - 0.496948781616935*nu)**2
                        + 188440788778196.0
                    )
                    - 9216.0*x0*(-630116198.873299*nu - 197773496.793534*x2 + 5805304367.87913)
                    + x10*(
                        -34560.0*nu*(11592.0*nu + 69847.0)
                        + r*(409207698.136075*nu + x9)
                        + x0*x8
                        + 17902080.0*x11
                    )
                    - 967680.0*x11*(-2675575.66847905*nu - 138240.0*x2 - 5278341.3229329)
                )
            )*(
                Tidal1
                + Tidal2
                + 7680.0*x12*(
                    2048.0*nu*x4*(336.0*r + x13 + 407.0)
                    + 28.0*nu*(1920.0*a6 + 733955.307463037)
                    - 7.0*r*(938918.400156317*nu - 185763.092693281*x2 - 245760.0)
                    - 5416406.59541186*x2
                    - 3440640.0
                )/(
                    241555486248.807*nu**4
                    + 32768.0*nu*x4*(
                        -38842241.4769507*nu
                        + 240.0*r*(-7466.27061066206*nu - 3024.0*x2 + 17264.0)
                        + 1920.0*x0*(588.0*nu + 1079.0)
                        + 960.0*x11*x15
                        + 480.0*x12*x15
                        + 161280.0*x14
                        - 1882456.23663972*x2
                        + 13447680.0
                    )
                    + 53760.0*nu*(
                        7680.0*a6*(x12 + x16)
                        + 113485.217444961*r*(-x12 + x16)
                        + 148.04406601634*r*(7704.0*r + 3852.0*x0 + 1926.0*x11 + 349.0*x12 + 36400.0)
                        + 128.0*r*(
                            13218.7851094412*r
                            + 8529.39255472061*x0
                            + 4264.6962773603*x11
                            - 6852.34813868015*x12
                            - 33722.4297811176
                        )
                    )
                    + 13212057600.0*x14
                    + 67645734912.0*x2*x5
                    + 7.0*x2*(
                        -39321600.0*a6*(3.0*r + 59.0)
                        + 745857848.115604*a6
                        + 122635399361.987*r
                        + 2064783811.32587*x0
                        - 6178501407.53758*x11
                        - 3089250703.76879*x12
                        + 1426660551.8844*x14
                        + 276057889687.011
                    )
                    + 1120.0*x3*(-163683964.822551*r - 17833256.898555*x0 - 1188987459.03162)
                )
                + x1/x0
            )/(
                x0 + x1
            )
        )
        return xi

    cpdef Hamiltonian_C_dynamics_return_t dynamics(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2
    ):
        """
        Compute the dynamics from the Hamiltonian,i.e., dHdr, dHdphi, dHdpr, dHdpphi,H and xi.
        """
        cdef double r = q[0]
        # cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double pphi = p[1]

        # Extra quantities used in the Hamiltonian
        cdef double M = self.EOBpars.p_params.M
        cdef double M2 = M * M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2

        cdef (double, double) tides1 = tidal_and_d_tidal_contribution(self.EOBpars, r, 1)
        cdef double Tidal1 = tides1[0]
        cdef double dTidal1 = tides1[1]
        cdef (double, double) tides2 = tidal_and_d_tidal_contribution(self.EOBpars, r, 2)
        cdef double Tidal2 = tides2[0]
        cdef double dTidal2 = tides2[1]

        # Non-spinning calibration coefficient
        cdef double a6 = self.EOBpars.c_coeffs.a6
        # Spin-orbit calibration coefficient
        cdef double dSO = self.EOBpars.c_coeffs.dSO
        # Spin-induced multipole moments
        cdef double CES21 = self.EOBpars.tidal_params.CES21
        cdef double CES22 = self.EOBpars.tidal_params.CES22
        cdef double CES41 = self.EOBpars.tidal_params.CES41
        cdef double CES42 = self.EOBpars.tidal_params.CES42
        cdef double CBS31 = self.EOBpars.tidal_params.CBS31
        cdef double CBS32 = self.EOBpars.tidal_params.CBS32
        cdef double x11 = r**4
        cdef double x12 = 1/x11
        cdef double x242 = 4.0*pphi**3*x12
        cdef double x60 = 2.0*pphi
        cdef double x0 = r**2
        cdef double x15 = 1/x0
        cdef double x241 = x15*x60
        cdef double x2 = X_2*chi_2
        cdef double x1 = X_1*chi_1
        cdef double x3 = x1 + x2
        cdef double x4 = x3**2
        cdef double x72 = x15*x4
        cdef double x63 = 1/r
        cdef double x71 = 2.0*x63 + 1.0
        cdef double x73 = x71*x72 + 1.0
        cdef double x218 = 1/x73
        cdef double x124 = 113485.217444961*r
        cdef double x123 = 148.04406601634*r
        cdef double x122 = 7704.0*r
        cdef double x121 = 128.0*r
        cdef double x120 = 7680.0*a6
        cdef double x118 = 8.0*r
        cdef double x18 = r**3
        cdef double x119 = 4.0*x0 + x118 + 2.0*x18 + 16.0
        cdef double x125 = (
            nu*(
                x120*(x11 + x119)
                + x121*(
                    13218.7851094412*r
                    + 8529.39255472061*x0
                    - 6852.34813868015*x11
                    + 4264.6962773603*x18
                    - 33722.4297811176
                )
                + x123*(3852.0*x0 + 349.0*x11 + x122 + 1926.0*x18 + 36400.0)
                + x124*(-x11 + x119)
            )
        )
        cdef double x102 = log(r)
        cdef double x116 = nu*x102
        cdef double x103 = 756.0*nu
        cdef double x113 = x103 + 1079.0
        cdef double x114 = x113*x18
        cdef double x112 = 588.0*nu + 1079.0
        cdef double x28 = r**5
        cdef double x23 = nu**2
        cdef double x115 = (
            -38842241.4769507*nu
            + 240.0*r*(-7466.27061066206*nu - 3024.0*x23 + 17264.0)
            + 1920.0*x0*x112
            + 480.0*x11*x113
            + 960.0*x114
            - 1882456.23663972*x23
            + 161280.0*x28
            + 13447680.0
        )
        cdef double x117 = x115*x116
        cdef double x111 = (
            x23*(
                -39321600.0*a6*(3.0*r + 59.0)
                + 745857848.115604*a6
                + 122635399361.987*r
                + 2064783811.32587*x0
                - 3089250703.76879*x11
                - 6178501407.53758*x18
                + 1426660551.8844*x28
                + 276057889687.011
            )
        )
        cdef double x109 = nu**3
        cdef double x110 = x109*(-163683964.822551*r - 17833256.898555*x0 - 1188987459.03162)
        cdef double x107 = x102**2
        cdef double x108 = x107*x23
        cdef double x106 = nu**4
        cdef double x126 = (
            1/(
                241555486248.807*x106
                + 67645734912.0*x108
                + 1120.0*x110
                + 7.0*x111
                + 32768.0*x117
                + 53760.0*x125
                + 13212057600.0*x28
            )
        )
        cdef double x127 = x11*x126
        cdef double x104 = 336.0*r + x103 + 407.0
        cdef double x105 = (
            2048.0*nu*x102*x104
            + 28.0*nu*(1920.0*a6 + 733955.307463037)
            - 7.0*r*(938918.400156317*nu - 185763.092693281*x23 - 245760.0)
            - 5416406.59541186*x23
            - 3440640.0
        )
        cdef double x128 = x105*x127
        cdef double x129 = Tidal1 + Tidal2 + 7680.0*x128
        cdef double x101 = X_2**2*chi_2**2*(1.0 - CES22)
        cdef double x100 = X_1**2*chi_1**2*(1.0 - CES21)
        cdef double x94 = X_2 - 1.0
        cdef double x33 = x1 - x2
        cdef double x51 = x3*x33
        cdef double x99 = x51*(CES21*x94 + X_2*(CES22 + 0.5) - 0.25)
        cdef double x97 = 4.0*nu
        cdef double x91 = 4.0*CES22
        cdef double x78 = 4.0*CES21
        cdef double x95 = -X_2*x91 + x78*x94
        cdef double x46 = x33**2
        cdef double x98 = x46*(x95 + x97 + 5.0)
        cdef double x96 = x4*(x95 + 13.0)
        cdef double x87 = 3.0*CES42
        cdef double x86 = 3.0*CES41
        cdef double x88 = x86 + x87
        cdef double x58 = 6.0*CES22
        cdef double x93 = x46*(CES21*(x58 + 4.0) - x88 + x91 - 8.0)
        cdef double x92 = x3**3*x33*(-6.0*CBS31 + 6.0*CBS32 + 4.0*CES21 - x86 + x87 - x91)
        cdef double x83 = CES41 + CES42
        cdef double x90 = x4*x46*(2.0*CES21*CES22 - x83)
        cdef double x49 = 12.0*CBS32
        cdef double x48 = 12.0*CBS31
        cdef double x89 = x3**4*(-18.0*CES21*CES22 + 8.0*CES21 + 8.0*CES22 - x48 - x49 - x88 + 32.0)
        cdef double x85 = x3*x33**3*(2.0*CBS31 - 2.0*CBS32 - CES41 + CES42)
        cdef double x54 = -4.0*CBS31
        cdef double x40 = 4.0*CBS32
        cdef double x55 = -x40 + x54
        cdef double x84 = x33**4*(-CES21*x58 - x55 - x83)
        cdef double x81 = 182.0*nu
        cdef double x75 = 34.0*X_2 + 69.0*nu - 34.0
        cdef double x79 = x75*x78
        cdef double x77 = 276.0*CES22*nu
        cdef double x82 = x51*(2.0*X_2*(68.0*CES22 + x81 - 409.0) - x77 + x79 - x81 + 409.0)
        cdef double x57 = CES22*X_2
        cdef double x80 = x46*(-741.0*nu + 196.0*x23 - 136.0*x57 + x77 + x79 + 115.0)
        cdef double x76 = x4*(12.0*CES21*x75 + 828.0*CES22*nu - 2881.0*nu - 408.0*x57 - 1167.0)
        cdef double x42 = 0.03125*x4
        cdef double x29 = 1/x28
        cdef double x19 = 1/x18
        cdef double x130 = (
            x12*(0.125*x96 + 0.125*x98 + x99)
            + x129
            + x19*(x100 + x101)
            + x29*(0.00223214285714286*x76 + 0.00669642857142857*x80 + 0.0133928571428571*x82)
            + x29*(x42*x93 + 0.046875*x84 + 0.1875*x85 + 0.015625*x89 + 0.1875*x90 + 0.0625*x92)
            + x72
        )
        cdef double x233 = x130*x218
        cdef double x175 = 102574080.0*x23 - 2119671837.36038
        cdef double x176 = 409207698.136075*nu + x175
        cdef double x177 = r*x176
        cdef double x173 = 14700.0*nu + 42911.0
        cdef double x174 = 5760.0*x173
        cdef double x172 = nu*(11592.0*nu + 69847.0)
        cdef double x178 = x102*(x0*x174 - 34560.0*x172 + x177 + 17902080.0*x18)
        cdef double x170 = (
            43393301259014.8*nu
            + 5927865218923.02*x109
            + 43133561885859.3*x23
            + 86618264430493.3*(1 - 0.496948781616935*nu)**2
            + 188440788778196.0
        )
        cdef double x171 = r*x170
        cdef double x169 = (
            nu*(-2510664218.28128*nu + 14515200.0*x109 - 42636451.6032331*x23 + 1002013764.01019)
        )
        cdef double x167 = -2675575.66847905*nu - 138240.0*x23 - 5278341.3229329
        cdef double x168 = x167*x18
        cdef double x165 = -630116198.873299*nu - 197773496.793534*x23 + 5805304367.87913
        cdef double x166 = x0*x165
        cdef double x164 = r*x107
        cdef double x213 = (
            5787938193408.0*x164 - 9216.0*x166 - 967680.0*x168 + 55296.0*x169 + x171 + 49152.0*x178
        )
        cdef double x214 = 1/x213
        cdef double x206 = 48.0*nu
        cdef double x208 = x91*(X_2*x206 - 38.0*X_2 + 87.0*nu + 16.0)
        cdef double x207 = x78*(X_2*(x206 - 38.0) - 135.0*nu + 22.0)
        cdef double x32 = X_1 - X_2
        cdef double x212 = x51*(-x207 - x208 - x32*(166.0*nu - 601.0))
        cdef double x209 = x207 - x208
        cdef double x211 = x46*(-773.0*nu - x209 + 4.0*x23 - 13.0)
        cdef double x210 = x4*(-2059.0*nu - x209 - 837.0)
        cdef double x199 = nu - 1.0
        cdef double x201 = x199*x91
        cdef double x200 = x199*x78
        cdef double x205 = x51*(14.0*X_2 + x200 - x201 - 7.0)
        cdef double x202 = x200 + x201
        cdef double x204 = x46*(x202 - x97 + 7.0)
        cdef double x203 = x4*(8.0*nu + x202 + 23.0)
        cdef double x196 = 5787938193408.0*x107
        cdef double x195 = 1822680546449.21*x23
        cdef double x185 = x102*(-516620178.136075*nu - r*x174 - 17902080.0*x0 - x175)
        cdef double x184 = x0*x23
        cdef double x183 = r*x173
        cdef double x181 = nu*x0
        cdef double x180 = nu*r
        cdef double x197 = (
            -12049908701745.2*nu
            + r*x195
            - 39476764256925.6*r
            + 5107745331375.71*x0
            + 6730497718123.02*x109
            + 10611661054566.2*x180
            + 2589101062873.81*x181
            - 326837426.241486*x183
            + 133772083200.0*x184
            - 49152.0*x185
            + x196
            + 80059249540278.2*x23
            + 275059053208689.0
        )
        cdef double x215 = (
            r*x129*x197*x214
            + x12*(0.015625*x210 + 0.015625*x211 + 0.03125*x212)
            + x19*(0.1875*x203 + 0.1875*x204 + 0.375*x205)
            + x72
        )
        cdef double x198 = 1/x197
        cdef double x162 = (
            0.000130208333333333*Tidal1 + 0.000130208333333333*Tidal2 + x128 + 0.000130208333333333*x72
        )
        cdef double x194 = x162**(-2)
        cdef double x193 = prst**2
        cdef double x7 = x0 + x4
        cdef double x192 = x7**2
        cdef double x216 = x192*x193*x194*x198*x213*x215
        cdef double x189 = 14.0*CES21 + 14.0*CES22
        cdef double x188 = 8.0*x23
        cdef double x190 = (
            0.15625*x4*(nu*(x189 + 5.0) - x188*(CES21 + CES22 + 2.0) + 5.0)
            - 0.15625*x46*(nu*(13.0 - x189) + x23*(8.0*CES21 + 8.0*CES22 - 4.0) + 3.0)
            - 0.3125*x51*(
                X_2*(36.0*nu - 2.0) - 2.0*nu*(7.0*CES21 - 7.0*CES22 + 9.0) + x188*(CES21 - CES22) + 1.0
            )
        )
        cdef double x182 = r*x23
        cdef double x186 = (
            -0.0438084424460039*nu
            - 0.143521050466841*r
            + 0.0185696317637669*x0
            + 0.0210425293255724*x107
            + 0.0244692826489756*x109
            + 0.0385795738434214*x180
            + 0.00941289164152486*x181
            + 0.00662650629087394*x182
            - 1.18824456940711e-6*x183
            + 0.000486339502879429*x184
            - 1.78696172427774e-10*x185
            + 0.291062041428379*x23
            + 1
        )
        cdef double x187 = x186**(-2)
        cdef double x179 = (
            (
                x164
                - 1.59227685093395e-9*x166
                - 1.67189069348064e-7*x168
                + 9.55366110560367e-9*x169
                + 1.72773095804465e-13*x171
                + 8.49214320498104e-9*x178
            )**2
        )
        cdef double x163 = x162**(-4)
        cdef double x161 = x7**4
        cdef double x131 = prst**4
        cdef double x191 = x131*x161*x163*x179*x187*x190
        cdef double x160 = r**(-13)
        cdef double x156 = r*x4
        cdef double x154 = r + 2.0
        cdef double x157 = x11 + x154*x156
        cdef double x158 = 1/x157
        cdef double x20 = pphi**2
        cdef double x159 = x158*x20*x63
        cdef double x155 = x154*x4
        cdef double x153 = x12*x131
        cdef double x152 = (
            nu*(452.542166996693 - 51.6952380952381*x102)
            + 602.318540416564*x109
            + x23*(118.4*x102 - 1796.13660498019)
        )
        cdef double x150 = 1.38977750996128*nu - 6.0*x106 + 3.42857142857143*x109 + 3.33842023648322*x23
        cdef double x138 = prst**8
        cdef double x151 = x138*x150
        cdef double x134 = prst**6
        cdef double x149 = x134*x19
        cdef double x148 = -33.9782122170436*nu - 14.0*x106 + 188.0*x109 - 89.5298327361234*x23
        cdef double x146 = -2.78300763695006*nu + 6.0*x109 - 5.4*x23
        cdef double x147 = x146*x15
        cdef double x145 = x131*x19
        cdef double x144 = 92.7110442849544*nu + 10.0*x109 - 131.0*x23
        cdef double x142 = 8.0*nu - 6.0*x23
        cdef double x143 = x142*x15
        cdef double x141 = 0.121954868780449*x138
        cdef double x140 = nu*x63
        cdef double x139 = nu*x138
        cdef double x137 = r**(-2.5)
        cdef double x135 = r**(-3.5)
        cdef double x136 = nu*x135
        cdef double x132 = r**(-4.5)
        cdef double x133 = nu*x132
        cdef double x64 = x15*x20
        cdef double x217 = (
            147.443752990146*x131*x133
            + x131*x143
            - 11.3175085791863*x134*x136
            + x134*x147
            + 1.48275342024365*x137*x139
            + x140*x141
            + x144*x145
            + x148*x149
            + x15*x151
            + x152*x153
            - x155*x159
            + 1.27277314139085e-19*x160*x191
            + 1.69542100694444e-8*x216*x29
            + x64
            + 1.0
        )
        cdef double x234 = 0.5*(x217*x233)**(-0.5)
        cdef double x240 = x233*x234
        cdef double x56 = CES21*X_2
        cdef double x50 = x48 - x49
        cdef double x47 = 0.03125*x46
        cdef double x44 = -15.0*CES22
        cdef double x43 = 3.0*CES21
        cdef double x38 = 9.0*CES21
        cdef double x59 = (
            x33*(
                0.0104166666666667*x4*(-27.0*CES22 - 44.0*X_2 + x38 + x50 + 18.0*x56 + 18.0*x57 + 22.0)
                + x47*(-X_2*x58 + 12.0*X_2 - x38 - x40 - x44 - x54 - 6.0*x56 - 6.0)
                + 0.0625*x51*(-3.0*CES22 - x43 - x55 - 2.0)
            )
        )
        cdef double x70 = x15*x59
        cdef double x45 = 12.0*X_2
        cdef double x41 = 4.0*CBS31 + x40
        cdef double x39 = 9.0*CES22 + x38
        cdef double x52 = (
            x42*(x39 + x41 - 34.0)
            + x47*(-CES21*x45 + CES22*x45 + x41 - x43 + x44 + 10.0)
            + 0.0208333333333333*x51*(x32*(x39 - 8.0) + x50)
        )
        cdef double x69 = x15*x52
        cdef double x67 = x32*x33
        cdef double x27 = pphi**4
        cdef double x65 = x12*x27
        cdef double x37 = -0.3515625*nu + 0.29296875*x23 - 0.41015625
        cdef double x36 = -0.2734375*nu - 0.798177083333333*x23 - 0.23046875
        cdef double x35 = 0.46875 - 0.28125*nu
        cdef double x34 = 0.34375*nu + 0.09375
        cdef double x21 = x19*x20
        cdef double x68 = (
            x67*(
                x15*(-0.03125*nu + 0.536458333333333*x23 + 0.078125)
                + x21*x36
                + x34*x63
                + x35*x64
                + x37*x65
                + 0.25
            )
        )
        cdef double x26 = 0.5859375*nu + 1.34765625*x23 + 0.41015625
        cdef double x24 = -2.0859375*nu - 2.07161458333333*x23 + 0.23046875
        cdef double x17 = -1.40625*nu - 0.46875
        cdef double x16 = 0.71875*nu - 0.09375
        cdef double x66 = (
            x3*(
                x15*(-5.53125*nu + 0.567708333333333*x23 - 0.078125)
                + x16*x63
                + x17*x64
                + x21*x24
                + x26*x65
                + 1.75
            )
        )
        cdef double x14 = dSO*nu
        cdef double x62 = x14*x19
        cdef double x61 = x19*x60
        cdef double x10 = pphi*x3
        cdef double x6 = 2.0*r
        cdef double x8 = r*(-x6 + x7)
        cdef double x5 = 2.0*x4
        cdef double x9 = 1/(2.0*x0 + x5 + x8)
        cdef double dHeffdpphi = (
            x240*(2.0*pphi*x15 - pphi*x154*x158*x5*x63)
            + x9*(
                pphi*x67*(x241*x35 + x242*x37 + x36*x61)
                + x10*(x17*x241 + x24*x61 + x242*x26)
                + x3*x62
                + x3*x69
                + x66
                + x68
                + x70
            )
        )
        cdef double x236 = prst**5
        cdef double x239 = 6.0*x236
        cdef double x235 = prst**3
        cdef double x238 = 4.0*x235
        cdef double x237 = prst**7
        cdef double x232 = x213*x29
        cdef double dHeffdpr = (
            x240*(
                11.8620273619492*nu*x137*x237
                + 3.39084201388889e-8*prst*x192*x194*x198*x215*x232
                + x12*x152*x238
                + 589.775011960583*x133*x235
                - 67.9050514751178*x136*x236
                + 0.975638950243592*x140*x237
                + x143*x238
                + x144*x19*x238
                + x147*x239
                + x148*x19*x239
                + 8.0*x15*x150*x237
                + 5.09109256556341e-19*x160*x161*x163*x179*x187*x190*x235
            )
        )
        cdef double dHeffdphi = 0
        cdef double x223 = 6.0*x0 + x118 + 8.0
        cdef double x222 = 4.0*x18
        cdef double x221 = x23*x63
        cdef double x224 = (
            1.31621673590926e-19*x105*x11*(
                53760.0*nu*(
                    3740417.71815805*r
                    + 2115968.85907902*x0
                    - 938918.400156317*x11
                    + x120*(x222 + x223)
                    + x121*(
                        17058.7851094412*r + 12794.0888320809*x0 - 27409.3925547206*x18 + 13218.7851094412
                    )
                    + x123*(5778.0*x0 + x122 + 1396.0*x18 + 7704.0)
                    + x124*(-x222 + x223)
                    + 1057984.42953951*x18
                    + 2888096.47013111
                )
                + 135291469824.0*x102*x221
                + 1120.0*x109*(-35666513.7971099*r - 163683964.822551)
                + 66060288000.0*x11
                + 32768.0*x115*x140
                + 32768.0*x116*(
                    -1791904.9465589*nu
                    + 3840.0*r*x112
                    + 2880.0*x0*x113
                    + 806400.0*x11
                    + 1920.0*x114
                    - 725760.0*x23
                    + 4143360.0
                )
                + 7.0*x23*(
                    -117964800.0*a6
                    + 4129567622.65173*r
                    - 18535504222.6128*x0
                    + 7133302759.42198*x11
                    - 12357002815.0752*x18
                    + 122635399361.987
                )
            )/(
                x106
                + 0.28004222119933*x108
                + 4.63661586574928e-9*x110
                + 2.8978849160933e-11*x111
                + 1.35654132757922e-7*x117
                + 2.22557561555966e-7*x125
                + 0.0546957463279941*x28
            )**2
        )
        cdef double x220 = (
            -6572428.80109422*nu + 2048.0*x104*x140 + 688128.0*x116 + 1300341.64885296*x23 + 1720320.0
        )
        cdef double x230 = dTidal1 + dTidal2 + 30720.0*x105*x126*x18 + 7680.0*x127*x220 - x224
        cdef double x74 = x19*x5
        cdef double x231 = x230 - x74
        cdef double x229 = x193*x215
        cdef double x225 = 11575876386816.0*x102
        cdef double x228 = (
            -18432.0*r*x165
            - 2903040.0*x0*x167
            + 49152.0*x102*(53706240.0*x0 + x176 + 11520.0*x183)
            + x170
            + x196
            + x225
            + x63*(283115520.0*x0*x173 - 1698693120.0*x172 + 49152.0*x177 + 879923036160.0*x18)
        )
        cdef double x227 = x131*x160*x161*x179*x190
        cdef double x226 = (
            5807150888816.34*nu
            + 10215490662751.4*r
            + 6291456.0*x102*(661500.0*nu + 279720.0*r + 1930995.0)
            + 5178202125747.62*x180
            + 267544166400.0*x182
            + x195
            + x225*x63
            - x63*(
                -25392914995744.3*nu
                - 879923036160.0*x0
                - 283115520.0*x183
                - 5041721180160.0*x23
                + 104186110149937.0
            )
            - 53501685054374.1
        )
        cdef double x219 = r**(-6)
        cdef double x53 = 2.0*x19
        cdef double x30 = 4.0*x29
        cdef double x31 = x27*x30
        cdef double x13 = 3.0*x12
        cdef double x25 = x13*x20
        cdef double x22 = 2.0*x21
        cdef double dHeffdr = (
            x234*(
                -x130*x217*(-x12*x5 - x71*x74)/x73**2
                + x217*x218*(
                    dTidal1
                    + dTidal2
                    + 30720.0*x105*x126*x18
                    + 7680.0*x11*x126*x220
                    - x12*(3.0*x100 + 3.0*x101)
                    - x219*(0.0111607142857143*x76 + 0.0334821428571429*x80 + 0.0669642857142857*x82)
                    - x219*(
                        0.15625*x4*x93 + 0.234375*x84 + 0.9375*x85 + 0.078125*x89 + 0.9375*x90 + 0.3125*x92
                    )
                    - x224
                    - x29*(0.5*x96 + 0.5*x98 + 4.0*x99)
                    - x74
                )
                + x233*(
                    -663.496888455656*nu*r**(-5.5)*x131
                    + 39.6112800271521*nu*x132*x134
                    - nu*x141*x15
                    + x12*x131*(-51.6952380952381*x140 + 118.4*x221)
                    + 6.78168402777778e-8*x12*x193*x194*x198*x213*x215*x7
                    - x13*x134*x148
                    - x131*x152*x30
                    + 7.59859378406358e-45*x131*x160*x161*x163*x187*x190*x213*x228
                    - 3.70688355060912*x135*x139
                    - 2.0*x142*x145
                    - 3.0*x144*x153
                    - 2.0*x146*x149
                    + x15*x154*x158*x20*x4
                    - x151*x53
                    + x154*x20*x4*x63*(x155 + x156 + x222)/x157**2
                    - x159*x4
                    - 9.25454462627843e-34*x163*x226*x227/x186**3
                    - 2.24091649004576e-37*x187*x192*x194*x213*x226*x229*x29
                    + 1.69542100694444e-8*x192*x193*x194*x198*x213*x29*(
                        r*x129*x214*x226
                        - 2.98505426338587e-26*r*x129*x197*x228/x179
                        + r*x197*x214*x230
                        - x12*(0.5625*x203 + 0.5625*x204 + 1.125*x205)
                        + x129*x197*x214
                        - x29*(0.0625*x210 + 0.0625*x211 + 0.125*x212)
                        - x74
                    )
                    + 1.69542100694444e-8*x192*x193*x194*x198*x215*x228*x29
                    - 8.47710503472222e-8*x216*x219
                    - x22
                    - 4.41515887225116e-12*x192*x198*x229*x231*x232/x162**3
                    - 6.62902677807736e-23*x187*x227*x231/x162**5
                    + 1.01821851311268e-18*x131*x163*x179*x187*x190*x7**3/r**12
                    - 1.65460508380811e-18*x191/r**14
                )
            )
            + x9*(
                pphi*x3*(
                    -x15*x16
                    - x17*x22
                    - x19*(-11.0625*nu + 1.13541666666667*x23 - 0.15625)
                    - x24*x25
                    - x26*x31
                )
                + pphi*x32*x33*(
                    -x15*x34 - x19*(-0.0625*nu + 1.07291666666667*x23 + 0.15625) - x22*x35 - x25*x36 - x31*x37
                )
                - x10*x13*x14
                - x10*x52*x53
                - x59*x61
            )
            - 0.25*(
                r*(x6 - 2.0) + x6 + x7
            )*(
                pphi*x66 + pphi*x68 + pphi*x70 + x10*x62 + x10*x69
            )/(
                x7 + 0.5*x8
            )**2
        )
        # Evaluate Hamiltonian
        cdef double H, xi
        H, xi = self.__call__(q, p, chi_1, chi_2, m_1, m_2, verbose=False)
        cdef double nuH = nu * H

        # Compute H Jacobian
        cdef double  dHdr = M2 * dHeffdr / nuH
        cdef double  dHdphi = M2 * dHeffdphi / nuH
        cdef double  dHdpr = M2 * dHeffdpr / nuH
        cdef double  dHdpphi = M2 * dHeffdpphi / nuH

        return dHdr, dHdphi, dHdpr, dHdpphi, H, xi

    cpdef double omega(self, qp_param_t q, qp_param_t p, double chi_1, double chi_2, double m_1, double m_2):
        """
        Compute the orbital frequency (dHdpphi) from the Hamiltonian.
        """
        cdef double r = q[0]
        # cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double pphi = p[1]

        # Extra quantities used in the Hamiltonian
        cdef double M = self.EOBpars.p_params.M
        cdef double M2 = M * M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2

        cdef double Tidal1 = tidal_contribution(self.EOBpars, r, 1)
        cdef double Tidal2 = tidal_contribution(self.EOBpars, r, 2)

        # Non-spinning calibration coefficient
        cdef double a6 = self.EOBpars.c_coeffs.a6
        # Spin-orbit calibration coefficient
        cdef double dSO = self.EOBpars.c_coeffs.dSO
        # Spin-induced multipole moments
        cdef double CES21 = self.EOBpars.tidal_params.CES21
        cdef double CES22 = self.EOBpars.tidal_params.CES22
        cdef double CES41 = self.EOBpars.tidal_params.CES41
        cdef double CES42 = self.EOBpars.tidal_params.CES42
        cdef double CBS31 = self.EOBpars.tidal_params.CBS31
        cdef double CBS32 = self.EOBpars.tidal_params.CBS32

        cdef double x109 = X_2 - 1.0
        cdef double x91 = 4.0*CES22
        cdef double x89 = 4.0*CES21
        cdef double x110 = -X_2*x91 + x109*x89
        cdef double x107 = 3.0*CES42
        cdef double x106 = 3.0*CES41
        cdef double x108 = x106 + x107
        cdef double x105 = CES41 + CES42
        cdef double x104 = 182.0*nu
        cdef double x101 = 34.0*X_2 + 69.0*nu - 34.0
        cdef double x103 = x101*x89
        cdef double x102 = 276.0*CES22*nu
        cdef double x7 = r**3
        cdef double x0 = r**2
        cdef double x82 = 8.0*r + 4.0*x0 + 2.0*x7 + 16.0
        cdef double x79 = 756.0*nu
        cdef double x81 = x79 + 1079.0
        cdef double x80 = r**5
        cdef double x56 = log(r)
        cdef double x63 = x56**2
        cdef double x55 = nu**4
        cdef double x54 = nu**3
        cdef double x17 = r**4
        cdef double x13 = nu**2
        cdef double x83 = (
            x17*(
                2048.0*nu*x56*(336.0*r + x79 + 407.0)
                + 28.0*nu*(1920.0*a6 + 733955.307463037)
                - 7.0*r*(938918.400156317*nu - 185763.092693281*x13 - 245760.0)
                - 5416406.59541186*x13
                - 3440640.0
            )/(
                32768.0*nu*x56*(
                    -38842241.4769507*nu
                    + 240.0*r*(-7466.27061066206*nu - 3024.0*x13 + 17264.0)
                    + 1920.0*x0*(588.0*nu + 1079.0)
                    - 1882456.23663972*x13
                    + 480.0*x17*x81
                    + 960.0*x7*x81
                    + 161280.0*x80
                    + 13447680.0
                )
                + 53760.0*nu*(
                    7680.0*a6*(x17 + x82)
                    + 113485.217444961*r*(-x17 + x82)
                    + 148.04406601634*r*(7704.0*r + 3852.0*x0 + 349.0*x17 + 1926.0*x7 + 36400.0)
                    + 128.0*r*(
                        13218.7851094412*r
                        + 8529.39255472061*x0
                        - 6852.34813868015*x17
                        + 4264.6962773603*x7
                        - 33722.4297811176
                    )
                )
                + 67645734912.0*x13*x63
                + 7.0*x13*(
                    -39321600.0*a6*(3.0*r + 59.0)
                    + 745857848.115604*a6
                    + 122635399361.987*r
                    + 2064783811.32587*x0
                    - 3089250703.76879*x17
                    - 6178501407.53758*x7
                    + 1426660551.8844*x80
                    + 276057889687.011
                )
                + 1120.0*x54*(-163683964.822551*r - 17833256.898555*x0 - 1188987459.03162)
                + 241555486248.807*x55
                + 13212057600.0*x80
            )
        )
        cdef double x100 = Tidal1 + Tidal2 + 7680.0*x83
        cdef double x95 = 4.0*nu
        cdef double x2 = X_2*chi_2
        cdef double x1 = X_1*chi_1
        cdef double x3 = x1 + x2
        cdef double x4 = x3**2
        cdef double x94 = 0.1875*x4
        cdef double x85 = 1/x80
        cdef double x10 = 1/x0
        cdef double x78 = x10*x4
        cdef double x49 = 6.0*CES22
        cdef double x48 = CES22*X_2
        cdef double x45 = -4.0*CBS31
        cdef double x33 = 4.0*CBS32
        cdef double x46 = -x33 + x45
        cdef double x24 = x1 - x2
        cdef double x44 = x24*x3
        cdef double x42 = 12.0*CBS32
        cdef double x41 = 12.0*CBS31
        cdef double x39 = x24**2
        cdef double x35 = 0.03125*x4
        cdef double x26 = 1/r
        cdef double x18 = 1/x17
        cdef double x8 = 1/x7
        cdef double x111 = (
            (
                x100
                + x18*(
                    0.125*x39*(x110 + x95 + 5.0)
                    + 0.125*x4*(x110 + 13.0)
                    + x44*(CES21*x109 + X_2*(CES22 + 0.5) - 0.25)
                )
                + x78
                + x8*(X_1**2*chi_1**2*(1.0 - CES21) + X_2**2*chi_2**2*(1.0 - CES22))
                + x85*(
                    0.00669642857142857*x39*(-741.0*nu + x102 + x103 + 196.0*x13 - 136.0*x48 + 115.0)
                    + 0.00223214285714286*x4*(
                        12.0*CES21*x101 + 828.0*CES22*nu - 2881.0*nu - 408.0*x48 - 1167.0
                    )
                    + 0.0133928571428571*x44*(
                        2.0*X_2*(68.0*CES22 + x104 - 409.0) - x102 + x103 - x104 + 409.0
                    )
                )
                + x85*(
                    0.046875*x24**4*(-CES21*x49 - x105 - x46)
                    + 0.1875*x24**3*x3*(2.0*CBS31 - 2.0*CBS32 - CES41 + CES42)
                    + 0.0625*x24*x3**3*(-6.0*CBS31 + 6.0*CBS32 + 4.0*CES21 - x106 + x107 - x91)
                    + 0.015625*x3**4*(-18.0*CES21*CES22 + 8.0*CES21 + 8.0*CES22 - x108 - x41 - x42 + 32.0)
                    + x35*x39*(CES21*(x49 + 4.0) - x108 + x91 - 8.0)
                    + x39*x94*(2.0*CES21*CES22 - x105)
                )
            )/(
                x78*(2.0*x26 + 1.0) + 1.0
            )
        )
        cdef double x96 = 48.0*nu
        cdef double x98 = x91*(X_2*x96 - 38.0*X_2 + 87.0*nu + 16.0)
        cdef double x97 = x89*(X_2*(x96 - 38.0) - 135.0*nu + 22.0)
        cdef double x99 = x97 - x98
        cdef double x88 = nu - 1.0
        cdef double x92 = x88*x91
        cdef double x90 = x88*x89
        cdef double x93 = x90 + x92
        cdef double x68 = 102574080.0*x13 - 2119671837.36038
        cdef double x64 = 14700.0*nu + 42911.0
        cdef double x67 = 5760.0*x64
        cdef double x77 = (
            x56*(-34560.0*nu*(11592.0*nu + 69847.0) + r*(409207698.136075*nu + x68) + x0*x67 + 17902080.0*x7)
        )
        cdef double x76 = (
            r*(
                43393301259014.8*nu
                + 43133561885859.3*x13
                + 5927865218923.02*x54
                + 86618264430493.3*(1 - 0.496948781616935*nu)**2
                + 188440788778196.0
            )
        )
        cdef double x75 = nu*(-2510664218.28128*nu - 42636451.6032331*x13 + 14515200.0*x54 + 1002013764.01019)
        cdef double x74 = x7*(-2675575.66847905*nu - 138240.0*x13 - 5278341.3229329)
        cdef double x73 = x0*(-630116198.873299*nu - 197773496.793534*x13 + 5805304367.87913)
        cdef double x72 = r*x63
        cdef double x87 = 5787938193408.0*x72 - 9216.0*x73 - 967680.0*x74 + 55296.0*x75 + x76 + 49152.0*x77
        cdef double x69 = x56*(-516620178.136075*nu - r*x67 - 17902080.0*x0 - x68)
        cdef double x66 = x0*x13
        cdef double x65 = r*x64
        cdef double x62 = r*x13
        cdef double x61 = nu*x0
        cdef double x60 = nu*r
        cdef double x86 = (
            -12049908701745.2*nu
            - 39476764256925.6*r
            + 5107745331375.71*x0
            + 80059249540278.2*x13
            + 6730497718123.02*x54
            + 10611661054566.2*x60
            + 2589101062873.81*x61
            + 1822680546449.21*x62
            + 5787938193408.0*x63
            - 326837426.241486*x65
            + 133772083200.0*x66
            - 49152.0*x69
            + 275059053208689.0
        )
        cdef double x84 = (
            0.000130208333333333*Tidal1 + 0.000130208333333333*Tidal2 + 0.000130208333333333*x78 + x83
        )
        cdef double x71 = 14.0*CES21 + 14.0*CES22
        cdef double x70 = 8.0*x13
        cdef double x57 = r + 2.0
        cdef double x58 = x4*x57
        cdef double x59 = x26/(r*x58 + x17)
        cdef double x52 = prst**8
        cdef double x53 = nu*x52
        cdef double x51 = prst**6
        cdef double x50 = prst**4
        cdef double x47 = CES21*X_2
        cdef double x43 = x41 - x42
        cdef double x40 = 0.03125*x39
        cdef double x38 = 12.0*X_2
        cdef double x37 = -15.0*CES22
        cdef double x36 = 3.0*CES21
        cdef double x34 = 4.0*CBS31 + x33
        cdef double x31 = 9.0*CES21
        cdef double x32 = 9.0*CES22 + x31
        cdef double x30 = pphi**4*x18
        cdef double x27 = pphi**2
        cdef double x29 = x27*x8
        cdef double x28 = x10*x27
        cdef double x23 = X_1 - X_2
        cdef double x25 = x23*x24
        cdef double x22 = -0.3515625*nu + 0.29296875*x13 - 0.41015625
        cdef double x21 = -0.2734375*nu - 0.798177083333333*x13 - 0.23046875
        cdef double x20 = 0.46875 - 0.28125*nu
        cdef double x19 = 4.0*pphi**3*x18
        cdef double x16 = 0.5859375*nu + 1.34765625*x13 + 0.41015625
        cdef double x11 = 2.0*pphi
        cdef double x15 = x11*x8
        cdef double x14 = -2.0859375*nu - 2.07161458333333*x13 + 0.23046875
        cdef double x12 = x10*x11
        cdef double x9 = -1.40625*nu - 0.46875
        cdef double x6 = x0 + x4
        cdef double x5 = 2.0*x4
        cdef double dHeffdpphi = (
            0.5*x111*(
                x111*(
                    147.443752990146*nu*r**(-4.5)*x50
                    - 11.3175085791863*nu*r**(-3.5)*x51
                    + 1.69542100694444e-8*prst**2*x6**2*x85*x87*(
                        r*x100*x86/x87
                        + x18*(
                            0.015625*x39*(-773.0*nu + 4.0*x13 - x99 - 13.0)
                            + 0.015625*x4*(-2059.0*nu - x99 - 837.0)
                            + 0.03125*x44*(-x23*(166.0*nu - 601.0) - x97 - x98)
                        )
                        + x78
                        + x8*(
                            0.1875*x39*(x93 - x95 + 7.0)
                            + 0.375*x44*(14.0*X_2 + x90 - x92 - 7.0)
                            + x94*(8.0*nu + x93 + 23.0)
                        )
                    )/(
                        x84**2*x86
                    )
                    + 1.48275342024365*r**(-2.5)*x53
                    + x10*x50*(8.0*nu - 6.0*x13)
                    + x10*x51*(-2.78300763695006*nu - 5.4*x13 + 6.0*x54)
                    + x10*x52*(1.38977750996128*nu + 3.33842023648322*x13 + 3.42857142857143*x54 - 6.0*x55)
                    + x18*x50*(
                        nu*(452.542166996693 - 51.6952380952381*x56)
                        + x13*(118.4*x56 - 1796.13660498019)
                        + 602.318540416564*x54
                    )
                    + 0.121954868780449*x26*x53
                    - x27*x58*x59
                    + x28
                    + x50*x8*(92.7110442849544*nu - 131.0*x13 + 10.0*x54)
                    + x51*x8*(-33.9782122170436*nu - 89.5298327361234*x13 + 188.0*x54 - 14.0*x55)
                    + 1.0
                    + 1.27277314139085e-19*x50*x6**4*(
                        -0.15625*x39*(nu*(13.0 - x71) + x13*(8.0*CES21 + 8.0*CES22 - 4.0) + 3.0)
                        + 0.15625*x4*(nu*(x71 + 5.0) - x70*(CES21 + CES22 + 2.0) + 5.0)
                        - 0.3125*x44*(
                            X_2*(36.0*nu - 2.0)
                            - 2.0*nu*(7.0*CES21 - 7.0*CES22 + 9.0)
                            + x70*(CES21 - CES22)
                            + 1.0
                        )
                    )*(
                        x72
                        - 1.59227685093395e-9*x73
                        - 1.67189069348064e-7*x74
                        + 9.55366110560367e-9*x75
                        + 1.72773095804465e-13*x76
                        + 8.49214320498104e-9*x77
                    )**2/(
                        r**13*x84**4*(
                            -0.0438084424460039*nu
                            - 0.143521050466841*r
                            + 0.0185696317637669*x0
                            + 0.291062041428379*x13
                            + 0.0244692826489756*x54
                            + 0.0385795738434214*x60
                            + 0.00941289164152486*x61
                            + 0.00662650629087394*x62
                            + 0.0210425293255724*x63
                            - 1.18824456940711e-6*x65
                            + 0.000486339502879429*x66
                            - 1.78696172427774e-10*x69
                            + 1
                        )**2
                    )
                )
            )**(
                -0.5
            )*(
                2.0*pphi*x10 - pphi*x5*x57*x59
            )
            + (
                dSO*nu*x3*x8
                + pphi*x25*(x12*x20 + x15*x21 + x19*x22)
                + pphi*x3*(x12*x9 + x14*x15 + x16*x19)
                + x10*x24*(
                    0.0104166666666667*x4*(-27.0*CES22 - 44.0*X_2 + x31 + x43 + 18.0*x47 + 18.0*x48 + 22.0)
                    + x40*(-X_2*x49 + 12.0*X_2 - x31 - x33 - x37 - x45 - 6.0*x47 - 6.0)
                    + 0.0625*x44*(-3.0*CES22 - x36 - x46 - 2.0)
                )
                + x10*x3*(
                    x35*(x32 + x34 - 34.0)
                    + x40*(-CES21*x38 + CES22*x38 + x34 - x36 + x37 + 10.0)
                    + 0.0208333333333333*x44*(x23*(x32 - 8.0) + x43)
                )
                + x25*(
                    x10*(-0.03125*nu + 0.536458333333333*x13 + 0.078125)
                    + x20*x28
                    + x21*x29
                    + x22*x30
                    + x26*(0.34375*nu + 0.09375)
                    + 0.25
                )
                + x3*(
                    x10*(-5.53125*nu + 0.567708333333333*x13 - 0.078125)
                    + x14*x29
                    + x16*x30
                    + x26*(0.71875*nu - 0.09375)
                    + x28*x9
                    + 1.75
                )
            )/(
                r*(-2.0*r + x6) + 2.0*x0 + x5
            )
        )
        # Evaluate Hamiltonian
        cdef double H, _xi
        H, _xi = self.__call__(q, p, chi_1, chi_2, m_1, m_2, verbose=False)
        cdef double nuH = nu * H

        # Compute H Jacobian

        cdef double omega = M2 * dHeffdpphi / (nuH)

        return omega

    cpdef Hamiltonian_C_auxderivs_return_t auxderivs(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2
    ):
        """
        Auxiliary derivatives of the SEOBNRv5HM Hamiltonian.
        """
        cdef double r = q[0]
        # cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double pphi = p[1]

        # Extra quantities used in the Hamiltonian
        cdef double M = self.EOBpars.p_params.M
        cdef double _M2 = M * M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2

        cdef (double, double) tides1 = tidal_and_d_tidal_contribution(self.EOBpars, r, 1)
        cdef double Tidal1 = tides1[0]
        cdef double dTidal1 = tides1[1]
        cdef (double, double) tides2 = tidal_and_d_tidal_contribution(self.EOBpars, r, 2)
        cdef double Tidal2 = tides2[0]
        cdef double dTidal2 = tides2[1]

        # Non-spinning calibration coefficient
        cdef double a6 = self.EOBpars.c_coeffs.a6
        # Spin-orbit calibration coefficient
        cdef double dSO = self.EOBpars.c_coeffs.dSO
        # Spin-induced multipole moments
        cdef double CES21 = self.EOBpars.tidal_params.CES21
        cdef double CES22 = self.EOBpars.tidal_params.CES22
        cdef double CES41 = self.EOBpars.tidal_params.CES41
        cdef double CES42 = self.EOBpars.tidal_params.CES42
        cdef double CBS31 = self.EOBpars.tidal_params.CBS31
        cdef double CBS32 = self.EOBpars.tidal_params.CBS32
        cdef double x187 = pphi**4
        cdef double x10 = r**4
        cdef double x11 = 1/x10
        cdef double x206 = x11*x187
        cdef double x181 = pphi**2
        cdef double x2 = r**2
        cdef double x3 = 1/x2
        cdef double x205 = x181*x3
        cdef double x5 = X_2*chi_2
        cdef double x4 = X_1*chi_1
        cdef double x21 = x4 - x5
        cdef double x203 = pphi*x21
        cdef double x202 = CES21*X_2
        cdef double x39 = 12.0*CBS32
        cdef double x38 = 12.0*CBS31
        cdef double x200 = x38 - x39
        cdef double x22 = x21**2
        cdef double x199 = 0.03125*x22
        cdef double x197 = -15.0*CES22
        cdef double x196 = 3.0*CES21
        cdef double x193 = 9.0*CES21
        cdef double x34 = 4.0*CBS32
        cdef double x33 = -4.0*CBS31
        cdef double x35 = x33 - x34
        cdef double x31 = 6.0*CES22
        cdef double x6 = x4 + x5
        cdef double x29 = x21*x6
        cdef double x18 = CES22*X_2
        cdef double x7 = x6**2
        cdef double x204 = (
            x203*(
                x199*(-X_2*x31 + 12.0*X_2 - x193 - x197 - 6.0*x202 - x33 - x34 - 6.0)
                + 0.0625*x29*(-3.0*CES22 - x196 - x35 - 2.0)
                + 0.0104166666666667*x7*(-27.0*CES22 - 44.0*X_2 + 18.0*x18 + x193 + x200 + 18.0*x202 + 22.0)
            )
        )
        cdef double x198 = 12.0*X_2
        cdef double x195 = 4.0*CBS31 + x34
        cdef double x194 = 9.0*CES22 + x193
        cdef double x177 = pphi*x6
        cdef double x102 = X_1 - X_2
        cdef double x47 = 0.03125*x7
        cdef double x201 = (
            x177*(
                x199*(-CES21*x198 + CES22*x198 + x195 - x196 + x197 + 10.0)
                + 0.0208333333333333*x29*(x102*(x194 - 8.0) + x200)
                + x47*(x194 + x195 - 34.0)
            )
        )
        cdef double x24 = nu**2
        cdef double x192 = -0.3515625*nu + 0.29296875*x24 - 0.41015625
        cdef double x191 = -0.2734375*nu - 0.798177083333333*x24 - 0.23046875
        cdef double x190 = 0.46875 - 0.28125*nu
        cdef double x189 = 0.34375*nu + 0.09375
        cdef double x16 = r**5
        cdef double x17 = 1/x16
        cdef double x188 = 4.0*x17*x187
        cdef double x186 = 0.5859375*nu + 1.34765625*x24 + 0.41015625
        cdef double x157 = 3.0*x11
        cdef double x185 = x157*x181
        cdef double x184 = -2.0859375*nu - 2.07161458333333*x24 + 0.23046875
        cdef double x13 = r**3
        cdef double x14 = 1/x13
        cdef double x182 = x14*x181
        cdef double x183 = 2.0*x182
        cdef double x180 = -1.40625*nu - 0.46875
        cdef double x179 = 0.71875*nu - 0.09375
        cdef double x178 = dSO*nu*x177
        cdef double x142 = x2 + x7
        cdef double x139 = 2.0*r
        cdef double x176 = r*(-x139 + x142)
        cdef double x153 = 2.0*x14
        cdef double x12 = 2.0*x7
        cdef double x0 = 1/r
        cdef double dHodddr = (
            (
                pphi*x102*x21*(
                    -x14*(-0.0625*nu + 1.07291666666667*x24 + 0.15625)
                    - x183*x190
                    - x185*x191
                    - x188*x192
                    - x189*x3
                )
                + pphi*x6*(
                    -x14*(-11.0625*nu + 1.13541666666667*x24 - 0.15625)
                    - x179*x3
                    - x180*x183
                    - x184*x185
                    - x186*x188
                )
                - x153*x201
                - x153*x204
                - x157*x178
            )/(
                x12 + x176 + 2.0*x2
            )
            - 0.25*(
                r*(x139 - 2.0) + x139 + x142
            )*(
                x102*x203*(
                    x0*x189
                    + x182*x191
                    + x190*x205
                    + x192*x206
                    + x3*(-0.03125*nu + 0.536458333333333*x24 + 0.078125)
                    + 0.25
                )
                + x14*x178
                + x177*(
                    x0*x179
                    + x180*x205
                    + x182*x184
                    + x186*x206
                    + x3*(-5.53125*nu + 0.567708333333333*x24 - 0.078125)
                    + 1.75
                )
                + x201*x3
                + x204*x3
            )/(
                x142 + 0.5*x176
            )**2
        )
        cdef double x172 = prst**5
        cdef double x175 = 6.0*x172
        cdef double x171 = prst**3
        cdef double x174 = 4.0*x171
        cdef double x173 = prst**7
        cdef double x169 = r**(-13)
        cdef double x160 = x142**4
        cdef double x170 = x160*x169
        cdef double x166 = 14.0*CES21 + 14.0*CES22
        cdef double x165 = 8.0*x24
        cdef double x167 = (
            -0.15625*x22*(nu*(13.0 - x166) + x24*(8.0*CES21 + 8.0*CES22 - 4.0) + 3.0)
            - 0.3125*x29*(
                X_2*(36.0*nu - 2.0) - 2.0*nu*(7.0*CES21 - 7.0*CES22 + 9.0) + x165*(CES21 - CES22) + 1.0
            )
            + 0.15625*x7*(nu*(x166 + 5.0) - x165*(CES21 + CES22 + 2.0) + 5.0)
        )
        cdef double x120 = 102574080.0*x24 - 2119671837.36038
        cdef double x107 = 14700.0*nu + 42911.0
        cdef double x119 = 5760.0*x107
        cdef double x129 = -516620178.136075*nu - r*x119 - x120 - 17902080.0*x2
        cdef double x128 = x2*x24
        cdef double x127 = nu*x2
        cdef double x108 = r*x107
        cdef double x105 = r*x24
        cdef double x103 = nu*r
        cdef double x64 = nu**3
        cdef double x57 = log(r)
        cdef double x62 = x57**2
        cdef double x163 = (
            -0.0438084424460039*nu
            - 0.143521050466841*r
            + 0.0385795738434214*x103
            + 0.00662650629087394*x105
            - 1.18824456940711e-6*x108
            + 0.00941289164152486*x127
            + 0.000486339502879429*x128
            - 1.78696172427774e-10*x129*x57
            + 0.0185696317637669*x2
            + 0.291062041428379*x24
            + 0.0210425293255724*x62
            + 0.0244692826489756*x64
            + 1
        )
        cdef double x164 = x163**(-2)
        cdef double x80 = 113485.217444961*r
        cdef double x79 = 148.04406601634*r
        cdef double x78 = 7704.0*r
        cdef double x77 = 128.0*r
        cdef double x76 = 7680.0*a6
        cdef double x74 = 2.0*x13
        cdef double x73 = 8.0*r
        cdef double x75 = 4.0*x2 + x73 + x74 + 16.0
        cdef double x81 = (
            nu*(
                x76*(x10 + x75)
                + x77*(
                    13218.7851094412*r
                    - 6852.34813868015*x10
                    + 4264.6962773603*x13
                    + 8529.39255472061*x2
                    - 33722.4297811176
                )
                + x79*(349.0*x10 + 1926.0*x13 + 3852.0*x2 + x78 + 36400.0)
                + x80*(-x10 + x75)
            )
        )
        cdef double x71 = nu*x57
        cdef double x58 = 756.0*nu
        cdef double x68 = x58 + 1079.0
        cdef double x69 = x13*x68
        cdef double x67 = 588.0*nu + 1079.0
        cdef double x70 = (
            -38842241.4769507*nu
            + 240.0*r*(-7466.27061066206*nu - 3024.0*x24 + 17264.0)
            + 480.0*x10*x68
            + 161280.0*x16
            + 1920.0*x2*x67
            - 1882456.23663972*x24
            + 960.0*x69
            + 13447680.0
        )
        cdef double x72 = x70*x71
        cdef double x66 = (
            x24*(
                -39321600.0*a6*(3.0*r + 59.0)
                + 745857848.115604*a6
                + 122635399361.987*r
                - 3089250703.76879*x10
                - 6178501407.53758*x13
                + 1426660551.8844*x16
                + 2064783811.32587*x2
                + 276057889687.011
            )
        )
        cdef double x65 = x64*(-163683964.822551*r - 17833256.898555*x2 - 1188987459.03162)
        cdef double x63 = x24*x62
        cdef double x61 = nu**4
        cdef double x82 = (
            1/(
                13212057600.0*x16
                + 241555486248.807*x61
                + 67645734912.0*x63
                + 1120.0*x65
                + 7.0*x66
                + 32768.0*x72
                + 53760.0*x81
            )
        )
        cdef double x83 = x10*x82
        cdef double x59 = 336.0*r + x58 + 407.0
        cdef double x60 = (
            2048.0*nu*x57*x59
            + 28.0*nu*(1920.0*a6 + 733955.307463037)
            - 7.0*r*(938918.400156317*nu - 185763.092693281*x24 - 245760.0)
            - 5416406.59541186*x24
            - 3440640.0
        )
        cdef double x84 = x60*x83
        cdef double x8 = x3*x7
        cdef double x161 = (
            0.000130208333333333*Tidal1 + 0.000130208333333333*Tidal2 + 0.000130208333333333*x8 + x84
        )
        cdef double x162 = x161**(-4)
        cdef double x121 = 409207698.136075*nu + x120
        cdef double x122 = r*x121
        cdef double x118 = nu*(11592.0*nu + 69847.0)
        cdef double x123 = -34560.0*x118 + x119*x2 + x122 + 17902080.0*x13
        cdef double x116 = (
            43393301259014.8*nu
            + 43133561885859.3*x24
            + 5927865218923.02*x64
            + 86618264430493.3*(1 - 0.496948781616935*nu)**2
            + 188440788778196.0
        )
        cdef double x117 = r*x116
        cdef double x115 = (
            nu*(-2510664218.28128*nu - 42636451.6032331*x24 + 14515200.0*x64 + 1002013764.01019)
        )
        cdef double x113 = -2675575.66847905*nu - 138240.0*x24 - 5278341.3229329
        cdef double x114 = x113*x13
        cdef double x111 = -630116198.873299*nu - 197773496.793534*x24 + 5805304367.87913
        cdef double x112 = x111*x2
        cdef double x131 = (
            (
                r*x62
                - 1.59227685093395e-9*x112
                - 1.67189069348064e-7*x114
                + 9.55366110560367e-9*x115
                + 1.72773095804465e-13*x117
                + 8.49214320498104e-9*x123*x57
            )**2
        )
        cdef double x168 = x131*x162*x164*x167
        cdef double x159 = (
            4.0*nu*(452.542166996693 - 51.6952380952381*x57)
            + 4.0*x24*(118.4*x57 - 1796.13660498019)
            + 2409.27416166626*x64
        )
        cdef double x158 = 1.38977750996128*nu + 3.33842023648322*x24 - 6.0*x61 + 3.42857142857143*x64
        cdef double x156 = -33.9782122170436*nu - 89.5298327361234*x24 - 14.0*x61 + 188.0*x64
        cdef double x155 = -2.78300763695006*nu - 5.4*x24 + 6.0*x64
        cdef double x154 = 92.7110442849544*nu - 131.0*x24 + 10.0*x64
        cdef double x152 = 8.0*nu - 6.0*x24
        cdef double x149 = r**(-3.5)
        cdef double x147 = r**(-4.5)
        cdef double x88 = nu*x0
        cdef double dQdprst = (
            11.8620273619492*nu*r**(-2.5)*x173
            + 589.775011960583*nu*x147*x171
            - 67.9050514751178*nu*x149*x172
            + x11*x159*x171
            + x14*x154*x174
            + x14*x156*x175
            + x152*x174*x3
            + x155*x175*x3
            + 8.0*x158*x173*x3
            + 5.09109256556341e-19*x168*x170*x171
            + 0.975638950243592*x173*x88
        )
        cdef double x150 = prst**8
        cdef double x151 = nu*x150
        cdef double x148 = prst**6
        cdef double x146 = prst**4
        cdef double x92 = 6.0*x2 + x73 + 8.0
        cdef double x91 = 4.0*x13
        cdef double x90 = x0*x24
        cdef double x93 = (
            1.31621673590926e-19*x10*x60*(
                53760.0*nu*(
                    3740417.71815805*r
                    - 938918.400156317*x10
                    + 1057984.42953951*x13
                    + 2115968.85907902*x2
                    + x76*(x91 + x92)
                    + x77*(17058.7851094412*r - 27409.3925547206*x13 + 12794.0888320809*x2 + 13218.7851094412)
                    + x79*(1396.0*x13 + 5778.0*x2 + x78 + 7704.0)
                    + x80*(-x91 + x92)
                    + 2888096.47013111
                )
                + 66060288000.0*x10
                + 7.0*x24*(
                    -117964800.0*a6
                    + 4129567622.65173*r
                    + 7133302759.42198*x10
                    - 12357002815.0752*x13
                    - 18535504222.6128*x2
                    + 122635399361.987
                )
                + 135291469824.0*x57*x90
                + 1120.0*x64*(-35666513.7971099*r - 163683964.822551)
                + 32768.0*x70*x88
                + 32768.0*x71*(
                    -1791904.9465589*nu
                    + 3840.0*r*x67
                    + 806400.0*x10
                    + 2880.0*x2*x68
                    - 725760.0*x24
                    + 1920.0*x69
                    + 4143360.0
                )
            )/(
                0.0546957463279941*x16
                + x61
                + 0.28004222119933*x63
                + 4.63661586574928e-9*x65
                + 2.8978849160933e-11*x66
                + 1.35654132757922e-7*x72
                + 2.22557561555966e-7*x81
            )**2
        )
        cdef double x89 = (
            -6572428.80109422*nu + 1300341.64885296*x24 + 2048.0*x59*x88 + 688128.0*x71 + 1720320.0
        )
        cdef double x134 = dTidal1 + dTidal2 + 30720.0*x13*x60*x82 + 7680.0*x83*x89 - x93
        cdef double x15 = x12*x14
        cdef double x145 = x134 - x15
        cdef double x124 = 49152.0*x57
        cdef double x110 = 5787938193408.0*x62
        cdef double x106 = 11575876386816.0*x57
        cdef double x132 = (
            -18432.0*r*x111
            + x0*(283115520.0*x107*x2 - 1698693120.0*x118 + 49152.0*x122 + 879923036160.0*x13)
            + x106
            + x110
            - 2903040.0*x113*x2
            + x116
            + x124*(11520.0*x108 + x121 + 53706240.0*x2)
        )
        cdef double x125 = r*x110 - 9216.0*x112 - 967680.0*x114 + 55296.0*x115 + x117 + x123*x124
        cdef double x104 = 1822680546449.21*x24
        cdef double x109 = (
            5807150888816.34*nu
            + 10215490662751.4*r
            + x0*x106
            - x0*(
                -25392914995744.3*nu
                - 283115520.0*x108
                - 879923036160.0*x2
                - 5041721180160.0*x24
                + 104186110149937.0
            )
            + 5178202125747.62*x103
            + x104
            + 267544166400.0*x105
            + 6291456.0*x57*(661500.0*nu + 279720.0*r + 1930995.0)
            - 53501685054374.1
        )
        cdef double dQdr = (
            -663.496888455656*nu*r**(-5.5)*x146
            + 39.6112800271521*nu*x147*x148
            - 9.25454462627843e-34*x109*x131*x146*x162*x167*x170/x163**3
            - 3.0*x11*x146*x154
            + x11*x146*(-51.6952380952381*x88 + 118.4*x90)
            + 7.59859378406358e-45*x125*x132*x146*x160*x162*x164*x167*x169
            - 6.62902677807736e-23*x131*x145*x146*x164*x167*x170/x161**5
            - x146*x152*x153
            - x146*x159*x17
            - x148*x153*x155
            - x148*x156*x157
            - 3.70688355060912*x149*x151
            - x150*x153*x158
            - 0.121954868780449*x151*x3
            + 1.01821851311268e-18*x131*x142**3*x146*x162*x164*x167/r**12
            - 1.65460508380811e-18*x146*x160*x168/r**14
        )
        cdef double x143 = 1/x142
        cdef double x130 = (
            -12049908701745.2*nu
            + r*x104
            - 39476764256925.6*r
            + 10611661054566.2*x103
            - 326837426.241486*x108
            + x110
            - x124*x129
            + 2589101062873.81*x127
            + 133772083200.0*x128
            + 5107745331375.71*x2
            + 80059249540278.2*x24
            + 6730497718123.02*x64
            + 275059053208689.0
        )
        cdef double x126 = 1/x125
        cdef double x140 = x126*x130
        cdef double x141 = sqrt(r*x140)
        cdef double x144 = x141*x143
        cdef double x133 = 2.98505426338587e-26*r*x130*x132/x131
        cdef double x85 = Tidal1 + Tidal2 + 7680.0*x84
        cdef double x86 = x8 + x85
        cdef double dxidr = (
            x139*x144*x86
            - x141*x74*x86/x142**2
            + x144*x145*x2
            + 0.5*x143*x2*x86*(r*x109*x126 - x133 + x140)/x141
        )
        cdef double x136 = r*x7
        cdef double x135 = r + 2.0
        cdef double x137 = x10 + x135*x136
        cdef double x138 = 1/x137
        cdef double dBnpadr = r*x135*(x135*x7 + x136 + x91)/x137**2 - r*x138 - x135*x138
        cdef double x98 = 48.0*nu
        cdef double x45 = 4.0*CES22
        cdef double x100 = x45*(X_2*x98 - 38.0*X_2 + 87.0*nu + 16.0)
        cdef double x25 = 4.0*CES21
        cdef double x99 = x25*(X_2*(x98 - 38.0) - 135.0*nu + 22.0)
        cdef double x101 = -x100 + x99
        cdef double x94 = nu - 1.0
        cdef double x96 = x45*x94
        cdef double x95 = x25*x94
        cdef double x97 = x95 + x96
        cdef double x52 = 4.0*nu
        cdef double dBnpdr = (
            r*x109*x126*x85
            + r*x126*x130*x134
            - x11*(
                0.5625*x22*(-x52 + x97 + 7.0)
                + 1.125*x29*(14.0*X_2 + x95 - x96 - 7.0)
                + 0.5625*x7*(8.0*nu + x97 + 23.0)
            )
            + x126*x130*x85
            - x133*x85
            - x15
            - x17*(
                0.0625*x22*(-773.0*nu - x101 + 4.0*x24 - 13.0)
                + 0.125*x29*(-x100 - x102*(166.0*nu - 601.0) - x99)
                + 0.0625*x7*(-2059.0*nu - x101 - 837.0)
            )
        )
        cdef double x87 = r**(-6)
        cdef double x56 = X_2**2*chi_2**2*(1.0 - CES22)
        cdef double x55 = X_1**2*chi_1**2*(1.0 - CES21)
        cdef double x49 = X_2 - 1.0
        cdef double x54 = x29*(CES21*x49 + X_2*(CES22 + 0.5) - 0.25)
        cdef double x50 = -X_2*x45 + x25*x49
        cdef double x53 = x22*(x50 + x52 + 5.0)
        cdef double x51 = x7*(x50 + 13.0)
        cdef double x41 = 3.0*CES42
        cdef double x40 = 3.0*CES41
        cdef double x42 = x40 + x41
        cdef double x48 = x22*(CES21*(x31 + 4.0) - x42 + x45 - 8.0)
        cdef double x46 = x21*x6**3*(-6.0*CBS31 + 6.0*CBS32 + 4.0*CES21 - x40 + x41 - x45)
        cdef double x32 = CES41 + CES42
        cdef double x44 = x22*x7*(2.0*CES21*CES22 - x32)
        cdef double x43 = x6**4*(-18.0*CES21*CES22 + 8.0*CES21 + 8.0*CES22 - x38 - x39 - x42 + 32.0)
        cdef double x37 = x21**3*x6*(2.0*CBS31 - 2.0*CBS32 - CES41 + CES42)
        cdef double x36 = x21**4*(-CES21*x31 - x32 - x35)
        cdef double x28 = 182.0*nu
        cdef double x19 = 34.0*X_2 + 69.0*nu - 34.0
        cdef double x26 = x19*x25
        cdef double x23 = 276.0*CES22*nu
        cdef double x30 = x29*(2.0*X_2*(68.0*CES22 + x28 - 409.0) - x23 + x26 - x28 + 409.0)
        cdef double x27 = x22*(-741.0*nu - 136.0*x18 + x23 + 196.0*x24 + x26 + 115.0)
        cdef double x20 = x7*(12.0*CES21*x19 + 828.0*CES22*nu - 2881.0*nu - 408.0*x18 - 1167.0)
        cdef double x1 = 2.0*x0 + 1.0
        cdef double x9 = x1*x8 + 1.0
        cdef double dAdr = (
            (
                dTidal1
                + dTidal2
                + 7680.0*x10*x82*x89
                - x11*(3.0*x55 + 3.0*x56)
                + 30720.0*x13*x60*x82
                - x15
                - x17*(0.5*x51 + 0.5*x53 + 4.0*x54)
                - x87*(0.0111607142857143*x20 + 0.0334821428571429*x27 + 0.0669642857142857*x30)
                - x87*(0.234375*x36 + 0.9375*x37 + 0.078125*x43 + 0.9375*x44 + 0.3125*x46 + 0.15625*x48*x7)
                - x93
            )/x9
            - (
                -x1*x15 - x11*x12
            )*(
                x11*(0.125*x51 + 0.125*x53 + x54)
                + x14*(x55 + x56)
                + x17*(0.00223214285714286*x20 + 0.00669642857142857*x27 + 0.0133928571428571*x30)
                + x17*(0.046875*x36 + 0.1875*x37 + 0.015625*x43 + 0.1875*x44 + 0.0625*x46 + x47*x48)
                + x86
            )/x9**2
        )
        # Return the auxiliary derivatives of H
        return [dAdr, dBnpdr, dBnpadr, dxidr, dQdr, dQdprst, dHodddr]
