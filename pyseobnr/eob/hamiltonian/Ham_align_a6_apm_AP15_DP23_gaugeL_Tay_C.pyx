
# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, profile=True,linetrace=True,binding=True
cimport cython
import numpy as np
cimport numpy as np
from pyseobnr.eob.utils.containers cimport EOBParams,CalibCoeffs

from pyseobnr.eob.hamiltonian.Hamiltonian_C cimport Hamiltonian_C
from libc.math cimport log, sqrt, exp, abs, tgamma,sin,cos

cpdef (double,double) evaluate_H(double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2,double M, double nu,
double X_1, double X_2, double a6, double dSO):
        """
        Evaluate the Hamiltonian and xi

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.
          M (double): Total mass.
          nu (double): Reduced mass ratio.
          X_1 (double): m_1/M
          X_2 (double): m_2/M
          a6 (doubble): nonspinning calibration parameter
          dSO (double): spin-orbit calibration parameter

        Returns:
           (tuple)  H,xi

        """
        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double pphi = L


        cdef double r2 = r*r
        cdef double r3 = r2*r
        cdef double r4 = r2*r2
        cdef double r5 = r*r4

        cdef double L2 = L*L
        cdef double L4 = L2*L2
        cdef double lr = log(r)

        cdef double nu2 = nu*nu
        cdef double nu3 = nu2*nu
        cdef double nu4 = nu3*nu

        cdef double prst2 = prst*prst
        cdef double prst4 = prst2*prst2
        cdef double prst6 = prst4*prst2
        cdef double prst8 = prst6*prst2

        # Actual Hamiltonian expressions
        cdef double d5 = 0

        cdef double Dbpm = r*(6730497718123.02*nu3 + 22295347200.0*nu2*d5 + 133772083200.0*nu2*r2 + 1822680546449.21*nu2*r + 80059249540278.2*nu2 + 22295347200.0*nu*d5*r - 193226342400.0*nu*d5 + 2589101062873.81*nu*r2 + 10611661054566.2*nu*r - 12049908701745.2*nu + 5107745331375.71*r2 - 326837426.241486*r*(14700.0*nu + 42911.0) - 39476764256925.6*r - (-5041721180160.0*nu2 - 25392914995744.3*nu - 879923036160.0*r2 - 283115520.0*r*(14700.0*nu + 42911.0) + 104186110149937.0)*lr + 5787938193408.0*lr**2 + 275059053208689.0)/(55296.0*nu*(14515200.0*nu3 - 42636451.6032331*nu2 - 7680.0*nu*(315.0*d5 + 890888.810272497) + 4331361844.61149*nu + 1002013764.01019) - 967680.0*r3*(-138240.0*nu2 - 2675575.66847905*nu - 5278341.3229329) - 9216.0*r2*(-197773496.793534*nu2 - 7680.0*nu*(315.0*d5 + 405152.309729121) + 2481453539.84635*nu + 5805304367.87913) + r*(5927865218923.02*nu3 + 70778880.0*nu2*(315.0*d5 + 2561145.80918574) - 138141470005001.0*nu2 - 4718592.0*nu*(40950.0*d5 + 86207832.4415642) + 450172889755120.0*nu + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0) + 5787938193408.0*r*lr**2 + (-1698693120.0*nu*(11592.0*nu + 69847.0) + 879923036160.0*r3 + 283115520.0*r2*(14700.0*nu + 42911.0) + 49152.0*r*(102574080.0*nu2 + 409207698.136075*nu - 2119671837.36038))*lr)

        cdef double Apm = 7680.0*r4*(-5416406.59541186*nu2 + 28.0*nu*(1920.0*a6 + 733955.307463037) + 2048.0*nu*(756.0*nu + 336.0*r + 407.0)*lr - 7.0*r*(-185763.092693281*nu2 + 938918.400156317*nu - 245760.0) - 3440640.0)/(241555486248.807*nu4 + 1120.0*nu3*(-17833256.898555*r2 - 163683964.822551*r - 1188987459.03162) + 7.0*nu2*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 1426660551.8844*r5 - 3089250703.76879*r4 - 6178501407.53758*r3 + 2064783811.32587*r2 + 122635399361.987*r + 276057889687.011) + 67645734912.0*nu2*lr**2 + 53760.0*nu*(7680.0*a6*(r4 + 2.0*r3 + 4.0*r2 + 8.0*r + 16.0) + 128.0*r*(-6852.34813868015*r4 + 4264.6962773603*r3 + 8529.39255472061*r2 + 13218.7851094412*r - 33722.4297811176) + 113485.217444961*r*(-r4 + 2.0*r3 + 4.0*r2 + 8.0*r + 16.0) + 148.04406601634*r*(349.0*r4 + 1926.0*r3 + 3852.0*r2 + 7704.0*r + 36400.0)) + 32768.0*nu*(-1882456.23663972*nu2 - 38842241.4769507*nu + 161280.0*r5 + 480.0*r4*(756.0*nu + 1079.0) + 960.0*r3*(756.0*nu + 1079.0) + 1920.0*r2*(588.0*nu + 1079.0) + 240.0*r*(-3024.0*nu2 - 7466.27061066206*nu + 17264.0) + 13447680.0)*lr + 13212057600.0*r5)

        cdef double ap = chi_1*X_1 + chi_2*X_2

        cdef double ap2 = ap*ap

        cdef double xi = sqrt(Dbpm)*r2*(Apm + ap2/r2)/(ap2 + r2)

        cdef double pr = prst/xi

        cdef double flagNLOSS2 = 1.00000000000000

        cdef double delta = X_1 - X_2

        cdef double am = chi_1*X_1 - chi_2*X_2

        cdef double apam = am*ap

        cdef double am2 = am*am

        cdef double apamd = apam*delta

        cdef double QSalign2 = flagNLOSS2*pr**4*(-0.46875*am2*(4.0*nu2 - 5.0*nu + 1.0) - 0.15625*ap2*(32.0*nu2 - 33.0*nu - 5.0) + 0.3125*apamd*(18.0*nu - 1.0))/r3

        cdef double flagQPN55 = 1.00000000000000

        cdef double flagQPN5 = 1.00000000000000

        cdef double flagQPN4 = 1.00000000000000

        cdef double Qpm = flagQPN4*(0.121954868780449*nu*prst8/r + prst6*(6.0*nu3 - 5.4*nu2 - 2.78300763695006*nu)/r2 + prst4*(10.0*nu3 - 131.0*nu2 + 92.7110442849544*nu)/r3) + flagQPN5*(prst8*(-6.0*nu4 + 3.42857142857143*nu3 + 3.33842023648322*nu2 + 1.38977750996128*nu)/r2 + prst6*(-14.0*nu4 + 188.0*nu3 - 89.5298327361234*nu2 - 33.9782122170436*nu)/r3 + prst4*(602.318540416564*nu3 + nu2*(118.4*lr - 1796.13660498019) + nu*(452.542166996693 - 51.6952380952381*lr))/r4) + flagQPN55*(1.48275342024365*nu*prst8/r**2.5 - 11.3175085791863*nu*prst6/r**3.5 + 147.443752990146*nu*prst4/r**4.5) + prst4*(-6.0*nu2 + 8.0*nu)/r2

        cdef double Qq = QSalign2 + Qpm

        cdef double Bnpa = -r*(r + 2.0)/(ap2*r*(r + 2.0) + r4)

        cdef double flagNLOSS = 1.00000000000000

        cdef double BnpSalign2 = flagNLOSS*(0.1875*am2*(4.0*nu - 1.0) + ap2*(3.0*nu + 2.8125) - 2.625*apamd)/r3 + flagNLOSS2*(0.015625*am2*(4.0*nu2 + 115.0*nu - 37.0) + 0.015625*ap2*(-1171.0*nu - 861.0) + 0.03125*apamd*(26.0*nu + 449.0))/r4

        cdef double Bnp = Apm*Dbpm + BnpSalign2 + ap2/r2 - 1.0

        cdef double ASalignCal2 = 0.0

        cdef double ASalign2 = flagNLOSS*(0.125*am2*(4.0*nu + 1.0) + 1.125*ap2 - 1.25*apamd)/r4 + flagNLOSS2*(0.046875*am2*(28.0*nu2 - 27.0*nu - 3.0) - 0.390625*ap2*(7.0*nu + 9.0) - 1.21875*apamd*(2.0*nu - 3.0))/r**5

        cdef double A = (ASalign2 + ASalignCal2 + Apm + ap2/r2)/(ap2*(1.0 + 2.0/r)/r2 + 1.0)

        cdef double lap = ap

        cdef double Heven = sqrt(A*(Bnpa*L2*lap**2/r2 + L2/r2 + Qq + prst2*(Bnp + 1.0)/xi**2 + 1.0))

        cdef double lam = am

        cdef double Ga3 = 0.0416666666666667*L*ap2*delta*lam/r2 + L*lap*(-0.25*ap2 + 0.208333333333333*apamd)/r2

        cdef double SOcalib = L*nu*dSO*lap/r3

        cdef double flagNLOSO2 = 1.00000000000000

        cdef double flagNLOSO = 1.00000000000000

        cdef double gam = flagNLOSO*(L2*(0.46875 - 0.28125*nu)/r2 + (0.34375*nu + 0.09375)/r) + flagNLOSO2*(L4*(0.29296875*nu2 - 0.3515625*nu - 0.41015625)/r4 + L2*(-0.798177083333333*nu2 - 0.2734375*nu - 0.23046875)/r3 + (0.536458333333333*nu2 - 0.03125*nu + 0.078125)/r2) + 0.25

        cdef double gap = flagNLOSO*(L2*(-1.40625*nu - 0.46875)/r2 + (0.71875*nu - 0.09375)/r) + flagNLOSO2*(L4*(1.34765625*nu2 + 0.5859375*nu + 0.41015625)/r4 + L2*(-2.07161458333333*nu2 - 2.0859375*nu + 0.23046875)/r3 + (0.567708333333333*nu2 - 5.53125*nu - 0.078125)/r2) + 1.75

        cdef double Hodd = (Ga3 + L*delta*gam*lam + L*gap*lap + SOcalib)/(2.0*ap2 + 2.0*r2 + r*(ap2 + r2 - 2.0*r))

        cdef double Heff = Heven + Hodd

        # Evaluate H_real/nu
        cdef double H = M * sqrt(1+2*nu*(Heff-1)) / nu

        return H,xi



cdef class Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C(Hamiltonian_C):
    cdef public CalibCoeffs calibration_coeffs
    #cdef public EOBParams EOBpars
    def __cinit__(self,EOBParams params):
        self.EOBpars = params

    def __call__(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2,bint verbose=False):
        return self._call(q,p,chi_1,chi_2,m_1,m_2,verbose=verbose)

    cpdef _call(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2,bint verbose=False):

        """
        Evaluate the aligned-spin SEOBNRv5HM Hamiltonian as well as several potentials.
        See Sec. 1B and 1C of
        https://dcc.ligo.org/DocDB/0186/T2300060/001/SEOBNRv5_theory.pdf

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.
          verbose (bint): Output additional potentials.

        Returns:
           (tuple)  H,xi, A, Bnp, Bnpa, Qq, Heven, Hodd

        """

        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double pphi = L

        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs['a6']
        cdef double dSO = c_coeffs['dSO']


        # Extra quantities used in the Hamiltonian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2

        cdef double r2 = r*r
        cdef double r3 = r2*r
        cdef double r4 = r2*r2
        cdef double r5 = r*r4
        cdef double L2 = L*L
        cdef double lr = log(r)

        cdef double nu2 = nu*nu
        cdef double nu3 = nu2*nu
        cdef double nu4 = nu3*nu
        # Actual Hamiltonian expressions
        cdef double d5 = 0

        cdef double Dbpm = r*(6730497718123.02*nu3 + 22295347200.0*nu2*d5 + 133772083200.0*nu2*r2 + 1822680546449.21*nu2*r + 80059249540278.2*nu2 + 22295347200.0*nu*d5*r - 193226342400.0*nu*d5 + 2589101062873.81*nu*r2 + 10611661054566.2*nu*r - 12049908701745.2*nu + 5107745331375.71*r2 - 326837426.241486*r*(14700.0*nu + 42911.0) - 39476764256925.6*r - (-5041721180160.0*nu2 - 25392914995744.3*nu - 879923036160.0*r2 - 283115520.0*r*(14700.0*nu + 42911.0) + 104186110149937.0)*lr + 5787938193408.0*lr**2 + 275059053208689.0)/(55296.0*nu*(14515200.0*nu3 - 42636451.6032331*nu2 - 7680.0*nu*(315.0*d5 + 890888.810272497) + 4331361844.61149*nu + 1002013764.01019) - 967680.0*r3*(-138240.0*nu2 - 2675575.66847905*nu - 5278341.3229329) - 9216.0*r2*(-197773496.793534*nu2 - 7680.0*nu*(315.0*d5 + 405152.309729121) + 2481453539.84635*nu + 5805304367.87913) + r*(5927865218923.02*nu3 + 70778880.0*nu2*(315.0*d5 + 2561145.80918574) - 138141470005001.0*nu2 - 4718592.0*nu*(40950.0*d5 + 86207832.4415642) + 450172889755120.0*nu + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0) + 5787938193408.0*r*lr**2 + (-1698693120.0*nu*(11592.0*nu + 69847.0) + 879923036160.0*r3 + 283115520.0*r2*(14700.0*nu + 42911.0) + 49152.0*r*(102574080.0*nu2 + 409207698.136075*nu - 2119671837.36038))*lr)

        cdef double Apm = 7680.0*r4*(-5416406.59541186*nu2 + 28.0*nu*(1920.0*a6 + 733955.307463037) + 2048.0*nu*(756.0*nu + 336.0*r + 407.0)*lr - 7.0*r*(-185763.092693281*nu2 + 938918.400156317*nu - 245760.0) - 3440640.0)/(241555486248.807*nu4 + 1120.0*nu3*(-17833256.898555*r2 - 163683964.822551*r - 1188987459.03162) + 7.0*nu2*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 1426660551.8844*r5 - 3089250703.76879*r4 - 6178501407.53758*r3 + 2064783811.32587*r2 + 122635399361.987*r + 276057889687.011) + 67645734912.0*nu2*lr**2 + 53760.0*nu*(7680.0*a6*(r4 + 2.0*r3 + 4.0*r2 + 8.0*r + 16.0) + 128.0*r*(-6852.34813868015*r4 + 4264.6962773603*r3 + 8529.39255472061*r2 + 13218.7851094412*r - 33722.4297811176) + 113485.217444961*r*(-r4 + 2.0*r3 + 4.0*r2 + 8.0*r + 16.0) + 148.04406601634*r*(349.0*r4 + 1926.0*r3 + 3852.0*r2 + 7704.0*r + 36400.0)) + 32768.0*nu*(-1882456.23663972*nu2 - 38842241.4769507*nu + 161280.0*r5 + 480.0*r4*(756.0*nu + 1079.0) + 960.0*r3*(756.0*nu + 1079.0) + 1920.0*r2*(588.0*nu + 1079.0) + 240.0*r*(-3024.0*nu2 - 7466.27061066206*nu + 17264.0) + 13447680.0)*lr + 13212057600.0*r5)

        cdef double ap = chi_1*X_1 + chi_2*X_2

        cdef double ap2 = ap*ap

        cdef double xi = sqrt(Dbpm)*r2*(Apm + ap2/r2)/(ap2 + r2)

        cdef double pr = prst/xi

        cdef double flagNLOSS2 = 1.00000000000000

        cdef double delta = X_1 - X_2

        cdef double am = chi_1*X_1 - chi_2*X_2

        cdef double apam = am*ap

        cdef double am2 = am*am

        cdef double apamd = apam*delta

        cdef double QSalign2 = flagNLOSS2*pr**4*(-0.46875*am2*(4.0*nu2 - 5.0*nu + 1.0) - 0.15625*ap2*(32.0*nu2 - 33.0*nu - 5.0) + 0.3125*apamd*(18.0*nu - 1.0))/r3

        cdef double flagQPN55 = 1.00000000000000

        cdef double flagQPN5 = 1.00000000000000

        cdef double flagQPN4 = 1.00000000000000

        cdef double Qpm = flagQPN4*(0.121954868780449*nu*prst**8/r + prst**6*(6.0*nu3 - 5.4*nu2 - 2.78300763695006*nu)/r2 + prst**4*(10.0*nu3 - 131.0*nu2 + 92.7110442849544*nu)/r3) + flagQPN5*(prst**8*(-6.0*nu4 + 3.42857142857143*nu3 + 3.33842023648322*nu2 + 1.38977750996128*nu)/r2 + prst**6*(-14.0*nu4 + 188.0*nu3 - 89.5298327361234*nu2 - 33.9782122170436*nu)/r3 + prst**4*(602.318540416564*nu3 + nu2*(118.4*lr - 1796.13660498019) + nu*(452.542166996693 - 51.6952380952381*lr))/r4) + flagQPN55*(1.48275342024365*nu*prst**8/r**2.5 - 11.3175085791863*nu*prst**6/r**3.5 + 147.443752990146*nu*prst**4/r**4.5) + prst**4*(-6.0*nu2 + 8.0*nu)/r2

        cdef double Qq = QSalign2 + Qpm

        cdef double Bnpa = -r*(r + 2.0)/(ap2*r*(r + 2.0) + r4)

        cdef double flagNLOSS = 1.00000000000000

        cdef double BnpSalign2 = flagNLOSS*(0.1875*am2*(4.0*nu - 1.0) + ap2*(3.0*nu + 2.8125) - 2.625*apamd)/r3 + flagNLOSS2*(0.015625*am2*(4.0*nu2 + 115.0*nu - 37.0) + 0.015625*ap2*(-1171.0*nu - 861.0) + 0.03125*apamd*(26.0*nu + 449.0))/r4

        cdef double Bnp = Apm*Dbpm + BnpSalign2 + ap2/r2 - 1.0

        cdef double ASalignCal2 = 0.0

        cdef double ASalign2 = flagNLOSS*(0.125*am2*(4.0*nu + 1.0) + 1.125*ap2 - 1.25*apamd)/r4 + flagNLOSS2*(0.046875*am2*(28.0*nu2 - 27.0*nu - 3.0) - 0.390625*ap2*(7.0*nu + 9.0) - 1.21875*apamd*(2.0*nu - 3.0))/r**5

        cdef double A = (ASalign2 + ASalignCal2 + Apm + ap2/r2)/(ap2*(1.0 + 2.0/r)/r2 + 1.0)

        cdef double lap = ap

        cdef double Heven = sqrt(A*(Bnpa*L2*lap**2/r2 + L2/r2 + Qq + prst**2*(Bnp + 1.0)/xi**2 + 1.0))

        cdef double lam = am

        cdef double Ga3 = 0.0416666666666667*L*ap2*delta*lam/r2 + L*lap*(-0.25*ap2 + 0.208333333333333*apamd)/r2

        cdef double SOcalib = L*nu*dSO*lap/r3

        cdef double flagNLOSO2 = 1.00000000000000

        cdef double flagNLOSO = 1.00000000000000

        cdef double gam = flagNLOSO*(L2*(0.46875 - 0.28125*nu)/r2 + (0.34375*nu + 0.09375)/r) + flagNLOSO2*(L**4*(0.29296875*nu2 - 0.3515625*nu - 0.41015625)/r4 + L2*(-0.798177083333333*nu2 - 0.2734375*nu - 0.23046875)/r3 + (0.536458333333333*nu2 - 0.03125*nu + 0.078125)/r2) + 0.25

        cdef double gap = flagNLOSO*(L2*(-1.40625*nu - 0.46875)/r2 + (0.71875*nu - 0.09375)/r) + flagNLOSO2*(L**4*(1.34765625*nu2 + 0.5859375*nu + 0.41015625)/r4 + L2*(-2.07161458333333*nu2 - 2.0859375*nu + 0.23046875)/r3 + (0.567708333333333*nu2 - 5.53125*nu - 0.078125)/r2) + 1.75

        cdef double Hodd = (Ga3 + L*delta*gam*lam + L*gap*lap + SOcalib)/(2.0*ap2 + 2.0*r2 + r*(ap2 + r2 - 2.0*r))

        cdef double Heff = Heven + Hodd



        # Evaluate H_real/nu
        cdef double H = M * sqrt(1+2*nu*(Heff-1)) / nu
        if not verbose:
            return H,xi
        else:
            return H,xi,A,Bnp,Bnpa,Qq,Heven,Hodd

    cpdef grad(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2):

        """
        Compute the gradient of the Hamiltonian in polar coordinates.

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.

        Returns:
           (tuple) dHdr, dHdphi, dHdpr, dHdpphi

        """

        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double pphi = L

        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs['a6']
        cdef double dSO = c_coeffs['dSO']


        # Extra quantities used in the Jacobian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double x0 = r**2
        cdef double x1 = chi_1*X_1
        cdef double x2 = chi_2*X_2
        cdef double x3 = x1 + x2
        cdef double x4 = x3**2
        cdef double x5 = 2.0*x4
        cdef double x6 = 2.0*r
        cdef double x7 = x0 + x4
        cdef double x8 = r*(-x6 + x7)
        cdef double x9 = (2.0*x0 + x5 + x8)**(-1)
        cdef double x10 = r**4
        cdef double x11 = x10**(-1)
        cdef double x12 = 3.0*nu
        cdef double x13 = pphi*x3
        cdef double x14 = r**3
        cdef double x15 = x14**(-1)
        cdef double x16 = x15*x4
        cdef double x17 = x1 - x2
        cdef double x18 = X_1 - X_2
        cdef double x19 = x17*x18
        cdef double x20 = pphi*x19
        cdef double x21 = 5.0*X_1 - 5.0*X_2
        cdef double x22 = x17*x3
        cdef double x23 = 0.0416666666666667*x21*x22 - 0.25*x4
        cdef double x24 = 2.0*x15
        cdef double x25 = x0**(-1)
        cdef double x26 = 0.71875*nu - 0.09375
        cdef double x27 = -1.40625*nu - 0.46875
        cdef double x28 = pphi**2
        cdef double x29 = x15*x28
        cdef double x30 = 2.0*x29
        cdef double x31 = nu**2
        cdef double x32 = -2.0859375*nu - 2.07161458333333*x31 + 0.23046875
        cdef double x33 = 3.0*x11
        cdef double x34 = x28*x33
        cdef double x35 = 0.5859375*nu + 1.34765625*x31 + 0.41015625
        cdef double x36 = pphi**4
        cdef double x37 = r**5
        cdef double x38 = x37**(-1)
        cdef double x39 = 4.0*x38
        cdef double x40 = x36*x39
        cdef double x41 = 0.34375*nu + 0.09375
        cdef double x42 = 0.46875 - 0.28125*nu
        cdef double x43 = -0.2734375*nu - 0.798177083333333*x31 - 0.23046875
        cdef double x44 = -0.3515625*nu + 0.29296875*x31 - 0.41015625
        cdef double x45 = nu*dSO*x15
        cdef double x46 = x25*x4
        cdef double x47 = 0.0416666666666667*x46
        cdef double x48 = x23*x25
        cdef double x49 = r**(-1)
        cdef double x50 = x25*x28
        cdef double x51 = x11*x36
        cdef double x52 = x3*(x25*(-5.53125*nu + 0.567708333333333*x31 - 0.078125) + x26*x49 + x27*x50 + x29*x32 + x35*x51 + 1.75)
        cdef double x53 = x19*(x25*(-0.03125*nu + 0.536458333333333*x31 + 0.078125) + x29*x43 + x41*x49 + x42*x50 + x44*x51 + 0.25)
        cdef double x54 = prst**4
        cdef double x55 = r**(-4.5)
        cdef double x56 = nu*x55
        cdef double x57 = prst**6
        cdef double x58 = r**(-3.5)
        cdef double x59 = nu*x58
        cdef double x60 = r**(-2.5)
        cdef double x61 = prst**8
        cdef double x62 = nu*x61
        cdef double x63 = nu*x49
        cdef double x64 = 0.121954868780449*x61
        cdef double x65 = 8.0*nu - 6.0*x31
        cdef double x66 = x25*x65
        cdef double x67 = nu**3
        cdef double x68 = 92.7110442849544*nu - 131.0*x31 + 10.0*x67
        cdef double x69 = x15*x54
        cdef double x70 = -2.78300763695006*nu - 5.4*x31 + 6.0*x67
        cdef double x71 = x25*x70
        cdef double x72 = nu**4
        cdef double x73 = -33.9782122170436*nu - 89.5298327361234*x31 + 188.0*x67 - 14.0*x72
        cdef double x74 = x15*x57
        cdef double x75 = 1.38977750996128*nu + 3.33842023648322*x31 + 3.42857142857143*x67 - 6.0*x72
        cdef double x76 = x61*x75
        cdef double x77 = log(r)
        cdef double x78 = nu*(452.542166996693 - 51.6952380952381*x77) + x31*(118.4*x77 - 1796.13660498019) + 602.318540416564*x67
        cdef double x79 = x11*x54
        cdef double x80 = r + 2.0
        cdef double x81 = x4*x80
        cdef double x82 = r*x4
        cdef double x83 = x10 + x80*x82
        cdef double x84 = x83**(-1)
        cdef double x85 = x28*x49*x84
        cdef double x86 = r**(-13)
        cdef double x87 = x7**4
        cdef double x88 = 756.0*nu
        cdef double x89 = 336.0*r + x88 + 407.0
        cdef double x90 = 2048.0*nu*x77*x89 + 28.0*nu*(1920.0*a6 + 733955.307463037) - 7.0*r*(938918.400156317*nu - 185763.092693281*x31 - 245760.0) - 5416406.59541186*x31 - 3440640.0
        cdef double x91 = x77**2
        cdef double x92 = x31*x91
        cdef double x93 = x67*(-163683964.822551*r - 17833256.898555*x0 - 1188987459.03162)
        cdef double x94 = x31*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 122635399361.987*r + 2064783811.32587*x0 - 3089250703.76879*x10 - 6178501407.53758*x14 + 1426660551.8844*x37 + 276057889687.011)
        cdef double x95 = 588.0*nu + 1079.0
        cdef double x96 = x88 + 1079.0
        cdef double x97 = x14*x96
        cdef double x98 = -38842241.4769507*nu + 240.0*r*(-7466.27061066206*nu - 3024.0*x31 + 17264.0) + 1920.0*x0*x95 + 480.0*x10*x96 - 1882456.23663972*x31 + 161280.0*x37 + 960.0*x97 + 13447680.0
        cdef double x99 = nu*x77
        cdef double x100 = x98*x99
        cdef double x101 = 8.0*r
        cdef double x102 = 4.0*x0 + x101 + 2.0*x14 + 16.0
        cdef double x103 = 7680.0*a6
        cdef double x104 = 128.0*r
        cdef double x105 = 7704.0*r
        cdef double x106 = 148.04406601634*r
        cdef double x107 = 113485.217444961*r
        cdef double x108 = nu*(x103*(x10 + x102) + x104*(13218.7851094412*r + 8529.39255472061*x0 - 6852.34813868015*x10 + 4264.6962773603*x14 - 33722.4297811176) + x106*(3852.0*x0 + 349.0*x10 + x105 + 1926.0*x14 + 36400.0) + x107*(-x10 + x102))
        cdef double x109 = (32768.0*x100 + 53760.0*x108 + 13212057600.0*x37 + 241555486248.807*x72 + 67645734912.0*x92 + 1120.0*x93 + 7.0*x94)**(-1)
        cdef double x110 = x10*x109*x90
        cdef double x111 = x110 + 0.000130208333333333*x46
        cdef double x112 = x111**(-4)
        cdef double x113 = r*x91
        cdef double x114 = -630116198.873299*nu - 197773496.793534*x31 + 5805304367.87913
        cdef double x115 = x0*x114
        cdef double x116 = -2675575.66847905*nu - 138240.0*x31 - 5278341.3229329
        cdef double x117 = x116*x14
        cdef double x118 = nu*(-2510664218.28128*nu - 42636451.6032331*x31 + 14515200.0*x67 + 1002013764.01019)
        cdef double x119 = 43393301259014.8*nu + 43133561885859.3*x31 + 5927865218923.02*x67 + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0
        cdef double x120 = r*x119
        cdef double x121 = 14700.0*nu + 42911.0
        cdef double x122 = 283115520.0*x121
        cdef double x123 = -1698693120.0*nu*(11592.0*nu + 69847.0) + 49152.0*r*(409207698.136075*nu + 102574080.0*x31 - 2119671837.36038) + x0*x122 + 879923036160.0*x14
        cdef double x124 = x123*x77
        cdef double x125 = (x113 - 1.59227685093395e-9*x115 - 1.67189069348064e-7*x117 + 9.55366110560367e-9*x118 + 1.72773095804465e-13*x120 + 1.72773095804465e-13*x124)**2
        cdef double x126 = nu*r
        cdef double x127 = nu*x0
        cdef double x128 = r*x31
        cdef double x129 = r*x121
        cdef double x130 = x0*x31
        cdef double x131 = 5041721180160.0*x31 - 104186110149937.0
        cdef double x132 = -25392914995744.3*nu - r*x122 - 879923036160.0*x0 - x131
        cdef double x133 = x132*x77
        cdef double x134 = -0.0438084424460039*nu - 0.143521050466841*r + 0.0185696317637669*x0 + 0.0385795738434214*x126 + 0.00941289164152486*x127 + 0.00662650629087394*x128 - 1.18824456940711e-6*x129 + 0.000486339502879429*x130 - 3.63558293513537e-15*x133 + 0.291062041428379*x31 + 0.0244692826489756*x67 + 0.0210425293255724*x91 + 1
        cdef double x135 = x134**(-2)
        cdef double x136 = 4.0*x31
        cdef double x137 = x17**2
        cdef double x138 = -0.46875*x137*(-5.0*nu + x136 + 1.0) + 0.0625*x17*x21*x3*(18.0*nu - 1.0) - 0.15625*x4*(-33.0*nu + 32.0*x31 - 5.0)
        cdef double x139 = x112*x125*x135*x138*x54*x87
        cdef double x140 = x7**2
        cdef double x141 = prst**2
        cdef double x142 = x111**(-2)
        cdef double x143 = 1822680546449.21*x31
        cdef double x144 = 5787938193408.0*x91
        cdef double x145 = -12049908701745.2*nu + r*x143 - 39476764256925.6*r + 5107745331375.71*x0 + 10611661054566.2*x126 + 2589101062873.81*x127 - 326837426.241486*x129 + 133772083200.0*x130 - x133 + x144 + 80059249540278.2*x31 + 6730497718123.02*x67 + 275059053208689.0
        cdef double x146 = x145**(-1)
        cdef double x147 = 0.0625*x137
        cdef double x148 = 0.125*x22
        cdef double x149 = -1171.0*nu - 861.0
        cdef double x150 = 0.015625*x4
        cdef double x151 = x137*(115.0*nu + x136 - 37.0)
        cdef double x152 = 0.03125*x22
        cdef double x153 = x18*(26.0*nu + 449.0)
        cdef double x154 = 5787938193408.0*x113 - 9216.0*x115 - 967680.0*x117 + 55296.0*x118 + x120 + x124
        cdef double x155 = x154**(-1)
        cdef double x156 = x145*x155*x37
        cdef double x157 = x109*x90
        cdef double x158 = x11*(x149*x150 + 0.015625*x151 + x152*x153) + x15*(x147*(12.0*nu - 3.0) + x148*(-21.0*X_1 + 21.0*X_2) + x4*(x12 + 2.8125)) + 7680.0*x156*x157 + x46
        cdef double x159 = x140*x141*x142*x146*x154*x158
        cdef double x160 = 1.27277314139085e-19*x139*x86 + 1.69542100694444e-8*x159*x38 + x25*x76 + x50 + 147.443752990146*x54*x56 + x54*x66 - 11.3175085791863*x57*x59 + x57*x71 + 1.48275342024365*x60*x62 + x63*x64 + x68*x69 + x73*x74 + x78*x79 - x81*x85 + 1.0
        cdef double x161 = x46*(2.0*x49 + 1.0) + 1.0
        cdef double x162 = x161**(-1)
        cdef double x163 = x137*(4.0*nu + 1.0)
        cdef double x164 = -x21*x22
        cdef double x165 = x137*(-27.0*nu + 28.0*x31 - 3.0)
        cdef double x166 = x152*(-39.0*X_1 + 39.0*X_2)
        cdef double x167 = x11*(0.125*x163 + 0.25*x164 + 1.125*x4) + 7680.0*x110 + x38*(-x150*(175.0*nu + 225.0) + 0.046875*x165 + x166*(2.0*nu - 3.0)) + x46
        cdef double x168 = x162*x167
        cdef double x169 = (x160*x168)**(-0.5)
        cdef double x170 = 4.0*x49 + 2.0
        cdef double x171 = r**(-6)
        cdef double x172 = x15*x5
        cdef double x173 = -6572428.80109422*nu + 1300341.64885296*x31 + 2048.0*x63*x89 + 688128.0*x99 + 1720320.0
        cdef double x174 = x31*x49
        cdef double x175 = 4.0*x14
        cdef double x176 = 6.0*x0 + x101 + 8.0
        cdef double x177 = 1.31621673590926e-19*x90*(53760.0*nu*(3740417.71815805*r + 2115968.85907902*x0 - 938918.400156317*x10 + x103*(x175 + x176) + x104*(17058.7851094412*r + 12794.0888320809*x0 - 27409.3925547206*x14 + 13218.7851094412) + x106*(5778.0*x0 + x105 + 1396.0*x14 + 7704.0) + x107*(-x175 + x176) + 1057984.42953951*x14 + 2888096.47013111) + 66060288000.0*x10 + 135291469824.0*x174*x77 + 7.0*x31*(-117964800.0*a6 + 4129567622.65173*r - 18535504222.6128*x0 + 7133302759.42198*x10 - 12357002815.0752*x14 + 122635399361.987) + 32768.0*x63*x98 + 7.0*x67*(-5706642207.53758*r - 26189434371.6082) + 32768.0*x99*(-1791904.9465589*nu + 3840.0*r*x95 + 2880.0*x0*x96 + 806400.0*x10 - 725760.0*x31 + 1920.0*x97 + 4143360.0))/(1.35654132757922e-7*x100 + 2.22557561555966e-7*x108 + 0.0546957463279941*x37 + x72 + 0.28004222119933*x92 + 4.63661586574928e-9*x93 + 2.8978849160933e-11*x94)**2
        cdef double x178 = -7680.0*x10*x109*x173 + x10*x177 - 30720.0*x109*x14*x90 + x172
        cdef double x179 = 11575876386816.0*x77
        cdef double x180 = 5807150888816.34*nu + 10215490662751.4*r + 5178202125747.62*x126 + 267544166400.0*x128 - x132*x49 + x143 + x179*x49 + x77*(4161798144000.0*nu + 1759846072320.0*r + 12148770078720.0) - 53501685054374.1
        cdef double x181 = x125*x138*x54*x86*x87
        cdef double x182 = -18432.0*r*x114 - 2903040.0*x0*x116 + x119 + x123*x49 + x144 + x179 + x77*(20113376778784.3*nu + 2639769108480.0*x0 + 566231040.0*x129 + x131)
        cdef double x183 = x141*x158
        cdef double x184 = -x178
        cdef double x185 = x154*x38
        cdef double x186 = prst**3
        cdef double x187 = prst**5
        cdef double x188 = prst**7
        cdef double x189 = 4.0*x186
        cdef double x190 = 6.0*x187
        cdef double x191 = x167*x169/(x170*x46 + 2.0)
        cdef double x192 = 2.0*pphi
        cdef double x193 = x192*x25
        cdef double x194 = x15*x192
        cdef double x195 = 4.0*pphi**3*x11


        # Evaluate Hamiltonian
        cdef double H,xi
        H,xi =  evaluate_H(q,p,chi_1,chi_2,m_1,m_2,M,nu,X_1,X_2,a6,dSO)

        # Heff Jacobian expressions
        cdef double dHeffdr = 0.5*x169*(x160*x162*(-x171*(-x150*(875.0*nu + 1125.0) + 0.234375*x165 + x166*(10.0*nu - 15.0)) - x178 - x38*(0.5*x163 + x164 + 4.5*x4)) - x160*x167*(-x11*x5 - x16*x170)/x161**2 + x168*(-663.496888455656*nu*r**(-5.5)*x54 - nu*x25*x64 + 39.6112800271521*nu*x55*x57 + 6.78168402777778e-8*x11*x141*x142*x146*x154*x158*x7 + x11*x54*(118.4*x174 - 51.6952380952381*x63) + 7.59859378406358e-45*x112*x135*x138*x154*x182*x54*x86*x87 - 9.25454462627843e-34*x112*x180*x181/x134**3 - 2.24091649004576e-37*x135*x140*x142*x154*x180*x183*x38 + 1.69542100694444e-8*x140*x141*x142*x146*x154*x38*(38400.0*x10*x109*x145*x155*x90 + 7680.0*x109*x145*x155*x173*x37 + 7680.0*x109*x155*x180*x37*x90 - x11*(x147*(36.0*nu - 9.0) + x148*(-63.0*X_1 + 63.0*X_2) + x4*(9.0*nu + 8.4375)) - x156*x177 - x172 - x38*(x148*x153 + 0.0625*x149*x4 + 0.0625*x151) - 2.29252167428035e-22*x145*x157*x182*x37/x125) + 1.69542100694444e-8*x140*x141*x142*x146*x158*x182*x38 - 8.47710503472222e-8*x159*x171 - x24*x76 + x25*x28*x4*x80*x84 + x28*x4*x49*x80*(x175 + x81 + x82)/x83**2 - x30 - x33*x57*x73 - x39*x54*x78 - x4*x85 - 3.70688355060912*x58*x62 - 2.0*x65*x69 - 3.0*x68*x79 - 2.0*x70*x74 - 4.41515887225116e-12*x140*x146*x183*x184*x185/x111**3 - 6.62902677807736e-23*x135*x181*x184/x111**5 + 1.01821851311268e-18*x112*x125*x135*x138*x54*x7**3/r**12 - 1.65460508380811e-18*x139/r**14)) + x9*(-dSO*x11*x12*x13 + pphi*x17*x18*(-x15*(-0.0625*nu + 1.07291666666667*x31 + 0.15625) - x25*x41 - x30*x42 - x34*x43 - x40*x44) + pphi*x3*(-x15*(-11.0625*nu + 1.13541666666667*x31 - 0.15625) - x25*x26 - x27*x30 - x32*x34 - x35*x40) - x13*x23*x24 - 0.0833333333333333*x16*x20) - 0.25*(r*(x6 - 2.0) + x6 + x7)*(pphi*x52 + pphi*x53 + x13*x45 + x13*x48 + x20*x47)/(x7 + 0.5*x8)**2

        cdef double dHeffdphi = 0

        cdef double dHeffdpr = x191*(11.8620273619492*nu*x188*x60 + 3.39084201388889e-8*prst*x140*x142*x146*x158*x185 + x11*x189*x78 + 5.09109256556341e-19*x112*x125*x135*x138*x186*x86*x87 + x15*x189*x68 + x15*x190*x73 + 589.775011960583*x186*x56 - 67.9050514751178*x187*x59 + 8.0*x188*x25*x75 + 0.975638950243592*x188*x63 + x189*x66 + x190*x71)

        cdef double dHeffdpphi = x191*(2.0*pphi*x25 - pphi*x49*x5*x80*x84) + x9*(x13*(x193*x27 + x194*x32 + x195*x35) + x19*x47 + x20*(x193*x42 + x194*x43 + x195*x44) + x3*x45 + x3*x48 + x52 + x53)

        # Compute H Jacobian
        cdef double dHdr = M * M * dHeffdr / (nu*H)
        cdef double dHdphi = M * M * dHeffdphi / (nu*H)
        cdef double dHdpr = M * M * dHeffdpr / (nu*H)
        cdef double dHdpphi = M * M * dHeffdpphi / (nu*H)

        return dHdr, dHdphi, dHdpr, dHdpphi

    cpdef hessian(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2):
        
        """
        Evaluate the Hessian of the Hamiltonian.

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.

        Returns:
           (np.array)  d2Hdr2, d2Hdrdphi, d2Hdrdpr, d2Hdrdpphi, d2Hdrdphi, d2Hdphi2, d2Hdphidpr, d2Hdphidpphi, d2Hdrdpr, d2Hdphidpr, d2Hdpr2, d2Hdprdpphi, d2Hdrdpphi, d2Hdphidpphi, d2Hdprdpphi, d2Hdpphi2

        """

        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double pphi = L

        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs['a6']
        cdef double dSO = c_coeffs['dSO']


        # Extra quantities used in the Jacobian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double x0 = r**2
        cdef double x1 = chi_1*X_1
        cdef double x2 = chi_2*X_2
        cdef double x3 = x1 + x2
        cdef double x4 = x3**2
        cdef double x5 = 2.0*x4
        cdef double x6 = 2.0*r
        cdef double x7 = x0 + x4
        cdef double x8 = r*(-x6 + x7)
        cdef double x9 = (2.0*x0 + x5 + x8)**(-1)
        cdef double x10 = r**4
        cdef double x11 = x10**(-1)
        cdef double x12 = 3.0*nu
        cdef double x13 = pphi*x3
        cdef double x14 = r**3
        cdef double x15 = x14**(-1)
        cdef double x16 = x15*x4
        cdef double x17 = x1 - x2
        cdef double x18 = X_1 - X_2
        cdef double x19 = x17*x18
        cdef double x20 = pphi*x19
        cdef double x21 = 5.0*X_1 - 5.0*X_2
        cdef double x22 = x17*x3
        cdef double x23 = 0.0416666666666667*x21*x22 - 0.25*x4
        cdef double x24 = 2.0*x15
        cdef double x25 = x0**(-1)
        cdef double x26 = 0.71875*nu - 0.09375
        cdef double x27 = -1.40625*nu - 0.46875
        cdef double x28 = pphi**2
        cdef double x29 = x15*x28
        cdef double x30 = 2.0*x29
        cdef double x31 = nu**2
        cdef double x32 = -2.0859375*nu - 2.07161458333333*x31 + 0.23046875
        cdef double x33 = 3.0*x11
        cdef double x34 = x28*x33
        cdef double x35 = 0.5859375*nu + 1.34765625*x31 + 0.41015625
        cdef double x36 = pphi**4
        cdef double x37 = r**5
        cdef double x38 = x37**(-1)
        cdef double x39 = 4.0*x38
        cdef double x40 = x36*x39
        cdef double x41 = 0.34375*nu + 0.09375
        cdef double x42 = 0.46875 - 0.28125*nu
        cdef double x43 = -0.2734375*nu - 0.798177083333333*x31 - 0.23046875
        cdef double x44 = -0.3515625*nu + 0.29296875*x31 - 0.41015625
        cdef double x45 = nu*dSO*x15
        cdef double x46 = x25*x4
        cdef double x47 = 0.0416666666666667*x46
        cdef double x48 = x23*x25
        cdef double x49 = r**(-1)
        cdef double x50 = x25*x28
        cdef double x51 = x11*x36
        cdef double x52 = x3*(x25*(-5.53125*nu + 0.567708333333333*x31 - 0.078125) + x26*x49 + x27*x50 + x29*x32 + x35*x51 + 1.75)
        cdef double x53 = x19*(x25*(-0.03125*nu + 0.536458333333333*x31 + 0.078125) + x29*x43 + x41*x49 + x42*x50 + x44*x51 + 0.25)
        cdef double x54 = prst**4
        cdef double x55 = r**(-4.5)
        cdef double x56 = nu*x55
        cdef double x57 = prst**6
        cdef double x58 = r**(-3.5)
        cdef double x59 = nu*x58
        cdef double x60 = r**(-2.5)
        cdef double x61 = prst**8
        cdef double x62 = nu*x61
        cdef double x63 = nu*x49
        cdef double x64 = 0.121954868780449*x61
        cdef double x65 = 8.0*nu - 6.0*x31
        cdef double x66 = x25*x65
        cdef double x67 = nu**3
        cdef double x68 = 92.7110442849544*nu - 131.0*x31 + 10.0*x67
        cdef double x69 = x15*x54
        cdef double x70 = -2.78300763695006*nu - 5.4*x31 + 6.0*x67
        cdef double x71 = x25*x70
        cdef double x72 = nu**4
        cdef double x73 = -33.9782122170436*nu - 89.5298327361234*x31 + 188.0*x67 - 14.0*x72
        cdef double x74 = x15*x57
        cdef double x75 = 1.38977750996128*nu + 3.33842023648322*x31 + 3.42857142857143*x67 - 6.0*x72
        cdef double x76 = x61*x75
        cdef double x77 = log(r)
        cdef double x78 = nu*(452.542166996693 - 51.6952380952381*x77) + x31*(118.4*x77 - 1796.13660498019) + 602.318540416564*x67
        cdef double x79 = x11*x54
        cdef double x80 = r + 2.0
        cdef double x81 = x4*x80
        cdef double x82 = r*x4
        cdef double x83 = x10 + x80*x82
        cdef double x84 = x83**(-1)
        cdef double x85 = x28*x49*x84
        cdef double x86 = r**(-13)
        cdef double x87 = x7**4
        cdef double x88 = 756.0*nu
        cdef double x89 = 336.0*r + x88 + 407.0
        cdef double x90 = 2048.0*nu*x77*x89 + 28.0*nu*(1920.0*a6 + 733955.307463037) - 7.0*r*(938918.400156317*nu - 185763.092693281*x31 - 245760.0) - 5416406.59541186*x31 - 3440640.0
        cdef double x91 = x77**2
        cdef double x92 = x31*x91
        cdef double x93 = x67*(-163683964.822551*r - 17833256.898555*x0 - 1188987459.03162)
        cdef double x94 = x31*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 122635399361.987*r + 2064783811.32587*x0 - 3089250703.76879*x10 - 6178501407.53758*x14 + 1426660551.8844*x37 + 276057889687.011)
        cdef double x95 = 588.0*nu + 1079.0
        cdef double x96 = x88 + 1079.0
        cdef double x97 = x14*x96
        cdef double x98 = -38842241.4769507*nu + 240.0*r*(-7466.27061066206*nu - 3024.0*x31 + 17264.0) + 1920.0*x0*x95 + 480.0*x10*x96 - 1882456.23663972*x31 + 161280.0*x37 + 960.0*x97 + 13447680.0
        cdef double x99 = nu*x77
        cdef double x100 = x98*x99
        cdef double x101 = 8.0*r
        cdef double x102 = 4.0*x0 + x101 + 2.0*x14 + 16.0
        cdef double x103 = 7680.0*a6
        cdef double x104 = 128.0*r
        cdef double x105 = 7704.0*r
        cdef double x106 = 148.04406601634*r
        cdef double x107 = 113485.217444961*r
        cdef double x108 = nu*(x103*(x10 + x102) + x104*(13218.7851094412*r + 8529.39255472061*x0 - 6852.34813868015*x10 + 4264.6962773603*x14 - 33722.4297811176) + x106*(3852.0*x0 + 349.0*x10 + x105 + 1926.0*x14 + 36400.0) + x107*(-x10 + x102))
        cdef double x109 = (32768.0*x100 + 53760.0*x108 + 13212057600.0*x37 + 241555486248.807*x72 + 67645734912.0*x92 + 1120.0*x93 + 7.0*x94)**(-1)
        cdef double x110 = x10*x109*x90
        cdef double x111 = x110 + 0.000130208333333333*x46
        cdef double x112 = x111**(-4)
        cdef double x113 = r*x91
        cdef double x114 = -630116198.873299*nu - 197773496.793534*x31 + 5805304367.87913
        cdef double x115 = x0*x114
        cdef double x116 = -2675575.66847905*nu - 138240.0*x31 - 5278341.3229329
        cdef double x117 = x116*x14
        cdef double x118 = nu*(-2510664218.28128*nu - 42636451.6032331*x31 + 14515200.0*x67 + 1002013764.01019)
        cdef double x119 = 43393301259014.8*nu + 43133561885859.3*x31 + 5927865218923.02*x67 + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0
        cdef double x120 = r*x119
        cdef double x121 = 14700.0*nu + 42911.0
        cdef double x122 = 283115520.0*x121
        cdef double x123 = -1698693120.0*nu*(11592.0*nu + 69847.0) + 49152.0*r*(409207698.136075*nu + 102574080.0*x31 - 2119671837.36038) + x0*x122 + 879923036160.0*x14
        cdef double x124 = x123*x77
        cdef double x125 = (x113 - 1.59227685093395e-9*x115 - 1.67189069348064e-7*x117 + 9.55366110560367e-9*x118 + 1.72773095804465e-13*x120 + 1.72773095804465e-13*x124)**2
        cdef double x126 = nu*r
        cdef double x127 = nu*x0
        cdef double x128 = r*x31
        cdef double x129 = r*x121
        cdef double x130 = x0*x31
        cdef double x131 = 5041721180160.0*x31 - 104186110149937.0
        cdef double x132 = -25392914995744.3*nu - r*x122 - 879923036160.0*x0 - x131
        cdef double x133 = x132*x77
        cdef double x134 = -0.0438084424460039*nu - 0.143521050466841*r + 0.0185696317637669*x0 + 0.0385795738434214*x126 + 0.00941289164152486*x127 + 0.00662650629087394*x128 - 1.18824456940711e-6*x129 + 0.000486339502879429*x130 - 3.63558293513537e-15*x133 + 0.291062041428379*x31 + 0.0244692826489756*x67 + 0.0210425293255724*x91 + 1
        cdef double x135 = x134**(-2)
        cdef double x136 = 4.0*x31
        cdef double x137 = x17**2
        cdef double x138 = -0.46875*x137*(-5.0*nu + x136 + 1.0) + 0.0625*x17*x21*x3*(18.0*nu - 1.0) - 0.15625*x4*(-33.0*nu + 32.0*x31 - 5.0)
        cdef double x139 = x112*x125*x135*x138*x54*x87
        cdef double x140 = x7**2
        cdef double x141 = prst**2
        cdef double x142 = x111**(-2)
        cdef double x143 = 1822680546449.21*x31
        cdef double x144 = 5787938193408.0*x91
        cdef double x145 = -12049908701745.2*nu + r*x143 - 39476764256925.6*r + 5107745331375.71*x0 + 10611661054566.2*x126 + 2589101062873.81*x127 - 326837426.241486*x129 + 133772083200.0*x130 - x133 + x144 + 80059249540278.2*x31 + 6730497718123.02*x67 + 275059053208689.0
        cdef double x146 = x145**(-1)
        cdef double x147 = 0.0625*x137
        cdef double x148 = 0.125*x22
        cdef double x149 = -1171.0*nu - 861.0
        cdef double x150 = 0.015625*x4
        cdef double x151 = x137*(115.0*nu + x136 - 37.0)
        cdef double x152 = 0.03125*x22
        cdef double x153 = x18*(26.0*nu + 449.0)
        cdef double x154 = 5787938193408.0*x113 - 9216.0*x115 - 967680.0*x117 + 55296.0*x118 + x120 + x124
        cdef double x155 = x154**(-1)
        cdef double x156 = x145*x155*x37
        cdef double x157 = x109*x90
        cdef double x158 = x11*(x149*x150 + 0.015625*x151 + x152*x153) + x15*(x147*(12.0*nu - 3.0) + x148*(-21.0*X_1 + 21.0*X_2) + x4*(x12 + 2.8125)) + 7680.0*x156*x157 + x46
        cdef double x159 = x140*x141*x142*x146*x154*x158
        cdef double x160 = 1.27277314139085e-19*x139*x86 + 1.69542100694444e-8*x159*x38 + x25*x76 + x50 + 147.443752990146*x54*x56 + x54*x66 - 11.3175085791863*x57*x59 + x57*x71 + 1.48275342024365*x60*x62 + x63*x64 + x68*x69 + x73*x74 + x78*x79 - x81*x85 + 1.0
        cdef double x161 = x46*(2.0*x49 + 1.0) + 1.0
        cdef double x162 = x161**(-1)
        cdef double x163 = x137*(4.0*nu + 1.0)
        cdef double x164 = -x21*x22
        cdef double x165 = x137*(-27.0*nu + 28.0*x31 - 3.0)
        cdef double x166 = x152*(-39.0*X_1 + 39.0*X_2)
        cdef double x167 = x11*(0.125*x163 + 0.25*x164 + 1.125*x4) + 7680.0*x110 + x38*(-x150*(175.0*nu + 225.0) + 0.046875*x165 + x166*(2.0*nu - 3.0)) + x46
        cdef double x168 = x162*x167
        cdef double x169 = (x160*x168)**(-0.5)
        cdef double x170 = 4.0*x49 + 2.0
        cdef double x171 = r**(-6)
        cdef double x172 = x15*x5
        cdef double x173 = -6572428.80109422*nu + 1300341.64885296*x31 + 2048.0*x63*x89 + 688128.0*x99 + 1720320.0
        cdef double x174 = x31*x49
        cdef double x175 = 4.0*x14
        cdef double x176 = 6.0*x0 + x101 + 8.0
        cdef double x177 = 1.31621673590926e-19*x90*(53760.0*nu*(3740417.71815805*r + 2115968.85907902*x0 - 938918.400156317*x10 + x103*(x175 + x176) + x104*(17058.7851094412*r + 12794.0888320809*x0 - 27409.3925547206*x14 + 13218.7851094412) + x106*(5778.0*x0 + x105 + 1396.0*x14 + 7704.0) + x107*(-x175 + x176) + 1057984.42953951*x14 + 2888096.47013111) + 66060288000.0*x10 + 135291469824.0*x174*x77 + 7.0*x31*(-117964800.0*a6 + 4129567622.65173*r - 18535504222.6128*x0 + 7133302759.42198*x10 - 12357002815.0752*x14 + 122635399361.987) + 32768.0*x63*x98 + 7.0*x67*(-5706642207.53758*r - 26189434371.6082) + 32768.0*x99*(-1791904.9465589*nu + 3840.0*r*x95 + 2880.0*x0*x96 + 806400.0*x10 - 725760.0*x31 + 1920.0*x97 + 4143360.0))/(1.35654132757922e-7*x100 + 2.22557561555966e-7*x108 + 0.0546957463279941*x37 + x72 + 0.28004222119933*x92 + 4.63661586574928e-9*x93 + 2.8978849160933e-11*x94)**2
        cdef double x178 = -7680.0*x10*x109*x173 + x10*x177 - 30720.0*x109*x14*x90 + x172
        cdef double x179 = 11575876386816.0*x77
        cdef double x180 = 5807150888816.34*nu + 10215490662751.4*r + 5178202125747.62*x126 + 267544166400.0*x128 - x132*x49 + x143 + x179*x49 + x77*(4161798144000.0*nu + 1759846072320.0*r + 12148770078720.0) - 53501685054374.1
        cdef double x181 = x125*x138*x54*x86*x87
        cdef double x182 = -18432.0*r*x114 - 2903040.0*x0*x116 + x119 + x123*x49 + x144 + x179 + x77*(20113376778784.3*nu + 2639769108480.0*x0 + 566231040.0*x129 + x131)
        cdef double x183 = x141*x158
        cdef double x184 = -x178
        cdef double x185 = x154*x38
        cdef double x186 = prst**3
        cdef double x187 = prst**5
        cdef double x188 = prst**7
        cdef double x189 = 4.0*x186
        cdef double x190 = 6.0*x187
        cdef double x191 = x167*x169/(x170*x46 + 2.0)
        cdef double x192 = 2.0*pphi
        cdef double x193 = x192*x25
        cdef double x194 = x15*x192
        cdef double x195 = 4.0*pphi**3*x11
        cdef double y0 = 2.0*r
        cdef double y1 = r**2
        cdef double y2 = chi_1*X_1
        cdef double y3 = chi_2*X_2
        cdef double y4 = y2 + y3
        cdef double y5 = y4**2
        cdef double y6 = y1 + y5
        cdef double y7 = r*(-y0 + y6)
        cdef double y8 = 2.0*y5
        cdef double y9 = 2.0*y1 + y8
        cdef double y10 = (y7 + y9)**(-1)
        cdef double y11 = r**5
        cdef double y12 = y11**(-1)
        cdef double y13 = 12.0*nu
        cdef double y14 = pphi*y4
        cdef double y15 = dSO*y14
        cdef double y16 = r**4
        cdef double y17 = y16**(-1)
        cdef double y18 = 0.25*y5
        cdef double y19 = y2 - y3
        cdef double y20 = X_1 - X_2
        cdef double y21 = y19*y20
        cdef double y22 = pphi*y21
        cdef double y23 = 5.0*X_1 - 5.0*X_2
        cdef double y24 = y19*y4
        cdef double y25 = -y18 + 0.0416666666666667*y23*y24
        cdef double y26 = 6.0*y17
        cdef double y27 = r**3
        cdef double y28 = y27**(-1)
        cdef double y29 = -1.40625*nu - 0.46875
        cdef double y30 = pphi**2
        cdef double y31 = y26*y30
        cdef double y32 = nu**2
        cdef double y33 = -2.0859375*nu - 2.07161458333333*y32 + 0.23046875
        cdef double y34 = 12.0*y12
        cdef double y35 = y30*y34
        cdef double y36 = r**(-6)
        cdef double y37 = 20.0*y36
        cdef double y38 = pphi**4
        cdef double y39 = 0.5859375*nu + 1.34765625*y32 + 0.41015625
        cdef double y40 = y38*y39
        cdef double y41 = 0.46875 - 0.28125*nu
        cdef double y42 = -0.2734375*nu - 0.798177083333333*y32 - 0.23046875
        cdef double y43 = -0.3515625*nu + 0.29296875*y32 - 0.41015625
        cdef double y44 = y38*y43
        cdef double y45 = y0 - 2.0
        cdef double y46 = 3.0*nu
        cdef double y47 = y17*y46
        cdef double y48 = y28*y5
        cdef double y49 = 0.0833333333333333*y48
        cdef double y50 = y25*y28
        cdef double y51 = y1**(-1)
        cdef double y52 = 0.71875*nu - 0.09375
        cdef double y53 = y28*y30
        cdef double y54 = 2.0*y53
        cdef double y55 = 3.0*y17
        cdef double y56 = y30*y55
        cdef double y57 = 4.0*y12
        cdef double y58 = -y28*(-11.0625*nu + 1.13541666666667*y32 - 0.15625) - y29*y54 - y33*y56 - y40*y57 - y51*y52
        cdef double y59 = 0.34375*nu + 0.09375
        cdef double y60 = -y28*(-0.0625*nu + 1.07291666666667*y32 + 0.15625) - y41*y54 - y42*y56 - y44*y57 - y51*y59
        cdef double y61 = y6 + 0.5*y7
        cdef double y62 = y61**(-2)
        cdef double y63 = 0.25*y62
        cdef double y64 = r*y45
        cdef double y65 = 0.5*y5
        cdef double y66 = nu*y28
        cdef double y67 = y5*y51
        cdef double y68 = 0.0416666666666667*y67
        cdef double y69 = pphi*y51
        cdef double y70 = y25*y4
        cdef double y71 = r**(-1)
        cdef double y72 = y30*y51
        cdef double y73 = y17*y38
        cdef double y74 = y4*(y29*y72 + y33*y53 + y39*y73 + y51*(-5.53125*nu + 0.567708333333333*y32 - 0.078125) + y52*y71 + 1.75)
        cdef double y75 = y21*(y41*y72 + y42*y53 + y43*y73 + y51*(-0.03125*nu + 0.536458333333333*y32 + 0.078125) + y59*y71 + 0.25)
        cdef double y76 = prst**4
        cdef double y77 = r**(-4.5)
        cdef double y78 = nu*y77
        cdef double y79 = r**(-3.5)
        cdef double y80 = prst**6
        cdef double y81 = nu*y80
        cdef double y82 = r**(-2.5)
        cdef double y83 = prst**8
        cdef double y84 = nu*y83
        cdef double y85 = nu*y71
        cdef double y86 = 0.121954868780449*y83
        cdef double y87 = 8.0*nu - 6.0*y32
        cdef double y88 = y76*y87
        cdef double y89 = nu**3
        cdef double y90 = 92.7110442849544*nu - 131.0*y32 + 10.0*y89
        cdef double y91 = y76*y90
        cdef double y92 = -2.78300763695006*nu - 5.4*y32 + 6.0*y89
        cdef double y93 = y80*y92
        cdef double y94 = nu**4
        cdef double y95 = -33.9782122170436*nu - 89.5298327361234*y32 + 188.0*y89 - 14.0*y94
        cdef double y96 = y80*y95
        cdef double y97 = 1.38977750996128*nu + 3.33842023648322*y32 + 3.42857142857143*y89 - 6.0*y94
        cdef double y98 = y83*y97
        cdef double y99 = log(r)
        cdef double y100 = nu*(452.542166996693 - 51.6952380952381*y99) + y32*(118.4*y99 - 1796.13660498019) + 602.318540416564*y89
        cdef double y101 = y17*y76
        cdef double y102 = r + 2.0
        cdef double y103 = r*y5
        cdef double y104 = y102*y103 + y16
        cdef double y105 = y104**(-1)
        cdef double y106 = y105*y30
        cdef double y107 = y102*y5
        cdef double y108 = y107*y71
        cdef double y109 = y99**2
        cdef double y110 = y109*y32
        cdef double y111 = y89*(-163683964.822551*r - 17833256.898555*y1 - 1188987459.03162)
        cdef double y112 = y32*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 122635399361.987*r + 2064783811.32587*y1 + 1426660551.8844*y11 - 3089250703.76879*y16 - 6178501407.53758*y27 + 276057889687.011)
        cdef double y113 = 588.0*nu + 1079.0
        cdef double y114 = 756.0*nu
        cdef double y115 = y114 + 1079.0
        cdef double y116 = y115*y27
        cdef double y117 = -38842241.4769507*nu + 240.0*r*(-7466.27061066206*nu - 3024.0*y32 + 17264.0) + 1920.0*y1*y113 + 161280.0*y11 + 480.0*y115*y16 + 960.0*y116 - 1882456.23663972*y32 + 13447680.0
        cdef double y118 = nu*y99
        cdef double y119 = y117*y118
        cdef double y120 = 8.0*r
        cdef double y121 = 4.0*y1 + y120 + 2.0*y27 + 16.0
        cdef double y122 = 7680.0*a6
        cdef double y123 = 128.0*r
        cdef double y124 = 7704.0*r
        cdef double y125 = 148.04406601634*r
        cdef double y126 = 113485.217444961*r
        cdef double y127 = nu*(y122*(y121 + y16) + y123*(13218.7851094412*r + 8529.39255472061*y1 - 6852.34813868015*y16 + 4264.6962773603*y27 - 33722.4297811176) + y125*(3852.0*y1 + y124 + 349.0*y16 + 1926.0*y27 + 36400.0) + y126*(y121 - y16))
        cdef double y128 = (13212057600.0*y11 + 67645734912.0*y110 + 1120.0*y111 + 7.0*y112 + 32768.0*y119 + 53760.0*y127 + 241555486248.807*y94)**(-1)
        cdef double y129 = 336.0*r + y114 + 407.0
        cdef double y130 = 2048.0*nu*y129*y99 + 28.0*nu*(1920.0*a6 + 733955.307463037) - 7.0*r*(938918.400156317*nu - 185763.092693281*y32 - 245760.0) - 5416406.59541186*y32 - 3440640.0
        cdef double y131 = y128*y130
        cdef double y132 = y131*y16
        cdef double y133 = y132 + 0.000130208333333333*y67
        cdef double y134 = y133**(-4)
        cdef double y135 = r*y109
        cdef double y136 = -630116198.873299*nu - 197773496.793534*y32 + 5805304367.87913
        cdef double y137 = y1*y136
        cdef double y138 = -2675575.66847905*nu - 138240.0*y32 - 5278341.3229329
        cdef double y139 = y138*y27
        cdef double y140 = nu*(-2510664218.28128*nu - 42636451.6032331*y32 + 14515200.0*y89 + 1002013764.01019)
        cdef double y141 = (1 - 0.496948781616935*nu)**2
        cdef double y142 = 43393301259014.8*nu + 86618264430493.3*y141 + 43133561885859.3*y32 + 5927865218923.02*y89 + 188440788778196.0
        cdef double y143 = r*y142
        cdef double y144 = 14700.0*nu + 42911.0
        cdef double y145 = 283115520.0*y144
        cdef double y146 = -1698693120.0*nu*(11592.0*nu + 69847.0) + 49152.0*r*(409207698.136075*nu + 102574080.0*y32 - 2119671837.36038) + y1*y145 + 879923036160.0*y27
        cdef double y147 = y146*y99
        cdef double y148 = y135 - 1.59227685093395e-9*y137 - 1.67189069348064e-7*y139 + 9.55366110560367e-9*y140 + 1.72773095804465e-13*y143 + 1.72773095804465e-13*y147
        cdef double y149 = y148**2
        cdef double y150 = nu*r
        cdef double y151 = nu*y1
        cdef double y152 = r*y32
        cdef double y153 = r*y144
        cdef double y154 = y1*y32
        cdef double y155 = 5041721180160.0*y32 - 104186110149937.0
        cdef double y156 = -25392914995744.3*nu - r*y145 - 879923036160.0*y1 - y155
        cdef double y157 = y156*y99
        cdef double y158 = -0.0438084424460039*nu - 0.143521050466841*r + 0.0185696317637669*y1 + 0.0210425293255724*y109 + 0.0385795738434214*y150 + 0.00941289164152486*y151 + 0.00662650629087394*y152 - 1.18824456940711e-6*y153 + 0.000486339502879429*y154 - 3.63558293513537e-15*y157 + 0.291062041428379*y32 + 0.0244692826489756*y89 + 1
        cdef double y159 = y158**(-2)
        cdef double y160 = 4.0*y32
        cdef double y161 = y19**2
        cdef double y162 = -0.46875*y161*(-5.0*nu + y160 + 1.0) + 0.0625*y19*y23*y4*(18.0*nu - 1.0) - 0.15625*y5*(-33.0*nu + 32.0*y32 - 5.0)
        cdef double y163 = y134*y149*y159*y162
        cdef double y164 = y163*y76
        cdef double y165 = r**(-13)
        cdef double y166 = y6**4
        cdef double y167 = y165*y166
        cdef double y168 = prst**2
        cdef double y169 = y133**(-2)
        cdef double y170 = 1822680546449.21*y32
        cdef double y171 = 5787938193408.0*y109
        cdef double y172 = -12049908701745.2*nu + r*y170 - 39476764256925.6*r + 5107745331375.71*y1 + 10611661054566.2*y150 + 2589101062873.81*y151 - 326837426.241486*y153 + 133772083200.0*y154 - y157 + y171 + 80059249540278.2*y32 + 6730497718123.02*y89 + 275059053208689.0
        cdef double y173 = y172**(-1)
        cdef double y174 = y168*y169*y173
        cdef double y175 = 0.0625*y161
        cdef double y176 = 0.125*y24
        cdef double y177 = -1171.0*nu - 861.0
        cdef double y178 = 0.015625*y5
        cdef double y179 = y161*(115.0*nu + y160 - 37.0)
        cdef double y180 = 0.03125*y24
        cdef double y181 = y20*(26.0*nu + 449.0)
        cdef double y182 = 5787938193408.0*y135 - 9216.0*y137 - 967680.0*y139 + 55296.0*y140 + y143 + y147
        cdef double y183 = y182**(-1)
        cdef double y184 = y11*y183
        cdef double y185 = y172*y184
        cdef double y186 = 7680.0*y131
        cdef double y187 = y17*(y177*y178 + 0.015625*y179 + y180*y181) + y185*y186 + y28*(y175*(y13 - 3.0) + y176*(-21.0*X_1 + 21.0*X_2) + y5*(y46 + 2.8125)) + y67
        cdef double y188 = y182*y187
        cdef double y189 = y174*y188
        cdef double y190 = y6**2
        cdef double y191 = y12*y190
        cdef double y192 = 1.69542100694444e-8*y191
        cdef double y193 = y100*y101 - y106*y108 + 1.27277314139085e-19*y164*y167 + y189*y192 + y28*y91 + y28*y96 + y51*y88 + y51*y93 + y51*y98 + y72 + 147.443752990146*y76*y78 - 11.3175085791863*y79*y81 + 1.48275342024365*y82*y84 + y85*y86 + 1.0
        cdef double y194 = y67*(2.0*y71 + 1.0) + 1.0
        cdef double y195 = y194**(-1)
        cdef double y196 = y161*(4.0*nu + 1.0)
        cdef double y197 = -y23*y24
        cdef double y198 = y17*(0.125*y196 + 0.25*y197 + 1.125*y5)
        cdef double y199 = y161*(-27.0*nu + 28.0*y32 - 3.0)
        cdef double y200 = -39.0*X_1 + 39.0*X_2
        cdef double y201 = y180*y200
        cdef double y202 = y12*(-y178*(175.0*nu + 225.0) + 0.046875*y199 + y201*(2.0*nu - 3.0))
        cdef double y203 = 7680.0*y132 + y198 + y202 + y67
        cdef double y204 = y195*y203
        cdef double y205 = y193*y204
        cdef double y206 = y205**(-1.5)
        cdef double y207 = 4.0*y71 + 2.0
        cdef double y208 = -y207
        cdef double y209 = -y17*y8 + y208*y48
        cdef double y210 = y194**(-2)
        cdef double y211 = y203*y210
        cdef double y212 = y193*y211
        cdef double y213 = 875.0*nu + 1125.0
        cdef double y214 = 10.0*nu - 15.0
        cdef double y215 = y28*y8
        cdef double y216 = 2048.0*y129
        cdef double y217 = -6572428.80109422*nu + 688128.0*y118 + y216*y85 + 1300341.64885296*y32 + 1720320.0
        cdef double y218 = y130*y16
        cdef double y219 = 0.0546957463279941*y11 + 0.28004222119933*y110 + 4.63661586574928e-9*y111 + 2.8978849160933e-11*y112 + 1.35654132757922e-7*y119 + 2.22557561555966e-7*y127 + y94
        cdef double y220 = y219**(-2)
        cdef double y221 = y89*(-5706642207.53758*r - 26189434371.6082)
        cdef double y222 = y71*y99
        cdef double y223 = y222*y32
        cdef double y224 = y32*(-117964800.0*a6 + 4129567622.65173*r - 18535504222.6128*y1 + 7133302759.42198*y16 - 12357002815.0752*y27 + 122635399361.987)
        cdef double y225 = y1*y115
        cdef double y226 = -1791904.9465589*nu + 3840.0*r*y113 + 1920.0*y116 + 806400.0*y16 + 2880.0*y225 - 725760.0*y32 + 4143360.0
        cdef double y227 = y118*y226
        cdef double y228 = y117*y85
        cdef double y229 = 4.0*y27
        cdef double y230 = 6.0*y1 + y120 + 8.0
        cdef double y231 = nu*(3740417.71815805*r + 2115968.85907902*y1 + y122*(y229 + y230) + y123*(17058.7851094412*r + 12794.0888320809*y1 - 27409.3925547206*y27 + 13218.7851094412) + y125*(5778.0*y1 + y124 + 1396.0*y27 + 7704.0) + y126*(-y229 + y230) - 938918.400156317*y16 + 1057984.42953951*y27 + 2888096.47013111)
        cdef double y232 = y220*(66060288000.0*y16 + 7.0*y221 + 135291469824.0*y223 + 7.0*y224 + 32768.0*y227 + 32768.0*y228 + 53760.0*y231)
        cdef double y233 = y218*y232
        cdef double y234 = -30720.0*y128*y130*y27 - 7680.0*y128*y16*y217 + y215 + 1.31621673590926e-19*y233
        cdef double y235 = -y12*(0.5*y196 + y197 + 4.5*y5) - y234 - y36*(-y178*y213 + 0.234375*y199 + y201*y214)
        cdef double y236 = y193*y195
        cdef double y237 = r**(-5.5)
        cdef double y238 = nu*y76
        cdef double y239 = nu*y51
        cdef double y240 = 2.0*y28
        cdef double y241 = 118.4*y32
        cdef double y242 = -51.6952380952381*nu*y71 + y241*y71
        cdef double y243 = y100*y76
        cdef double y244 = y5*y71
        cdef double y245 = y104**(-2)
        cdef double y246 = y103 + y107 + y229
        cdef double y247 = r**(-14)
        cdef double y248 = y164*y166
        cdef double y249 = r**(-12)
        cdef double y250 = y6**3
        cdef double y251 = 5178202125747.62*nu
        cdef double y252 = 267544166400.0*y32
        cdef double y253 = 11575876386816.0*y71
        cdef double y254 = y253*y99
        cdef double y255 = y99*(4161798144000.0*nu + 1759846072320.0*r + 12148770078720.0)
        cdef double y256 = y156*y71
        cdef double y257 = 5807150888816.34*nu + r*y251 + r*y252 + 10215490662751.4*r + y170 + y254 + y255 - y256 - 53501685054374.1
        cdef double y258 = y158**(-3)
        cdef double y259 = y257*y258
        cdef double y260 = y134*y162*y76
        cdef double y261 = y149*y260
        cdef double y262 = y167*y261
        cdef double y263 = 9.25454462627843e-34*y262
        cdef double y264 = y146*y71
        cdef double y265 = y99*(20113376778784.3*nu + 2639769108480.0*y1 + 566231040.0*y153 + y155)
        cdef double y266 = r*y136
        cdef double y267 = y1*y138
        cdef double y268 = y142 + y171 + y264 + y265 - 18432.0*y266 - 2903040.0*y267 + 11575876386816.0*y99
        cdef double y269 = y189*y190
        cdef double y270 = y169*y188
        cdef double y271 = y168*y270
        cdef double y272 = y159*y271
        cdef double y273 = y257*y272
        cdef double y274 = 2.24091649004576e-37*y191
        cdef double y275 = y133**(-5)
        cdef double y276 = -y234
        cdef double y277 = y275*y276
        cdef double y278 = y159*y167
        cdef double y279 = y162*y278
        cdef double y280 = y149*y76
        cdef double y281 = y279*y280
        cdef double y282 = 6.62902677807736e-23*y281
        cdef double y283 = y168*y173*y188
        cdef double y284 = y133**(-3)
        cdef double y285 = y276*y284
        cdef double y286 = y283*y285
        cdef double y287 = 36.0*nu
        cdef double y288 = y287 - 9.0
        cdef double y289 = -63.0*X_1 + 63.0*X_2
        cdef double y290 = 0.0625*y5
        cdef double y291 = y149**(-1)
        cdef double y292 = y268*y291
        cdef double y293 = y11*y172
        cdef double y294 = y131*y293
        cdef double y295 = 2.29252167428035e-22*y294
        cdef double y296 = y130*y232
        cdef double y297 = 7680.0*y11*y128*y130*y183*y257 + 7680.0*y11*y128*y172*y183*y217 - y12*(y176*y181 + y177*y290 + 0.0625*y179) + 38400.0*y128*y130*y16*y172*y183 - y17*(y175*y288 + y176*y289 + y5*(9.0*nu + 8.4375)) - 1.31621673590926e-19*y185*y296 - y215 - y292*y295
        cdef double y298 = 39.6112800271521*nu*y77*y80 - 3.0*y101*y90 + y102*y105*y30*y5*y51 + y102*y245*y246*y30*y5*y71 - y106*y244 + 1.69542100694444e-8*y12*y168*y169*y173*y182*y190*y297 + 1.69542100694444e-8*y12*y168*y169*y173*y187*y190*y268 + 1.01821851311268e-18*y134*y149*y159*y162*y249*y250*y76 + 7.59859378406358e-45*y134*y159*y162*y165*y166*y182*y268*y76 + 6.78168402777778e-8*y168*y169*y17*y173*y182*y187*y6 + y17*y242*y76 - 4.41515887225116e-12*y191*y286 - 663.496888455656*y237*y238 - y239*y86 - y240*y88 - y240*y93 - y240*y98 - y243*y57 - 1.65460508380811e-18*y247*y248 - y259*y263 - 8.47710503472222e-8*y269*y36 - y273*y274 - y277*y282 - y54 - y55*y96 - 3.70688355060912*y79*y84
        cdef double y299 = y204*y298 - y209*y212 + y235*y236
        cdef double y300 = y17*y5
        cdef double y301 = 8.0*y71 + 4.0
        cdef double y302 = -4.0*y300 - y301*y48
        cdef double y303 = r**(-7)
        cdef double y304 = 0.03125*y5
        cdef double y305 = 0.0625*y200*y24
        cdef double y306 = 6.0*y300
        cdef double y307 = y128*y16
        cdef double y308 = -7680.0*y216*y239 + 10569646080.0*y85
        cdef double y309 = y128*y217
        cdef double y310 = 2.63243347181853e-19*y232
        cdef double y311 = y217*y310
        cdef double y312 = 1.99471718230171e-8*(0.48828125*y16 + 5.17401430341932e-11*y221 + y223 + 5.17401430341932e-11*y224 + 2.42203000992063e-7*y227 + 2.42203000992063e-7*y228 + 3.97364298502604e-7*y231)**2/y219**3
        cdef double y313 = 135291469824.0*y32*y51
        cdef double y314 = 12.0*y1
        cdef double y315 = 12.0*r + 8.0
        cdef double y316 = 1.31621673590926e-19*y220*(53760.0*nu*(8463875.43631609*r + 6347906.57723707*y1 + y122*(y314 + y315) + y123*(25588.1776641618*r - 82228.1776641618*y1 + 17058.7851094412) + y125*(11556.0*r + 4188.0*y1 + 7704.0) + y126*(-y314 + y315) - 7511347.20125054*y27 + 7480835.43631609) - 32768.0*y117*y239 + 32768.0*y118*(2257920.0*nu + 5760.0*r*y115 + 5760.0*y225 + 3225600.0*y27 + 4143360.0) + 65536.0*y226*y85 + 264241152000.0*y27 - y313*y99 + y313 + 7.0*y32*(-37071008445.2255*r - 37071008445.2255*y1 + 28533211037.6879*y27 + 4129567622.65173) - 39946495452.7631*y89)
        cdef double y317 = 92160.0*y1*y131 - y16*y311 + y218*y312 - y218*y316 - 1.05297338872741e-18*y27*y296 + 61440.0*y27*y309 + y306 + y307*y308
        cdef double y318 = y105*y8
        cdef double y319 = y102*y318
        cdef double y320 = y245*y30
        cdef double y321 = y102*y245*y246*y8
        cdef double y322 = (0.230275523363951*nu + 0.0307148905018684*y109 + 0.459657725867658*y141 + 5.30670671930294e-15*y264 + 5.30670671930294e-15*y265 - 9.78132182501918e-11*y266 - 1.54055818744052e-8*y267 + 0.228897162687159*y32 + 0.031457442188381*y89 + 0.0614297810037367*y99 + 1)**2
        cdef double y323 = y260*y278
        cdef double y324 = 11575876386816.0*y51
        cdef double y325 = 8323596288000.0*nu + 24297540157440.0
        cdef double y326 = y156*y51 + y251 + y252 - y324*y99 + y324 + y71*(3519692144640.0*r + y325) + 1759846072320.0*y99 + 10215490662751.4
        cdef double y327 = y166*y247
        cdef double y328 = y259*y261
        cdef double y329 = y249*y250
        cdef double y330 = (0.108541457767442*nu + 0.190937736865098*r + 0.0967857763822762*y150 + 0.00500066803742898*y152 + 0.216364706551791*y222 + 1.86910000868887e-14*y255 - 1.86910000868887e-14*y256 + 0.0340677222520525*y32 - 1)**2
        cdef double y331 = 11614301777632.7*nu - 5806080.0*r*y138 - y146*y51 + y253 + y254 + 3645361092898.41*y32 + y71*(40226753557568.7*nu + 5279538216960.0*y1 + 1132462080.0*y153 + 10083442360320.0*y32 - 208372220299875.0) + y99*(5279538216960.0*r + y325) - 107003370108748.0
        cdef double y332 = y159*y327
        cdef double y333 = y182*y268
        cdef double y334 = y260*y333
        cdef double y335 = y159*y329
        cdef double y336 = y167*y259
        cdef double y337 = y174*y192
        cdef double y338 = y174*y268
        cdef double y339 = y187*y338
        cdef double y340 = y17*y6
        cdef double y341 = 1.35633680555556e-7*y340
        cdef double y342 = y190*y36
        cdef double y343 = 1.69542100694444e-7*y342
        cdef double y344 = y168*y191
        cdef double y345 = y187*y268*y344
        cdef double y346 = 4.48183298009152e-37*y159*y257
        cdef double y347 = y169*y346
        cdef double y348 = y162*y277*y280
        cdef double y349 = (y128*y130*y27 + 0.25*y128*y16*y217 - 4.28455968720463e-24*y233 - 6.51041666666667e-5*y48)**2
        cdef double y350 = y191*y283
        cdef double y351 = 8.83031774450231e-12*y285
        cdef double y352 = y173*y351
        cdef double y353 = y182*y297
        cdef double y354 = y174*y353
        cdef double y355 = 3.39084201388889e-8*y191
        cdef double y356 = y344*y353
        cdef double y357 = y184*y257
        cdef double y358 = 76800.0*y183
        cdef double y359 = y172*y183
        cdef double y360 = 4.5850433485607e-22*y292
        cdef double y361 = y130*y185
        cdef double y362 = y205**(-0.5)
        cdef double y363 = (y207*y67 + 2.0)**(-1)
        cdef double y364 = 12.0*y168
        cdef double y365 = y51*y87
        cdef double y366 = y28*y90
        cdef double y367 = 30.0*y76
        cdef double y368 = y51*y92
        cdef double y369 = y28*y95
        cdef double y370 = y51*y97
        cdef double y371 = y100*y17
        cdef double y372 = y163*y167
        cdef double y373 = y173*y270
        cdef double y374 = y355*y373
        cdef double y375 = prst**3
        cdef double y376 = y375*y78
        cdef double y377 = prst**5
        cdef double y378 = nu*y377*y79
        cdef double y379 = prst**7
        cdef double y380 = nu*y379
        cdef double y381 = y380*y82
        cdef double y382 = y379*y85
        cdef double y383 = 0.00678224732971111*y375
        cdef double y384 = 0.0101733709945667*y377
        cdef double y385 = y370*y379
        cdef double y386 = y372*y375
        cdef double y387 = prst*y373
        cdef double y388 = y206*y210*(y133 + 0.000130208333333333*y198 + 0.000130208333333333*y202)**2
        cdef double y389 = y17*y39
        cdef double y390 = 12.0*y30
        cdef double y391 = 2.0*y2 + 2.0*y3
        cdef double y392 = 2.0*y51
        cdef double y393 = pphi*y392
        cdef double y394 = pphi*y28
        cdef double y395 = 2.0*y394
        cdef double y396 = pphi**3
        cdef double y397 = 4.0*y396
        cdef double y398 = y29*y393 + y33*y395 + y389*y397
        cdef double y399 = y17*y43
        cdef double y400 = y19*(y393*y41 + y395*y42 + y397*y399)
        cdef double y401 = 4.0*y375
        cdef double y402 = 6.0*y377
        cdef double y403 = prst*y374 + y365*y401 + y366*y401 + y368*y402 + y369*y402 + y371*y401 + 589.775011960583*y376 - 67.9050514751178*y378 + 11.8620273619492*y381 + 0.975638950243592*y382 + 8.0*y385 + 5.09109256556341e-19*y386
        cdef double y404 = y209*y211
        cdef double y405 = y195*y235
        cdef double y406 = 16.0*y12
        cdef double y407 = y203*y206*y299/(y301*y67 + 4.0)
        cdef double y408 = dSO*y4
        cdef double y409 = 4.0*y394
        cdef double y410 = pphi*y26
        cdef double y411 = y396*y406
        cdef double y412 = pphi*y71
        cdef double y413 = y318*y412
        cdef double y414 = -2.0*pphi*y51 + y102*y413
        cdef double y415 = -y414

        # Evaluate Hamiltonian
        cdef double H,xi
        H,xi = self._call(q,p,chi_1,chi_2,m_1,m_2)

        # Evaluate Heff Jacobian
        cdef double dHeffdr = 0.5*x169*(x160*x162*(-x171*(-x150*(875.0*nu + 1125.0) + 0.234375*x165 + x166*(10.0*nu - 15.0)) - x178 - x38*(0.5*x163 + x164 + 4.5*x4)) - x160*x167*(-x11*x5 - x16*x170)/x161**2 + x168*(-663.496888455656*nu*r**(-5.5)*x54 - nu*x25*x64 + 39.6112800271521*nu*x55*x57 + 6.78168402777778e-8*x11*x141*x142*x146*x154*x158*x7 + x11*x54*(118.4*x174 - 51.6952380952381*x63) + 7.59859378406358e-45*x112*x135*x138*x154*x182*x54*x86*x87 - 9.25454462627843e-34*x112*x180*x181/x134**3 - 2.24091649004576e-37*x135*x140*x142*x154*x180*x183*x38 + 1.69542100694444e-8*x140*x141*x142*x146*x154*x38*(38400.0*x10*x109*x145*x155*x90 + 7680.0*x109*x145*x155*x173*x37 + 7680.0*x109*x155*x180*x37*x90 - x11*(x147*(36.0*nu - 9.0) + x148*(-63.0*X_1 + 63.0*X_2) + x4*(9.0*nu + 8.4375)) - x156*x177 - x172 - x38*(x148*x153 + 0.0625*x149*x4 + 0.0625*x151) - 2.29252167428035e-22*x145*x157*x182*x37/x125) + 1.69542100694444e-8*x140*x141*x142*x146*x158*x182*x38 - 8.47710503472222e-8*x159*x171 - x24*x76 + x25*x28*x4*x80*x84 + x28*x4*x49*x80*(x175 + x81 + x82)/x83**2 - x30 - x33*x57*x73 - x39*x54*x78 - x4*x85 - 3.70688355060912*x58*x62 - 2.0*x65*x69 - 3.0*x68*x79 - 2.0*x70*x74 - 4.41515887225116e-12*x140*x146*x183*x184*x185/x111**3 - 6.62902677807736e-23*x135*x181*x184/x111**5 + 1.01821851311268e-18*x112*x125*x135*x138*x54*x7**3/r**12 - 1.65460508380811e-18*x139/r**14)) + x9*(-dSO*x11*x12*x13 + pphi*x17*x18*(-x15*(-0.0625*nu + 1.07291666666667*x31 + 0.15625) - x25*x41 - x30*x42 - x34*x43 - x40*x44) + pphi*x3*(-x15*(-11.0625*nu + 1.13541666666667*x31 - 0.15625) - x25*x26 - x27*x30 - x32*x34 - x35*x40) - x13*x23*x24 - 0.0833333333333333*x16*x20) - 0.25*(r*(x6 - 2.0) + x6 + x7)*(pphi*x52 + pphi*x53 + x13*x45 + x13*x48 + x20*x47)/(x7 + 0.5*x8)**2

        cdef double dHeffdphi = 0

        cdef double dHeffdpr = x191*(11.8620273619492*nu*x188*x60 + 3.39084201388889e-8*prst*x140*x142*x146*x158*x185 + x11*x189*x78 + 5.09109256556341e-19*x112*x125*x135*x138*x186*x86*x87 + x15*x189*x68 + x15*x190*x73 + 589.775011960583*x186*x56 - 67.9050514751178*x187*x59 + 8.0*x188*x25*x75 + 0.975638950243592*x188*x63 + x189*x66 + x190*x71)

        cdef double dHeffdpphi = x191*(2.0*pphi*x25 - pphi*x49*x5*x80*x84) + x9*(x13*(x193*x27 + x194*x32 + x195*x35) + x19*x47 + x20*(x193*x42 + x194*x43 + x195*x44) + x3*x45 + x3*x48 + x52 + x53)

        # Evaluate Heff Hessian
        cdef double d2Heffdr2 = y10*(y12*y13*y15 + y14*y25*y26 + y14*(y17*(-33.1875*nu + 3.40625*y32 - 0.46875) + y28*(1.4375*nu - 0.1875) + y29*y31 + y33*y35 + y37*y40) + y17*y18*y22 + y22*(y17*(-0.1875*nu + 3.21875*y32 + 0.46875) + y28*(0.6875*nu + 0.1875) + y31*y41 + y35*y42 + y37*y44)) - 0.25*y206*y299**2 + 0.5*y362*(-y193*y210*y235*y302 + 8.0*y193*y203*(y208*y28*y65 - y300)**2/y194**3 + y195*y298*(-y12*(y196 + 2.0*y197 + 9.0*y5) + 61440.0*y128*y130*y27 + 15360.0*y128*y16*y217 - 2.63243347181853e-19*y233 - y36*(0.46875*y199 - y213*y304 + y214*y305) - 4.0*y48) + y204*(3649.23288650611*r**(-6.5)*y238 + 6.0*y101*y87 + y101*(51.6952380952381*nu*y51 - y241*y51) + y108*y320*(y314 + y8) - 6.103515625e-7*y12*y189*y6 - 8.0*y12*y242*y76 + 1.62760416666667e-6*y134*y349*y350 + 1.16714400523217e-40*y159*y188*y257*y285*y344 - 2.5455462827817e-17*y164*y165*y250 + 7.59859378406358e-45*y182*y323*y331 + y182*y337*(-y11*y131*y257*y360 + y12*(0.25*y161*y288 + 0.5*y24*y289 + y5*(y287 + 33.75)) + y128*y185*y308 - y130*y310*y357 + 153600.0*y131*y27*y359 - 2.29252167428035e-21*y132*y172*y292 + y132*y257*y358 + y172*y217*y307*y358 + y184*y186*y326 - y185*y311 - 1.31621673590926e-18*y233*y359 - y291*y295*y331 + 7.85795675813156e-45*y292*y293*y296 - y293*y309*y360 + y306 + 15360.0*y309*y357 + y312*y361 - y316*y361 + y36*(y176*y20*(130.0*nu + 2245.0) + 0.3125*y179 + y290*(-5855.0*nu - 4305.0)) + 2.81299777100766e-6*y294*y322/y148**3) + y187*y331*y337 + 1.35633680555556e-7*y189*y28 + 4.66406554828496e-24*y191*y258*y271*y330 - 178.250760122184*y237*y81 + y243*y37 + y246*y320*y71*y8 - y258*y263*y326 + y26*y93 + y26*y98 + 5.08626302083333e-7*y269*y303 - y272*y274*y326 - 1.79273319203661e-36*y273*y340 + 2.24091649004576e-36*y273*y342 - y275*y282*y317 - 7.91520185839956e-48*y277*y279*y333*y76 + 0.243909737560898*y28*y84 - 4.41515887225116e-12*y284*y317*y350 - 3.53212709780093e-11*y286*y340 + 4.41515887225116e-11*y286*y342 + y297*y338*y355 + y31 + y318*y72 - y319*y53 - y321*y72 + 2.69825540021952e-16*y322*y323 + 2.40618160283239e-32*y327*y328 - 1.48072714020455e-32*y328*y329 - 1.97563438385653e-43*y332*y334 + 1.72354696230011e-21*y332*y348 + 1.21577500545017e-43*y334*y335 - 1.10501271569469e-58*y334*y336 - 1.06064428449238e-21*y335*y348 + 9.64015065237337e-37*y336*y348 + y339*y341 - y339*y343 + y34*y91 + y34*y96 + y341*y354 - y343*y354 - y345*y347 - y345*y352 - y347*y356 - y352*y356 + 12.9740924271319*y78*y83 + 2.88925109089694e-20*y262*y330/y158**4 + 4.07287405245073e-17*y281*y349/y133**6 - 32.0*y108*y30*(r*y18 + y102*y18 + y27)**2/y104**3 + 6.10931107867609e-18*y164*y190/r**11 + 2.31644711733135e-17*y248/r**15) - y211*y298*y302 - y212*(y300*(12.0*y71 + 6.0) + y34*y5) + y236*(y303*(1.40625*y199 - y304*(2625.0*nu + 3375.0) + y305*(30.0*nu - 45.0)) + y317 + y36*(0.5*y161*(20.0*nu + 5.0) + y24*(-25.0*X_1 + 25.0*X_2) + 22.5*y5))) - y63*(4.0*r + y0*y45 + y9)*(pphi*y19*y20*y60 + pphi*y4*y58 - 2.0*y14*y50 - y15*y47 - y22*y49) + (-1.5*r*y62 + 1.0*(r + 0.5*y1 + 0.5*y64 + y65)**2/y61**3)*(pphi*y74 + pphi*y75 + y15*y66 + y22*y68 + y69*y70)

        cdef double d2Heffdphi2 = 0

        cdef double d2Heffdpr2 = y203*y362*y363*(1.52732776966902e-18*y168*y372 + 1769.32503588175*y168*y78 - 339.525257375589*y238*y79 + y364*y365 + y364*y366 + y364*y371 + y367*y368 + y367*y369 + 56.0*y370*y80 + y374 + 6.82947265356779*y80*y85 + 83.0341915336443*y81*y82) - 5129029357728.48*y388*(5.74938229854254e-11*y191*y387 + y365*y383 + y366*y383 + y368*y384 + y369*y384 + y371*y383 + y376 - 0.115137213510253*y378 + 0.020112800850135*y381 + 0.00165425616626294*y382 + 0.0135644946594222*y385 + 8.63226223952612e-22*y386)**2

        cdef double d2Heffdpphi2 = y10*(y14*(y28*(-4.171875*nu - 4.14322916666667*y32 + 0.4609375) + y389*y390 + y51*(-2.8125*nu - 0.9375)) + y22*(y28*(-0.546875*nu - 1.59635416666667*y32 - 0.4609375) + y390*y399 + y51*(0.9375 - 0.5625*nu)) + y391*y398 + y400*(2.0*X_1 - 2.0*X_2)) + y203*y362*y363*(y105*y244*(-y0 - 4.0) + y392) - 58982400.0*y388*(-pphi*y105*y108 + pphi*y51)**2

        cdef double  d2Heffdrdphi = 0

        cdef double d2Heffdrdpr = 0.5*y362*(y204*(-2653.98755382262*nu*y237*y375 + 237.667680162912*nu*y377*y77 + 3.39084201388889e-8*prst*y12*y169*y173*y182*y190*y297 + 3.39084201388889e-8*prst*y12*y169*y173*y187*y190*y268 + 1.35633680555556e-7*prst*y169*y17*y173*y182*y187*y6 - prst*y173*y188*y191*y351 - prst*y191*y270*y346 - y100*y375*y406 + 4.07287405245073e-18*y134*y149*y159*y162*y249*y250*y375 - 3.70181785051137e-33*y134*y149*y162*y336*y375 + 3.03943751362543e-44*y134*y159*y162*y165*y166*y182*y268*y375 - 2.65161071123094e-22*y149*y277*y279*y375 - 6.61842033523243e-18*y163*y327*y375 + 4.0*y17*y242*y375 - 12.0*y17*y375*y90 - 18.0*y17*y377*y95 - 0.975638950243592*y239*y379 - 8.0*y28*y375*y87 - 12.0*y28*y377*y92 - 16.0*y28*y379*y97 - y343*y387 - 29.655068404873*y380*y79) - y403*y404 + y403*y405) - y403*y407

        cdef double d2Heffdrdpphi = y10*(y14*(-y29*y409 - y33*y410 - y39*y411) - y21*y49 + y21*y60 + y22*(-y409*y41 - y410*y42 - y411*y43) - y391*y50 + y4*y58 - y408*y47) + 0.5*y362*(y204*(y319*y69 + y321*y412 - y409 - y413) - y404*y415 + y405*y415) - y407*y415 - y63*(y0 + y6 + y64)*(pphi*y20*y400 + y14*y398 + y21*y68 + y408*y66 + y51*y70 + y74 + y75)

        cdef double d2Heffdphidpr = 0

        cdef double d2Heffdphidpphi = 0

        cdef double d2Heffdprdpphi = 14745600.0*y388*y403*y414

        # Compute H Hessian
        cdef double d2Hdr2 = (-(dHeffdr**2/H**3)*(M**2/nu) + d2Heffdr2/H) * M*M / nu
        cdef double d2Hdphi2 = (-(dHeffdphi**2/H**3)*(M**2/nu) + d2Heffdphi2/H) * M*M / nu
        cdef double d2Hdpr2 = (-(dHeffdpr**2/H**3)*(M**2/nu) + d2Heffdpr2/H) * M*M / nu
        cdef double d2Hdpphi2 = (-(dHeffdpphi**2/H**3)*(M**2/nu) + d2Heffdpphi2/H) * M*M / nu
        cdef double d2Hdrdphi = (-(dHeffdr*dHeffdphi/H**3)*(M**2/nu) + d2Heffdrdphi/H) * M*M / nu
        cdef double d2Hdrdpr = (-(dHeffdr*dHeffdpr/H**3)*(M**2/nu) + d2Heffdrdpr/H) * M*M / nu
        cdef double d2Hdrdpphi = (-(dHeffdr*dHeffdpphi/H**3)*(M**2/nu) + d2Heffdrdpphi/H) * M*M / nu
        cdef double d2Hdphidpr = (-(dHeffdphi*dHeffdpr/H**3)*(M**2/nu) + d2Heffdphidpr/H) * M*M / nu
        cdef double d2Hdphidpphi = (-(dHeffdphi*dHeffdpphi/H**3)*(M**2/nu) + d2Heffdphidpphi/H) * M*M / nu
        cdef double d2Hdprdpphi = (-(dHeffdpr*dHeffdpphi/H**3)*(M**2/nu) + d2Heffdprdpphi/H) * M*M / nu

        return np.array([d2Hdr2, d2Hdrdphi, d2Hdrdpr, d2Hdrdpphi, d2Hdrdphi, d2Hdphi2, d2Hdphidpr, d2Hdphidpphi, d2Hdrdpr, d2Hdphidpr, d2Hdpr2, d2Hdprdpphi, d2Hdrdpphi, d2Hdphidpphi, d2Hdprdpphi, d2Hdpphi2]).reshape(4, 4)

    cdef double xi(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2):

        """
        Compute the tortoise factor \csi to convert between pr and prst.

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.

        Returns:
           (double) xi
        """

        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double pphi = L

        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs.a6
        cdef double dSO = c_coeffs.dSO


        # Extra quantities used in the Jacobian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2

        # Actual Hamiltonian expressions
        cdef double d5 = 0

        cdef double Dbpm = r*(6730497718123.02*nu**3 + 22295347200.0*nu**2*d5 + 133772083200.0*nu**2*r**2 + 1822680546449.21*nu**2*r + 80059249540278.2*nu**2 + 22295347200.0*nu*d5*r - 193226342400.0*nu*d5 + 2589101062873.81*nu*r**2 + 10611661054566.2*nu*r - 12049908701745.2*nu + 5107745331375.71*r**2 - 326837426.241486*r*(14700.0*nu + 42911.0) - 39476764256925.6*r - (-5041721180160.0*nu**2 - 25392914995744.3*nu - 879923036160.0*r**2 - 283115520.0*r*(14700.0*nu + 42911.0) + 104186110149937.0)*log(r) + 5787938193408.0*log(r)**2 + 275059053208689.0)/(55296.0*nu*(14515200.0*nu**3 - 42636451.6032331*nu**2 - 7680.0*nu*(315.0*d5 + 890888.810272497) + 4331361844.61149*nu + 1002013764.01019) - 967680.0*r**3*(-138240.0*nu**2 - 2675575.66847905*nu - 5278341.3229329) - 9216.0*r**2*(-197773496.793534*nu**2 - 7680.0*nu*(315.0*d5 + 405152.309729121) + 2481453539.84635*nu + 5805304367.87913) + r*(5927865218923.02*nu**3 + 70778880.0*nu**2*(315.0*d5 + 2561145.80918574) - 138141470005001.0*nu**2 - 4718592.0*nu*(40950.0*d5 + 86207832.4415642) + 450172889755120.0*nu + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0) + 5787938193408.0*r*log(r)**2 + (-1698693120.0*nu*(11592.0*nu + 69847.0) + 879923036160.0*r**3 + 283115520.0*r**2*(14700.0*nu + 42911.0) + 49152.0*r*(102574080.0*nu**2 + 409207698.136075*nu - 2119671837.36038))*log(r))

        cdef double Apm = 7680.0*r**4*(-5416406.59541186*nu**2 + 28.0*nu*(1920.0*a6 + 733955.307463037) + 2048.0*nu*(756.0*nu + 336.0*r + 407.0)*log(r) - 7.0*r*(-185763.092693281*nu**2 + 938918.400156317*nu - 245760.0) - 3440640.0)/(241555486248.807*nu**4 + 1120.0*nu**3*(-17833256.898555*r**2 - 163683964.822551*r - 1188987459.03162) + 7.0*nu**2*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 1426660551.8844*r**5 - 3089250703.76879*r**4 - 6178501407.53758*r**3 + 2064783811.32587*r**2 + 122635399361.987*r + 276057889687.011) + 67645734912.0*nu**2*log(r)**2 + 53760.0*nu*(7680.0*a6*(r**4 + 2.0*r**3 + 4.0*r**2 + 8.0*r + 16.0) + 128.0*r*(-6852.34813868015*r**4 + 4264.6962773603*r**3 + 8529.39255472061*r**2 + 13218.7851094412*r - 33722.4297811176) + 113485.217444961*r*(-r**4 + 2.0*r**3 + 4.0*r**2 + 8.0*r + 16.0) + 148.04406601634*r*(349.0*r**4 + 1926.0*r**3 + 3852.0*r**2 + 7704.0*r + 36400.0)) + 32768.0*nu*(-1882456.23663972*nu**2 - 38842241.4769507*nu + 161280.0*r**5 + 480.0*r**4*(756.0*nu + 1079.0) + 960.0*r**3*(756.0*nu + 1079.0) + 1920.0*r**2*(588.0*nu + 1079.0) + 240.0*r*(-3024.0*nu**2 - 7466.27061066206*nu + 17264.0) + 13447680.0)*log(r) + 13212057600.0*r**5)

        cdef double ap = chi_1*X_1 + chi_2*X_2

        cdef double ap2 = ap**2

        cdef double xi = Dbpm**0.5*r**2*(Apm + ap2/r**2)/(ap2 + r**2)

        return xi

    cpdef dynamics(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2):

        """
        Compute the dynamics from the Hamiltonian,i.e., dHdr, dHdphi, dHdpr, dHdpphi,H and xi.

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.

        Returns:
           (tuple) dHdr, dHdphi, dHdpr, dHdpphi, H and xi
        """

        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double pphi = L

        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs['a6']
        cdef double dSO = c_coeffs['dSO']

        # Extra quantities used in the Jacobian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double x0 = r**2
        cdef double x1 = chi_1*X_1
        cdef double x2 = chi_2*X_2
        cdef double x3 = x1 + x2
        cdef double x4 = x3**2
        cdef double x5 = 2.0*x4
        cdef double x6 = 2.0*r
        cdef double x7 = x0 + x4
        cdef double x8 = r*(-x6 + x7)
        cdef double x9 = (2.0*x0 + x5 + x8)**(-1)
        cdef double x10 = r**4
        cdef double x11 = 1/x10
        cdef double x12 = 3.0*nu
        cdef double x13 = pphi*x3
        cdef double x14 = r**3
        cdef double x15 = 1/x14
        cdef double x16 = x15*x4
        cdef double x17 = x1 - x2
        cdef double x18 = X_1 - X_2
        cdef double x19 = x17*x18
        cdef double x20 = pphi*x19
        cdef double x21 = 5.0*X_1 - 5.0*X_2
        cdef double x22 = x17*x3
        cdef double x23 = 0.0416666666666667*x21*x22 - 0.25*x4
        cdef double x24 = 2.0*x15
        cdef double x25 = 1/x0
        cdef double x26 = 0.71875*nu - 0.09375
        cdef double x27 = -1.40625*nu - 0.46875
        cdef double x28 = pphi**2
        cdef double x29 = x15*x28
        cdef double x30 = 2.0*x29
        cdef double x31 = nu**2
        cdef double x32 = -2.0859375*nu - 2.07161458333333*x31 + 0.23046875
        cdef double x33 = 3.0*x11
        cdef double x34 = x28*x33
        cdef double x35 = 0.5859375*nu + 1.34765625*x31 + 0.41015625
        cdef double x36 = pphi**4
        cdef double x37 = r**5
        cdef double x38 = 1/x37
        cdef double x39 = 4.0*x38
        cdef double x40 = x36*x39
        cdef double x41 = 0.34375*nu + 0.09375
        cdef double x42 = 0.46875 - 0.28125*nu
        cdef double x43 = -0.2734375*nu - 0.798177083333333*x31 - 0.23046875
        cdef double x44 = -0.3515625*nu + 0.29296875*x31 - 0.41015625
        cdef double x45 = nu*dSO*x15
        cdef double x46 = x25*x4
        cdef double x47 = 0.0416666666666667*x46
        cdef double x48 = x23*x25
        cdef double x49 = r**(-1)
        cdef double x50 = x25*x28
        cdef double x51 = x11*x36
        cdef double x52 = x3*(x25*(-5.53125*nu + 0.567708333333333*x31 - 0.078125) + x26*x49 + x27*x50 + x29*x32 + x35*x51 + 1.75)
        cdef double x53 = x19*(x25*(-0.03125*nu + 0.536458333333333*x31 + 0.078125) + x29*x43 + x41*x49 + x42*x50 + x44*x51 + 0.25)
        cdef double x54 = prst**4
        cdef double x55 = r**(-4.5)
        cdef double x56 = nu*x55
        cdef double x57 = prst**6
        cdef double x58 = r**(-3.5)
        cdef double x59 = nu*x58
        cdef double x60 = r**(-2.5)
        cdef double x61 = prst**8
        cdef double x62 = nu*x61
        cdef double x63 = nu*x49
        cdef double x64 = 0.121954868780449*x61
        cdef double x65 = 8.0*nu - 6.0*x31
        cdef double x66 = x25*x65
        cdef double x67 = nu**3
        cdef double x68 = 92.7110442849544*nu - 131.0*x31 + 10.0*x67
        cdef double x69 = x15*x54
        cdef double x70 = -2.78300763695006*nu - 5.4*x31 + 6.0*x67
        cdef double x71 = x25*x70
        cdef double x72 = nu**4
        cdef double x73 = -33.9782122170436*nu - 89.5298327361234*x31 + 188.0*x67 - 14.0*x72
        cdef double x74 = x15*x57
        cdef double x75 = 1.38977750996128*nu + 3.33842023648322*x31 + 3.42857142857143*x67 - 6.0*x72
        cdef double x76 = x61*x75
        cdef double x77 = log(r)
        cdef double x78 = nu*(452.542166996693 - 51.6952380952381*x77) + x31*(118.4*x77 - 1796.13660498019) + 602.318540416564*x67
        cdef double x79 = x11*x54
        cdef double x80 = r + 2.0
        cdef double x81 = x4*x80
        cdef double x82 = r*x4
        cdef double x83 = x10 + x80*x82
        cdef double x84 = 1/x83
        cdef double x85 = x28*x49*x84
        cdef double x86 = r**(-13)
        cdef double x87 = x7**4
        cdef double x88 = 756.0*nu
        cdef double x89 = 336.0*r + x88 + 407.0
        cdef double x90 = 2048.0*nu*x77*x89 + 28.0*nu*(1920.0*a6 + 733955.307463037) - 7.0*r*(938918.400156317*nu - 185763.092693281*x31 - 245760.0) - 5416406.59541186*x31 - 3440640.0
        cdef double x91 = x77**2
        cdef double x92 = x31*x91
        cdef double x93 = x67*(-163683964.822551*r - 17833256.898555*x0 - 1188987459.03162)
        cdef double x94 = x31*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 122635399361.987*r + 2064783811.32587*x0 - 3089250703.76879*x10 - 6178501407.53758*x14 + 1426660551.8844*x37 + 276057889687.011)
        cdef double x95 = 588.0*nu + 1079.0
        cdef double x96 = x88 + 1079.0
        cdef double x97 = x14*x96
        cdef double x98 = -38842241.4769507*nu + 240.0*r*(-7466.27061066206*nu - 3024.0*x31 + 17264.0) + 1920.0*x0*x95 + 480.0*x10*x96 - 1882456.23663972*x31 + 161280.0*x37 + 960.0*x97 + 13447680.0
        cdef double x99 = nu*x77
        cdef double x100 = x98*x99
        cdef double x101 = 8.0*r
        cdef double x102 = 4.0*x0 + x101 + 2.0*x14 + 16.0
        cdef double x103 = 7680.0*a6
        cdef double x104 = 128.0*r
        cdef double x105 = 7704.0*r
        cdef double x106 = 148.04406601634*r
        cdef double x107 = 113485.217444961*r
        cdef double x108 = nu*(x103*(x10 + x102) + x104*(13218.7851094412*r + 8529.39255472061*x0 - 6852.34813868015*x10 + 4264.6962773603*x14 - 33722.4297811176) + x106*(3852.0*x0 + 349.0*x10 + x105 + 1926.0*x14 + 36400.0) + x107*(-x10 + x102))
        cdef double x109 = (32768.0*x100 + 53760.0*x108 + 13212057600.0*x37 + 241555486248.807*x72 + 67645734912.0*x92 + 1120.0*x93 + 7.0*x94)**(-1)
        cdef double x110 = x10*x109*x90
        cdef double x111 = x110 + 0.000130208333333333*x46
        cdef double x112 = x111**(-4)
        cdef double x113 = r*x91
        cdef double x114 = -630116198.873299*nu - 197773496.793534*x31 + 5805304367.87913
        cdef double x115 = x0*x114
        cdef double x116 = -2675575.66847905*nu - 138240.0*x31 - 5278341.3229329
        cdef double x117 = x116*x14
        cdef double x118 = nu*(-2510664218.28128*nu - 42636451.6032331*x31 + 14515200.0*x67 + 1002013764.01019)
        cdef double x119 = 43393301259014.8*nu + 43133561885859.3*x31 + 5927865218923.02*x67 + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0
        cdef double x120 = r*x119
        cdef double x121 = 14700.0*nu + 42911.0
        cdef double x122 = 283115520.0*x121
        cdef double x123 = -1698693120.0*nu*(11592.0*nu + 69847.0) + 49152.0*r*(409207698.136075*nu + 102574080.0*x31 - 2119671837.36038) + x0*x122 + 879923036160.0*x14
        cdef double x124 = x123*x77
        cdef double x125 = (x113 - 1.59227685093395e-9*x115 - 1.67189069348064e-7*x117 + 9.55366110560367e-9*x118 + 1.72773095804465e-13*x120 + 1.72773095804465e-13*x124)**2
        cdef double x126 = nu*r
        cdef double x127 = nu*x0
        cdef double x128 = r*x31
        cdef double x129 = r*x121
        cdef double x130 = x0*x31
        cdef double x131 = 5041721180160.0*x31 - 104186110149937.0
        cdef double x132 = -25392914995744.3*nu - r*x122 - 879923036160.0*x0 - x131
        cdef double x133 = x132*x77
        cdef double x134 = -0.0438084424460039*nu - 0.143521050466841*r + 0.0185696317637669*x0 + 0.0385795738434214*x126 + 0.00941289164152486*x127 + 0.00662650629087394*x128 - 1.18824456940711e-6*x129 + 0.000486339502879429*x130 - 3.63558293513537e-15*x133 + 0.291062041428379*x31 + 0.0244692826489756*x67 + 0.0210425293255724*x91 + 1
        cdef double x135 = x134**(-2)
        cdef double x136 = 4.0*x31
        cdef double x137 = x17**2
        cdef double x138 = -0.46875*x137*(-5.0*nu + x136 + 1.0) + 0.0625*x17*x21*x3*(18.0*nu - 1.0) - 0.15625*x4*(-33.0*nu + 32.0*x31 - 5.0)
        cdef double x139 = x112*x125*x135*x138*x54*x87
        cdef double x140 = x7**2
        cdef double x141 = prst**2
        cdef double x142 = x111**(-2)
        cdef double x143 = 1822680546449.21*x31
        cdef double x144 = 5787938193408.0*x91
        cdef double x145 = -12049908701745.2*nu + r*x143 - 39476764256925.6*r + 5107745331375.71*x0 + 10611661054566.2*x126 + 2589101062873.81*x127 - 326837426.241486*x129 + 133772083200.0*x130 - x133 + x144 + 80059249540278.2*x31 + 6730497718123.02*x67 + 275059053208689.0
        cdef double x146 = 1/x145
        cdef double x147 = 0.0625*x137
        cdef double x148 = 0.125*x22
        cdef double x149 = -1171.0*nu - 861.0
        cdef double x150 = 0.015625*x4
        cdef double x151 = x137*(115.0*nu + x136 - 37.0)
        cdef double x152 = 0.03125*x22
        cdef double x153 = x18*(26.0*nu + 449.0)
        cdef double x154 = 5787938193408.0*x113 - 9216.0*x115 - 967680.0*x117 + 55296.0*x118 + x120 + x124
        cdef double x155 = 1/x154
        cdef double x156 = x145*x155*x37
        cdef double x157 = x109*x90
        cdef double x158 = x11*(x149*x150 + 0.015625*x151 + x152*x153) + x15*(x147*(12.0*nu - 3.0) + x148*(-21.0*X_1 + 21.0*X_2) + x4*(x12 + 2.8125)) + 7680.0*x156*x157 + x46
        cdef double x159 = x140*x141*x142*x146*x154*x158
        cdef double x160 = 1.27277314139085e-19*x139*x86 + 1.69542100694444e-8*x159*x38 + x25*x76 + x50 + 147.443752990146*x54*x56 + x54*x66 - 11.3175085791863*x57*x59 + x57*x71 + 1.48275342024365*x60*x62 + x63*x64 + x68*x69 + x73*x74 + x78*x79 - x81*x85 + 1.0
        cdef double x161 = x46*(2.0*x49 + 1.0) + 1.0
        cdef double x162 = 1/x161
        cdef double x163 = x137*(4.0*nu + 1.0)
        cdef double x164 = -x21*x22
        cdef double x165 = x137*(-27.0*nu + 28.0*x31 - 3.0)
        cdef double x166 = x152*(-39.0*X_1 + 39.0*X_2)
        cdef double x167 = x11*(0.125*x163 + 0.25*x164 + 1.125*x4) + 7680.0*x110 + x38*(-x150*(175.0*nu + 225.0) + 0.046875*x165 + x166*(2.0*nu - 3.0)) + x46
        cdef double x168 = x162*x167
        cdef double x169 = (x160*x168)**(-0.5)
        cdef double x170 = 4.0*x49 + 2.0
        cdef double x171 = r**(-6)
        cdef double x172 = x15*x5
        cdef double x173 = -6572428.80109422*nu + 1300341.64885296*x31 + 2048.0*x63*x89 + 688128.0*x99 + 1720320.0
        cdef double x174 = x31*x49
        cdef double x175 = 4.0*x14
        cdef double x176 = 6.0*x0 + x101 + 8.0
        cdef double x177 = 1.31621673590926e-19*x90*(53760.0*nu*(3740417.71815805*r + 2115968.85907902*x0 - 938918.400156317*x10 + x103*(x175 + x176) + x104*(17058.7851094412*r + 12794.0888320809*x0 - 27409.3925547206*x14 + 13218.7851094412) + x106*(5778.0*x0 + x105 + 1396.0*x14 + 7704.0) + x107*(-x175 + x176) + 1057984.42953951*x14 + 2888096.47013111) + 66060288000.0*x10 + 135291469824.0*x174*x77 + 7.0*x31*(-117964800.0*a6 + 4129567622.65173*r - 18535504222.6128*x0 + 7133302759.42198*x10 - 12357002815.0752*x14 + 122635399361.987) + 32768.0*x63*x98 + 7.0*x67*(-5706642207.53758*r - 26189434371.6082) + 32768.0*x99*(-1791904.9465589*nu + 3840.0*r*x95 + 2880.0*x0*x96 + 806400.0*x10 - 725760.0*x31 + 1920.0*x97 + 4143360.0))/(1.35654132757922e-7*x100 + 2.22557561555966e-7*x108 + 0.0546957463279941*x37 + x72 + 0.28004222119933*x92 + 4.63661586574928e-9*x93 + 2.8978849160933e-11*x94)**2
        cdef double x178 = -7680.0*x10*x109*x173 + x10*x177 - 30720.0*x109*x14*x90 + x172
        cdef double x179 = 11575876386816.0*x77
        cdef double x180 = 5807150888816.34*nu + 10215490662751.4*r + 5178202125747.62*x126 + 267544166400.0*x128 - x132*x49 + x143 + x179*x49 + x77*(4161798144000.0*nu + 1759846072320.0*r + 12148770078720.0) - 53501685054374.1
        cdef double x181 = x125*x138*x54*x86*x87
        cdef double x182 = -18432.0*r*x114 - 2903040.0*x0*x116 + x119 + x123*x49 + x144 + x179 + x77*(20113376778784.3*nu + 2639769108480.0*x0 + 566231040.0*x129 + x131)
        cdef double x183 = x141*x158
        cdef double x184 = -x178
        cdef double x185 = x154*x38
        cdef double x186 = prst**3
        cdef double x187 = prst**5
        cdef double x188 = prst**7
        cdef double x189 = 4.0*x186
        cdef double x190 = 6.0*x187
        cdef double x191 = x167*x169/(x170*x46 + 2.0)
        cdef double x192 = 2.0*pphi
        cdef double x193 = x192*x25
        cdef double x194 = x15*x192
        cdef double x195 = 4.0*pphi**3*x11


        # Evaluate Hamiltonian
        cdef double H,xi
        H,xi = evaluate_H(q,p,chi_1,chi_2,m_1,m_2,M,nu,X_1,X_2,a6,dSO)

        # Heff Jacobian expressions
        cdef double dHeffdr = 0.5*x169*(x160*x162*(-x171*(-x150*(875.0*nu + 1125.0) + 0.234375*x165 + x166*(10.0*nu - 15.0)) - x178 - x38*(0.5*x163 + x164 + 4.5*x4)) - x160*x167*(-x11*x5 - x16*x170)/x161**2 + x168*(-663.496888455656*nu*r**(-5.5)*x54 - nu*x25*x64 + 39.6112800271521*nu*x55*x57 + 6.78168402777778e-8*x11*x141*x142*x146*x154*x158*x7 + x11*x54*(118.4*x174 - 51.6952380952381*x63) + 7.59859378406358e-45*x112*x135*x138*x154*x182*x54*x86*x87 - 9.25454462627843e-34*x112*x180*x181/x134**3 - 2.24091649004576e-37*x135*x140*x142*x154*x180*x183*x38 + 1.69542100694444e-8*x140*x141*x142*x146*x154*x38*(38400.0*x10*x109*x145*x155*x90 + 7680.0*x109*x145*x155*x173*x37 + 7680.0*x109*x155*x180*x37*x90 - x11*(x147*(36.0*nu - 9.0) + x148*(-63.0*X_1 + 63.0*X_2) + x4*(9.0*nu + 8.4375)) - x156*x177 - x172 - x38*(x148*x153 + 0.0625*x149*x4 + 0.0625*x151) - 2.29252167428035e-22*x145*x157*x182*x37/x125) + 1.69542100694444e-8*x140*x141*x142*x146*x158*x182*x38 - 8.47710503472222e-8*x159*x171 - x24*x76 + x25*x28*x4*x80*x84 + x28*x4*x49*x80*(x175 + x81 + x82)/x83**2 - x30 - x33*x57*x73 - x39*x54*x78 - x4*x85 - 3.70688355060912*x58*x62 - 2.0*x65*x69 - 3.0*x68*x79 - 2.0*x70*x74 - 4.41515887225116e-12*x140*x146*x183*x184*x185/x111**3 - 6.62902677807736e-23*x135*x181*x184/x111**5 + 1.01821851311268e-18*x112*x125*x135*x138*x54*x7**3/r**12 - 1.65460508380811e-18*x139/r**14)) + x9*(-dSO*x11*x12*x13 + pphi*x17*x18*(-x15*(-0.0625*nu + 1.07291666666667*x31 + 0.15625) - x25*x41 - x30*x42 - x34*x43 - x40*x44) + pphi*x3*(-x15*(-11.0625*nu + 1.13541666666667*x31 - 0.15625) - x25*x26 - x27*x30 - x32*x34 - x35*x40) - x13*x23*x24 - 0.0833333333333333*x16*x20) - 0.25*(r*(x6 - 2.0) + x6 + x7)*(pphi*x52 + pphi*x53 + x13*x45 + x13*x48 + x20*x47)/(x7 + 0.5*x8)**2

        cdef double  dHeffdphi = 0

        cdef double  dHeffdpr = x191*(11.8620273619492*nu*x188*x60 + 3.39084201388889e-8*prst*x140*x142*x146*x158*x185 + x11*x189*x78 + 5.09109256556341e-19*x112*x125*x135*x138*x186*x86*x87 + x15*x189*x68 + x15*x190*x73 + 589.775011960583*x186*x56 - 67.9050514751178*x187*x59 + 8.0*x188*x25*x75 + 0.975638950243592*x188*x63 + x189*x66 + x190*x71)

        cdef double  dHeffdpphi = x191*(2.0*pphi*x25 - pphi*x49*x5*x80*x84) + x9*(x13*(x193*x27 + x194*x32 + x195*x35) + x19*x47 + x20*(x193*x42 + x194*x43 + x195*x44) + x3*x45 + x3*x48 + x52 + x53)
        cdef double  M2 = M*M
        cdef double  nuH = nu*H
        # Compute H Jacobian
        cdef double  dHdr = M2 * dHeffdr / nuH
        cdef double  dHdphi = M2 * dHeffdphi / nuH
        cdef double  dHdpr = M2 * dHeffdpr / nuH
        cdef double  dHdpphi = M2 * dHeffdpphi / nuH
        cdef double result[6]
        result[:] = [dHdr, dHdphi, dHdpr, dHdpphi,H,xi]
        return result
        #return dHdr, dHdphi, dHdpr, dHdpphi,H,xi

    cpdef double omega(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2):

        """
        Compute the orbital frequency from the Hamiltonian.

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.

        Returns:
           (double) dHdpphi
        """

        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double pphi = L

        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs['a6']
        cdef double dSO = c_coeffs['dSO']


        # Extra quantities used in the Jacobian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double z0 = r**2
        cdef double z1 = chi_1*X_1
        cdef double z2 = chi_2*X_2
        cdef double z3 = z1 + z2
        cdef double z4 = z3**2
        cdef double z5 = 2.0*z4
        cdef double z6 = z0 + z4
        cdef double z7 = r**3
        cdef double z8 = 1/z7
        cdef double z9 = 1/z0
        cdef double z10 = z4*z9
        cdef double z11 = z1 - z2
        cdef double z12 = z11*(X_1 - X_2)
        cdef double z13 = 5.0*X_1 - 5.0*X_2
        cdef double z14 = z11*z3
        cdef double z15 = -1.40625*nu - 0.46875
        cdef double z16 = 2.0*pphi
        cdef double z17 = z16*z9
        cdef double z18 = nu**2
        cdef double z19 = -2.0859375*nu - 2.07161458333333*z18 + 0.23046875
        cdef double z20 = z16*z8
        cdef double z21 = 0.5859375*nu + 1.34765625*z18 + 0.41015625
        cdef double z22 = r**4
        cdef double z23 = 1/z22
        cdef double z24 = 4.0*pphi**3*z23
        cdef double z25 = 0.46875 - 0.28125*nu
        cdef double z26 = -0.2734375*nu - 0.798177083333333*z18 - 0.23046875
        cdef double z27 = -0.3515625*nu + 0.29296875*z18 - 0.41015625
        cdef double z28 = r**(-1)
        cdef double z29 = pphi**2
        cdef double z30 = z29*z9
        cdef double z31 = z29*z8
        cdef double z32 = pphi**4*z23
        cdef double z33 = r + 2.0
        cdef double z34 = z33*z4
        cdef double z35 = z28/(r*z34 + z22)
        cdef double z36 = z11**2
        cdef double z37 = r**5
        cdef double z38 = 1/z37
        cdef double z39 = 0.015625*z4
        cdef double z40 = nu**4
        cdef double z41 = log(r)
        cdef double z42 = z41**2
        cdef double z43 = nu**3
        cdef double z44 = 756.0*nu
        cdef double z45 = z44 + 1079.0
        cdef double z46 = 8.0*r + 4.0*z0 + 2.0*z7 + 16.0
        cdef double z47 = (2048.0*nu*z41*(336.0*r + z44 + 407.0) + 28.0*nu*(1920.0*a6 + 733955.307463037) - 7.0*r*(938918.400156317*nu - 185763.092693281*z18 - 245760.0) - 5416406.59541186*z18 - 3440640.0)/(32768.0*nu*z41*(-38842241.4769507*nu + 240.0*r*(-7466.27061066206*nu - 3024.0*z18 + 17264.0) + 1920.0*z0*(588.0*nu + 1079.0) - 1882456.23663972*z18 + 480.0*z22*z45 + 161280.0*z37 + 960.0*z45*z7 + 13447680.0) + 53760.0*nu*(7680.0*a6*(z22 + z46) + 113485.217444961*r*(-z22 + z46) + 148.04406601634*r*(7704.0*r + 3852.0*z0 + 349.0*z22 + 1926.0*z7 + 36400.0) + 128.0*r*(13218.7851094412*r + 8529.39255472061*z0 - 6852.34813868015*z22 + 4264.6962773603*z7 - 33722.4297811176)) + 67645734912.0*z18*z42 + 7.0*z18*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 122635399361.987*r + 2064783811.32587*z0 - 3089250703.76879*z22 + 1426660551.8844*z37 - 6178501407.53758*z7 + 276057889687.011) + 13212057600.0*z37 + 241555486248.807*z40 + 1120.0*z43*(-163683964.822551*r - 17833256.898555*z0 - 1188987459.03162))
        cdef double z48 = z22*z47
        cdef double z49 = z10 + z23*(-0.25*z13*z14 + 0.125*z36*(4.0*nu + 1.0) + 1.125*z4) + z38*(0.03125*z14*(2.0*nu - 3.0)*(-39.0*X_1 + 39.0*X_2) + 0.046875*z36*(-27.0*nu + 28.0*z18 - 3.0) - z39*(175.0*nu + 225.0)) + 7680.0*z48
        cdef double z50 = prst**4
        cdef double z51 = prst**6
        cdef double z52 = prst**8
        cdef double z53 = nu*z52
        cdef double z54 = 4.0*z18
        cdef double z55 = nu*r
        cdef double z56 = nu*z0
        cdef double z57 = r*z18
        cdef double z58 = 14700.0*nu + 42911.0
        cdef double z59 = r*z58
        cdef double z60 = z0*z18
        cdef double z61 = 283115520.0*z58
        cdef double z62 = z41*(-25392914995744.3*nu - r*z61 - 879923036160.0*z0 - 5041721180160.0*z18 + 104186110149937.0)
        cdef double z63 = r*z42
        cdef double z64 = z0*(-630116198.873299*nu - 197773496.793534*z18 + 5805304367.87913)
        cdef double z65 = z7*(-2675575.66847905*nu - 138240.0*z18 - 5278341.3229329)
        cdef double z66 = nu*(-2510664218.28128*nu - 42636451.6032331*z18 + 14515200.0*z43 + 1002013764.01019)
        cdef double z67 = r*(43393301259014.8*nu + 43133561885859.3*z18 + 5927865218923.02*z43 + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0)
        cdef double z68 = z41*(-1698693120.0*nu*(11592.0*nu + 69847.0) + 49152.0*r*(409207698.136075*nu + 102574080.0*z18 - 2119671837.36038) + z0*z61 + 879923036160.0*z7)
        cdef double z69 = 0.000130208333333333*z10 + z48
        cdef double z70 = -12049908701745.2*nu - 39476764256925.6*r + 5107745331375.71*z0 + 80059249540278.2*z18 + 5787938193408.0*z42 + 6730497718123.02*z43 + 10611661054566.2*z55 + 2589101062873.81*z56 + 1822680546449.21*z57 - 326837426.241486*z59 + 133772083200.0*z60 - z62 + 275059053208689.0
        cdef double z71 = 5787938193408.0*z63 - 9216.0*z64 - 967680.0*z65 + 55296.0*z66 + z67 + z68


        # Evaluate Hamiltonian
        cdef double H,xi
        H,xi =  evaluate_H(q,p,chi_1,chi_2,m_1,m_2,M,nu,X_1,X_2,a6,dSO)

        # Heff Jacobian expressions

        cdef double dHeffdpphi = z49*(z49*(147.443752990146*nu*r**(-4.5)*z50 - 11.3175085791863*nu*r**(-3.5)*z51 + 1.69542100694444e-8*prst**2*z38*z6**2*z71*(z10 + z23*(0.03125*z12*z3*(26.0*nu + 449.0) + 0.015625*z36*(115.0*nu + z54 - 37.0) + z39*(-1171.0*nu - 861.0)) + 7680.0*z37*z47*z70/z71 + z8*(0.125*z14*(-21.0*X_1 + 21.0*X_2) + 0.0625*z36*(12.0*nu - 3.0) + z4*(3.0*nu + 2.8125)))/(z69**2*z70) + 1.48275342024365*r**(-2.5)*z53 + z23*z50*(nu*(452.542166996693 - 51.6952380952381*z41) + z18*(118.4*z41 - 1796.13660498019) + 602.318540416564*z43) + 0.121954868780449*z28*z53 - z29*z34*z35 + z30 + z50*z8*(92.7110442849544*nu - 131.0*z18 + 10.0*z43) + z50*z9*(8.0*nu - 6.0*z18) + z51*z8*(-33.9782122170436*nu - 89.5298327361234*z18 - 14.0*z40 + 188.0*z43) + z51*z9*(-2.78300763695006*nu - 5.4*z18 + 6.0*z43) + z52*z9*(1.38977750996128*nu + 3.33842023648322*z18 - 6.0*z40 + 3.42857142857143*z43) + 1.0 + 1.27277314139085e-19*z50*z6**4*(0.0625*z11*z13*z3*(18.0*nu - 1.0) - 0.46875*z36*(-5.0*nu + z54 + 1.0) - 0.15625*z4*(-33.0*nu + 32.0*z18 - 5.0))*(z63 - 1.59227685093395e-9*z64 - 1.67189069348064e-7*z65 + 9.55366110560367e-9*z66 + 1.72773095804465e-13*z67 + 1.72773095804465e-13*z68)**2/(r**13*z69**4*(-0.0438084424460039*nu - 0.143521050466841*r + 0.0185696317637669*z0 + 0.291062041428379*z18 + 0.0210425293255724*z42 + 0.0244692826489756*z43 + 0.0385795738434214*z55 + 0.00941289164152486*z56 + 0.00662650629087394*z57 - 1.18824456940711e-6*z59 + 0.000486339502879429*z60 - 3.63558293513537e-15*z62 + 1)**2))/(z10*(2.0*z28 + 1.0) + 1.0))**(-0.5)*(-pphi*z33*z35*z5 + 2.0*pphi*z9)/(z10*(4.0*z28 + 2.0) + 2.0) + (nu*dSO*z3*z8 + pphi*z12*(z17*z25 + z20*z26 + z24*z27) + pphi*z3*(z15*z17 + z19*z20 + z21*z24) + 0.0416666666666667*z10*z12 + z12*(z25*z30 + z26*z31 + z27*z32 + z28*(0.34375*nu + 0.09375) + z9*(-0.03125*nu + 0.536458333333333*z18 + 0.078125) + 0.25) + z3*z9*(0.0416666666666667*z13*z14 - 0.25*z4) + z3*(z15*z30 + z19*z31 + z21*z32 + z28*(0.71875*nu - 0.09375) + z9*(-5.53125*nu + 0.567708333333333*z18 - 0.078125) + 1.75))/(r*(-2.0*r + z6) + 2.0*z0 + z5)

        # Compute H Jacobian

        cdef double omega = M * M * dHeffdpphi / (nu*H)

        return omega

    cpdef auxderivs(self, double[:]q,double[:]p,double chi_1,double chi_2,double m_1,double m_2):

        """
        Compute derivatives of the potentials which are used in the post-adiabatic approximation.

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.

        Returns:
           (tuple) dAdr, dBnpdr, dBnpadr, dxidr, dQdr, dQdprst, dHodddr

        """

        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double pphi = L
        cdef double Chi1 = chi_1
        cdef double Chi2 = chi_2
        cdef double Nu = m_1*m_2
        cdef double M = m_1+m_2
        cdef double X1 = m_1/M
        cdef double X2 = m_2/M
        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs['a6']
        cdef double dSO = c_coeffs['dSO']

        cdef double x0 =r**2
        cdef double  x1=1/x0
        cdef double  x2=Chi1*X1
        cdef double  x3=Chi2*X2
        cdef double  x4=x2 + x3
        cdef double  x5=x4**2
        cdef double  x6=x1*x5
        cdef double x7 =Nu**2
        cdef double x8 =log(r)
        cdef double x9 =756.0*Nu
        cdef double x10 =336.0*r + x9 + 407.0
        cdef double x11 =2048.0*Nu*x10*x8 + 28.0*Nu*(1920.0*a6 + 733955.307463037) - 7.0*r*(938918.400156317*Nu - 185763.092693281*x7 - 245760.0) - 5416406.59541186*x7 - 3440640.0
        cdef double x12 =r**4
        cdef double x13 =Nu**4
        cdef double x14 =r**5
        cdef double x15 =x8**2
        cdef double x16 =x15*x7
        cdef double x17 =Nu**3
        cdef double x18 =x17*(-163683964.822551*r - 17833256.898555*x0 - 1188987459.03162)
        cdef double x19 =r**3
        cdef double x20 =x7*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 122635399361.987*r + 2064783811.32587*x0 - 3089250703.76879*x12 + 1426660551.8844*x14 - 6178501407.53758*x19 + 276057889687.011)
        cdef double x21 =588.0*Nu + 1079.0
        cdef double x22 =x9 + 1079.0
        cdef double x23 =x19*x22
        cdef double x24 =-38842241.4769507*Nu + 240.0*r*(-7466.27061066206*Nu - 3024.0*x7 + 17264.0) + 1920.0*x0*x21 + 480.0*x12*x22 + 161280.0*x14 + 960.0*x23 - 1882456.23663972*x7 + 13447680.0
        cdef double x25 =Nu*x8
        cdef double x26 =x24*x25
        cdef double x27 =8.0*r
        cdef double x28 =2.0*x19
        cdef double x29 =4.0*x0 + x27 + x28 + 16.0
        cdef double x30 =7680.0*a6
        cdef double x31 =128.0*r
        cdef double x32 =7704.0*r
        cdef double x33 =148.04406601634*r
        cdef double x34 =113485.217444961*r
        cdef double x35 =Nu*(x30*(x12 + x29) + x31*(13218.7851094412*r + 8529.39255472061*x0 - 6852.34813868015*x12 + 4264.6962773603*x19 - 33722.4297811176) + x33*(3852.0*x0 + 349.0*x12 + 1926.0*x19 + x32 + 36400.0) + x34*(-x12 + x29))
        cdef double x36 =(241555486248.807*x13 + 13212057600.0*x14 + 67645734912.0*x16 + 1120.0*x18 + 7.0*x20 + 32768.0*x26 + 53760.0*x35)**(-1)
        cdef double x37 =x11*x12*x36
        cdef double x38 =7680.0*x37 + x6
        cdef double x39 =2.0*r
        cdef double x40 =Nu*r
        cdef double x41 =Nu*x0
        cdef double x42 =1822680546449.21*x7
        cdef double x43 =5787938193408.0*x15
        cdef double x44 =14700.0*Nu + 42911.0
        cdef double x45 =r*x44
        cdef double x46 =x0*x7
        cdef double x47 =283115520.0*x44
        cdef double x48 =5041721180160.0*x7 - 104186110149937.0
        cdef double x49 =-25392914995744.3*Nu - r*x47 - 879923036160.0*x0 - x48
        cdef double x50 =-12049908701745.2*Nu + r*x42 - 39476764256925.6*r + 5107745331375.71*x0 + 6730497718123.02*x17 + 10611661054566.2*x40 + 2589101062873.81*x41 + x43 - 326837426.241486*x45 + 133772083200.0*x46 - x49*x8 + 80059249540278.2*x7 + 275059053208689.0
        cdef double x51 =-630116198.873299*Nu - 197773496.793534*x7 + 5805304367.87913
        cdef double x52 =x0*x51
        cdef double x53 =-2675575.66847905*Nu - 138240.0*x7 - 5278341.3229329
        cdef double x54 =x19*x53
        cdef double x55 =Nu*(-2510664218.28128*Nu + 14515200.0*x17 - 42636451.6032331*x7 + 1002013764.01019)
        cdef double x56 =43393301259014.8*Nu + 5927865218923.02*x17 + 43133561885859.3*x7 + 86618264430493.3*(1.0 - 0.496948781616935*Nu)**2 + 188440788778196.0
        cdef double x57 =r*x56
        cdef double x58 =Nu*(11592.0*Nu + 69847.0)
        cdef double x59 =r*(409207698.136075*Nu + 102574080.0*x7 - 2119671837.36038)
        cdef double x60 =x0*x47 + 879923036160.0*x19 - 1698693120.0*x58 + 49152.0*x59
        cdef double x61 =r*x43 - 9216.0*x52 - 967680.0*x54 + 55296.0*x55 + x57 + x60*x8
        cdef double x62 =1/x61
        cdef double x63 =x50*x62
        cdef double x64 = sqrt(r*x63)
        cdef double x65 =x0 + x5
        cdef double x66 =x64/x65
        cdef double x67 =2.0*x5
        cdef double x68 =2.0*x0 + x67
        cdef double x69 =r*x7
        cdef double x70 =r**(-1)
        cdef double x71 =11575876386816.0*x8
        cdef double x72 =5807150888816.34*Nu + 10215490662751.4*r + 5178202125747.62*x40 + x42 - x49*x70 + 267544166400.0*x69 + x70*x71 + x8*(4161798144000.0*Nu + 1759846072320.0*r + 12148770078720.0) - 53501685054374.1
        cdef double x73 =(r*x15 - 1.59227685093395e-9*x52 - 1.67189069348064e-7*x54 + 9.55366110560367e-9*x55 + 1.72773095804465e-13*x57 + x8*(4.89147448606909e-5*x0*x44 + 0.152027027027027*x19 - 0.000293488469164146*x58 + 8.49214320498106e-9*x59))**2
        cdef double x74 =-18432.0*r*x51 - 2903040.0*x0*x53 + x43 + x56 + x60*x70 + x71 + x8*(20113376778784.3*Nu + 2639769108480.0*x0 + 566231040.0*x45 + x48)
        cdef double x75 =x50*x74/x73
        cdef double x76 =1/x19
        cdef double x77 =x67*x76
        cdef double x78 =Nu*x70
        cdef double x79 =-6572428.80109422*Nu + 2048.0*x10*x78 + 688128.0*x25 + 1300341.64885296*x7 + 1720320.0
        cdef double x80 =x7*x70
        cdef double x81 =4.0*x19
        cdef double x82 =6.0*x0 + x27 + 8.0
        cdef double x83 =1.31621673590926e-19*x11*(53760.0*Nu*(3740417.71815805*r + 2115968.85907902*x0 - 938918.400156317*x12 + 1057984.42953951*x19 + x30*(x81 + x82) + x31*(17058.7851094412*r + 12794.0888320809*x0 - 27409.3925547206*x19 + 13218.7851094412) + x33*(5778.0*x0 + 1396.0*x19 + x32 + 7704.0) + x34*(-x81 + x82) + 2888096.47013111) + 66060288000.0*x12 + 7.0*x17*(-5706642207.53758*r - 26189434371.6082) + 32768.0*x24*x78 + 32768.0*x25*(-1791904.9465589*Nu + 3840.0*r*x21 + 2880.0*x0*x22 + 806400.0*x12 + 1920.0*x23 - 725760.0*x7 + 4143360.0) + 7.0*x7*(-117964800.0*a6 + 4129567622.65173*r - 18535504222.6128*x0 + 7133302759.42198*x12 - 12357002815.0752*x19 + 122635399361.987) + 135291469824.0*x8*x80)/(x13 + 0.0546957463279941*x14 + 0.28004222119933*x16 + 4.63661586574928e-9*x18 + 2.8978849160933e-11*x20 + 1.35654132757922e-7*x26 + 2.22557561555966e-7*x35)**2
        cdef double x84 =-30720.0*x11*x19*x36 - 7680.0*x12*x36*x79 + x12*x83 + x77
        cdef double x85 =-x84
        cdef double x86 =r*(-x39 + x65)
        cdef double x87 =1/x12
        cdef double x88 =3.0*x87
        cdef double x89 =L*x4
        cdef double x90 =Nu*dSO*x89
        cdef double x91 =x5*x76
        cdef double x92 =x2 - x3
        cdef double x93 =X1 - X2
        cdef double x94 =L*x92*x93
        cdef double x95 =0.25*x5
        cdef double x96 =x4*x92
        cdef double x97 =x89*(-x95 + x96*(0.208333333333333*X1 - 0.208333333333333*X2))
        cdef double x98 =2.0*x76
        cdef double x99 =0.71875*Nu - 0.09375
        cdef double x100 =-1.40625*Nu - 0.46875
        cdef double x101 =L**2
        cdef double x102 =x101*x76
        cdef double x103 =2.0*x102
        cdef double x104 =-2.0859375*Nu - 2.07161458333333*x7 + 0.23046875
        cdef double x105 =x101*x88
        cdef double x106 =0.5859375*Nu + 1.34765625*x7 + 0.41015625
        cdef double x107 =L**4
        cdef double x108 =1/x14
        cdef double x109 =4.0*x108
        cdef double x110 =x107*x109
        cdef double x111 =0.34375*Nu + 0.09375
        cdef double x112 =0.46875 - 0.28125*Nu
        cdef double x113 =0.0625*Nu
        cdef double x114 =-0.2734375*Nu - 0.798177083333333*x7 - 0.23046875
        cdef double x115 =-0.3515625*Nu + 0.29296875*x7 - 0.41015625
        cdef double x116 =x1*x101
        cdef double x117 =x107*x87
        cdef double x118 =prst**3
        cdef double x119 =r**(-4.5)
        cdef double x120 =prst**5
        cdef double x121 =Nu*r**(-3.5)
        cdef double x122 =prst**7
        cdef double x123 =8.0*Nu - 6.0*x7
        cdef double x124 =4.0*x118
        cdef double x125 =92.7110442849544*Nu + 10.0*x17 - 131.0*x7
        cdef double x126 =-2.78300763695006*Nu + 6.0*x17 - 5.4*x7
        cdef double x127 =6.0*x120
        cdef double x128 =-33.9782122170436*Nu - 14.0*x13 + 188.0*x17 - 89.5298327361234*x7
        cdef double x129 =1.38977750996128*Nu - 6.0*x13 + 3.42857142857143*x17 + 3.33842023648322*x7
        cdef double x130 =Nu*(452.542166996693 - 51.6952380952381*x8) + 602.318540416564*x17 + x7*(118.4*x8 - 1796.13660498019)
        cdef double x131 =r**(-13)
        cdef double x132 =x65**4
        cdef double x133 =x37 + 0.000130208333333333*x6
        cdef double x134 =x133**(-4)
        cdef double x135 =-0.0438084424460039*Nu - 0.143521050466841*r + 0.0185696317637669*x0 + 0.0210425293255724*x15 + 0.0244692826489756*x17 + 0.0385795738434214*x40 + 0.00941289164152486*x41 - 1.18824456940711e-6*x45 + 0.000486339502879429*x46 + 0.00662650629087394*x69 + 0.291062041428379*x7 - x8*(-0.092318048431871*Nu - 0.0031990331744958*x0 - 1.02928995318398e-6*x45 - 0.0183295954863003*x7 + 0.378777244139245) + 1.0
        cdef double x136 =x135**(-2)
        cdef double x137 =4.0*x7
        cdef double x138 =x92**2
        cdef double x139 =5.0*X1 - 5.0*X2
        cdef double x140 =-0.46875*x138*(-5.0*Nu + x137 + 1.0) + x139*x4*x92*(1.125*Nu - 0.0625) - 0.15625*x5*(-33.0*Nu + 32.0*x7 - 5.0)
        cdef double x141 =x132*x134*x136*x140*x73
        cdef double x142 =prst**4
        cdef double x143 =prst**6
        cdef double x144 =prst**8
        cdef double x145 =r + 2.0
        cdef double x146 =r*x5
        cdef double x147 =x12 + x145*x146
        cdef double x148 =1/x147
        cdef double x149 =x6*(2.0*x70 + 1.0) + 1.0
        cdef double x150 =x138*(-27.0*Nu + 28.0*x7 - 3.0)
        cdef double x151 =x96*(-39.0*X1 + 39.0*X2)
        cdef double dxidr=x0*x38*(r*x62*x72 - 2.98505426338587e-26*r*x75 + x63)/(x64*x68) + x0*x66*x85 - x28*x38*x64/x65**2 + x38*x39*x66
        cdef double dHodddr=(L*x4*(-x1*x99 - x100*x103 - x104*x105 - x106*x110 - x76*(-11.0625*Nu + 1.13541666666667*x7 - 0.15625)) + L*x92*x93*(-x1*x111 - x103*x112 - x105*x114 - x110*x115 - x76*(-x113 + 1.07291666666667*x7 + 0.15625)) - x88*x90 - 0.0833333333333333*x91*x94 - x97*x98)/(x68 + x86) - (0.25*r*(x39 - 2.0) + 0.5*r + 0.25*x0 + x95)*(x1*x97 + 0.0416666666666667*x6*x94 + x76*x90 + x89*(x1*(-5.53125*Nu + 0.567708333333333*x7 - 0.078125) + x100*x116 + x102*x104 + x106*x117 + x70*x99 + 1.75) + x94*(x1*(-0.03125*Nu + 0.536458333333333*x7 + 0.078125) + x102*x114 + x111*x70 + x112*x116 + x115*x117 + 0.25))/(x65 + 0.5*x86)**2
        cdef double dQdprst=11.8620273619492*Nu*r**(-2.5)*x122 + 589.775011960583*Nu*x118*x119 + 8.0*x1*x122*x129 + x1*x123*x124 + x1*x126*x127 + 5.09109256556341e-19*x118*x131*x141 - 67.9050514751178*x120*x121 + 0.975638950243592*x122*x78 + x124*x125*x76 + x124*x130*x87 + x127*x128*x76
        cdef double dQdr=-663.496888455656*Nu*r**(-5.5)*x142 - 0.121954868780449*Nu*x1*x144 + 39.6112800271521*Nu*x119*x143 - x109*x130*x142 - 3.70688355060912*x121*x144 - x123*x142*x98 - 3.0*x125*x142*x87 - x126*x143*x98 - x128*x143*x88 - x129*x144*x98 + 7.59859378406358e-45*x131*x132*x134*x136*x140*x142*x61*x74 - 9.25454462627843e-34*x131*x132*x134*x140*x142*x72*x73/x135**3 - 6.62902677807736e-23*x131*x132*x136*x140*x142*x73*x85/x133**5 + x142*x87*(-51.6952380952381*x78 + 118.4*x80) + 1.01821851311268e-18*x134*x136*x140*x142*x65**3*x73/r**12 - 1.65460508380811e-18*x141*x142/r**14
        cdef double dBnpadr=r*x145*(x145*x5 + x146 + x81)/x147**2 - r*x148 - x145*x148
        cdef double dBnpdr=-x108*(0.0625*x138*(115.0*Nu + x137 - 37.0) + x5*(-73.1875*Nu - 53.8125) + x93*x96*(3.25*Nu + 56.125)) + 38400.0*x11*x12*x36*x50*x62 + 7680.0*x11*x14*x36*x62*x72 - 2.29252167428035e-22*x11*x14*x36*x75 + 7680.0*x14*x36*x50*x62*x79 - x14*x63*x83 - x77 - x87*(x138*(2.25*Nu - 0.5625) + x5*(9.0*Nu + 8.4375) + x96*(-7.875*X1 + 7.875*X2))
        cdef double dAdr=(-x108*(x138*(2.0*Nu + 0.5) - x139*x96 + 4.5*x5) - x84 - (0.234375*x150 + x151*(0.3125*Nu - 0.46875) - x5*(13.671875*Nu + 17.578125))/r**6)/x149 - (-x67*x87 + x91*(-4.0*x70 - 2.0))*(x108*(0.046875*x150 + x151*(x113 - 0.09375) - x5*(2.734375*Nu + 3.515625)) + x38 + x87*(x138*(0.5*Nu + 0.125) + 1.125*x5 + x96*(-1.25*X1 + 1.25*X2)))/x149**2

        return dAdr,dBnpdr,dBnpadr,dxidr,dQdr,dQdprst,dHodddr