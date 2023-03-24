
# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False,profile=True, linetrace=True

cimport cython
import numpy as np
cimport numpy as np

from pyseobnr.eob.utils.containers cimport EOBParams,CalibCoeffs
from pyseobnr.eob.hamiltonian.Hamiltonian_v5PHM_C cimport Hamiltonian_v5PHM_C


from libc.math cimport log, sqrt, exp, abs, tgamma,sin,cos




cdef class Ham_AvgS2precess_simple_cython_PA_AD(Hamiltonian_v5PHM_C):
    cdef public CalibCoeffs calibration_coeffs
    def __cinit__(self,EOBParams params):
        self.EOBpars = params

    def __call__(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL1,double chiL2):
        return self._call(q,p,chi1_v,chi2_v,m_1,m_2,chi_1,chi_2,chiL1,chiL2)

    cpdef _call(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL1,double chiL2):
        """
        Evaluate the Hamiltonian as well as several potentials.

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1_v (double[:]): Dimensionless spin vector of the primary.
          chi2_v (double[:]): Dimensionless spin vector of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.
          chi_1 (double): Projection of chi1_v onto the Newtonian orbital angular momentum unit vector (lN).
          chi_2 (double): Projection of chi2_v onto the Newtonian orbital angular momentum unit vector (lN).
          chiL1 (double): Projection of chi1_v onto the orbital angular momentum unit vector (l).
          chiL2 (double): Projection of chi2_v onto the orbital angular momentum unit vector (l).

        Returns:
           (tuple)  H,xi, A, Bnp, Bnpa, Qq, Heven, Hodd, Bp

        """


        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double chix1 = chi1_v[0]
        cdef double chiy1 = chi1_v[1]
        cdef double chiz1 = chi1_v[2]

        cdef double chix2 = chi2_v[0]
        cdef double chiy2 = chi2_v[1]
        cdef double chiz2 = chi2_v[2]

        cdef double pphi = L

        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs['a6']
        cdef double dSO = c_coeffs['dSO']


        # Extra quantities used in the Hamiltonian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double delta = self.EOBpars.p_params.delta


        # Actual Hamiltonian expressions
        cdef double Dbpm = r*(6730497718123.02*nu**3 + 133772083200.0*nu**2*r**2 + 1822680546449.21*nu**2*r + 80059249540278.2*nu**2 + 2589101062873.81*nu*r**2 + 10611661054566.2*nu*r - 12049908701745.2*nu + 5107745331375.71*r**2 - 326837426.241486*r*(14700.0*nu + 42911.0) - 39476764256925.6*r - (-5041721180160.0*nu**2 - 25392914995744.3*nu - 879923036160.0*r**2 - 283115520.0*r*(14700.0*nu + 42911.0) + 104186110149937.0)*log(r) + 5787938193408.0*log(r)**2 + 275059053208689.0)/(55296.0*nu*(14515200.0*nu**3 - 42636451.6032331*nu**2 - 2510664218.28128*nu + 1002013764.01019) - 967680.0*r**3*(-138240.0*nu**2 - 2675575.66847905*nu - 5278341.3229329) - 9216.0*r**2*(-197773496.793534*nu**2 - 630116198.873299*nu + 5805304367.87913) + r*(5927865218923.02*nu**3 + 43133561885859.3*nu**2 + 43393301259014.8*nu + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0) + 5787938193408.0*r*log(r)**2 + (-1698693120.0*nu*(11592.0*nu + 69847.0) + 879923036160.0*r**3 + 283115520.0*r**2*(14700.0*nu + 42911.0) + 49152.0*r*(102574080.0*nu**2 + 409207698.136075*nu - 2119671837.36038))*log(r))

        cdef double Apm = 7680.0*r**4*(-5416406.59541186*nu**2 + 28.0*nu*(1920.0*a6 + 733955.307463037) + 2048.0*nu*(756.0*nu + 336.0*r + 407.0)*log(r) - 7.0*r*(-185763.092693281*nu**2 + 938918.400156317*nu - 245760.0) - 3440640.0)/(241555486248.807*nu**4 + 1120.0*nu**3*(-17833256.898555*r**2 - 163683964.822551*r - 1188987459.03162) + 7.0*nu**2*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 1426660551.8844*r**5 - 3089250703.76879*r**4 - 6178501407.53758*r**3 + 2064783811.32587*r**2 + 122635399361.987*r + 276057889687.011) + 67645734912.0*nu**2*log(r)**2 + 53760.0*nu*(7680.0*a6*(r**4 + 2.0*r**3 + 4.0*r**2 + 8.0*r + 16.0) + 128.0*r*(-6852.34813868015*r**4 + 4264.6962773603*r**3 + 8529.39255472061*r**2 + 13218.7851094412*r - 33722.4297811176) + 113485.217444961*r*(-r**4 + 2.0*r**3 + 4.0*r**2 + 8.0*r + 16.0) + 148.04406601634*r*(349.0*r**4 + 1926.0*r**3 + 3852.0*r**2 + 7704.0*r + 36400.0)) + 32768.0*nu*(-1882456.23663972*nu**2 - 38842241.4769507*nu + 161280.0*r**5 + 480.0*r**4*(756.0*nu + 1079.0) + 960.0*r**3*(756.0*nu + 1079.0) + 1920.0*r**2*(588.0*nu + 1079.0) + 240.0*r*(-3024.0*nu**2 - 7466.27061066206*nu + 17264.0) + 13447680.0)*log(r) + 13212057600.0*r**5)

        cdef double t2 = chix2**2 + chiy2**2 + chiz2**2

        cdef double t1 = chix1**2 + chiy1**2 + chiz1**2

        cdef double ap2 = X_1**2*t1 + X_1*X_2*(2.0*chix1*chix2 + 2.0*chiy1*chiy2 + 2.0*chiz1*chiz2) + X_2**2*t2

        cdef double xi = Dbpm**0.5*r**2*(Apm + ap2/r**2)/(ap2 + r**2)

        cdef double apam = X_1**2*t1 - X_2**2*t2

        cdef double am2 = X_1**2*t1 - X_1*X_2*(2.0*chix1*chix2 + 2.0*chiy1*chiy2 + 2.0*chiz1*chiz2) + X_2**2*t2

        cdef double QSalign2 = prst**4*(-0.46875*am2*(4.0*nu**2 - 5.0*nu + 1.0) - 0.15625*ap2*(32.0*nu**2 - 33.0*nu - 5.0) + 0.3125*apam*delta*(18.0*nu - 1.0))/(r**3*xi**4)

        cdef double Qpm = 1.48275342024365*nu*prst**8/r**2.5 + 0.121954868780449*nu*prst**8/r - 11.3175085791863*nu*prst**6/r**3.5 + 147.443752990146*nu*prst**4/r**4.5 + prst**8*(-6.0*nu**4 + 3.42857142857143*nu**3 + 3.33842023648322*nu**2 + 1.38977750996128*nu)/r**2 + prst**6*(6.0*nu**3 - 5.4*nu**2 - 2.78300763695006*nu)/r**2 + prst**6*(-14.0*nu**4 + 188.0*nu**3 - 89.5298327361234*nu**2 - 33.9782122170436*nu)/r**3 + prst**4*(-6.0*nu**2 + 8.0*nu)/r**2 + prst**4*(10.0*nu**3 - 131.0*nu**2 + 92.7110442849544*nu)/r**3 + prst**4*(602.318540416564*nu**3 + nu**2*(118.4*log(r) - 1796.13660498019) + nu*(452.542166996693 - 51.6952380952381*log(r)))/r**4

        cdef double Qq = QSalign2 + Qpm

        cdef double Bnpa = -r*(r + 2.0)/(ap2*r*(r + 2.0) + r**4)

        cdef double BnpSalign2 = (0.1875*am2*(4.0*nu - 1.0) + ap2*(3.0*nu + 2.8125) - 2.625*apam*delta)/r**3 + (0.015625*am2*(4.0*nu**2 + 115.0*nu - 37.0) + 0.015625*ap2*(-1171.0*nu - 861.0) + 0.03125*apam*delta*(26.0*nu + 449.0))/r**4

        cdef double Bnp = Apm*Dbpm + BnpSalign2 + ap2/r**2 - 1.0

        cdef double amz = chi_1*X_1 - chi_2*X_2

        cdef double apz = chi_1*X_1 + chi_2*X_2

        cdef double napnam = -0.5*amz*apz + 0.5*apam

        cdef double amz2 = amz**2

        cdef double nam2 = 0.5*am2 - 0.5*amz2

        cdef double apz2 = apz**2

        cdef double nap2 = 0.5*ap2 - 0.5*apz2

        cdef double BpSprec2 = -nap2/r**2 + (nam2*(0.1875 - 0.75*nu) + nap2*(-1.75*nu - 0.9375) + napnam*(0.75 - 1.5*X_2))/r**3 + (-0.125*delta*napnam*(98.0*nu + 43.0) + 0.015625*nam2*(152.0*nu**2 - 1090.0*nu + 219.0) + 0.00520833333333333*nap2*(264.0*nu**2 - 1610.0*nu + 375.0))/r**4

        cdef double Bp = BpSprec2 + 1.0

        cdef double ASprec2 = 2.0*nap2/r**3 + (4.125*delta*napnam + 0.125*nam2*(-4.0*nu - 3.0) + 0.25*nap2*(7.0*nu - 31.0))/r**4 + (0.25*delta*napnam*(68.0*nu - 1.0) + 0.015625*nam2*(-328.0*nu**2 + 1166.0*nu - 171.0) + 0.00520833333333333*nap2*(-264.0*nu**2 + 2870.0*nu + 561.0))/r**5

        cdef double ASalign2 = (0.125*am2*(4.0*nu + 1.0) + 1.125*ap2 - 1.25*apam*delta)/r**4 + (0.046875*am2*(28.0*nu**2 - 27.0*nu - 3.0) - 0.390625*ap2*(7.0*nu + 9.0) - 1.21875*apam*delta*(2.0*nu - 3.0))/r**5

        cdef double A = (ASalign2 + ASprec2 + Apm + ap2/r**2)/(ap2*(1.0 + 2.0/r)/r**2 + 1.0)

        cdef double ap = X_1*chiL1 + X_2*chiL2

        cdef double lap = ap

        cdef double Heven = (A*(Bnpa*L**2*lap**2/r**2 + Bp*L**2/r**2 + Qq + prst**2*(Bnp + 1.0)/xi**2 + 1.0))**0.5

        cdef double am = X_1*chiL1 - X_2*chiL2

        cdef double lam = am

        cdef double Ga3 = L*lam*(-0.25*L**2*delta*nap2/r**3 + (0.0416666666666667*ap2*delta + nap2*(0.416666666666667 - 0.833333333333333*X_2))/r**2) + L*lap*(L**2*(-0.25*nap2 + napnam*(0.5 - X_2))/r**3 + (-0.25*ap2 + 0.208333333333333*apam*delta - 1.66666666666667*delta*napnam - 0.75*nap2)/r**2)

        cdef double SOcalib = L*nu*dSO*lap/r**3

        cdef double gam = L**4*(0.29296875*nu**2 - 0.3515625*nu - 0.41015625)/r**4 + L**2*(0.46875 - 0.28125*nu)/r**2 + L**2*(-0.798177083333333*nu**2 - 0.2734375*nu - 0.23046875)/r**3 + 0.25 + (0.34375*nu + 0.09375)/r + (0.536458333333333*nu**2 - 0.03125*nu + 0.078125)/r**2

        cdef double gap = L**4*(1.34765625*nu**2 + 0.5859375*nu + 0.41015625)/r**4 + L**2*(-1.40625*nu - 0.46875)/r**2 + L**2*(-2.07161458333333*nu**2 - 2.0859375*nu + 0.23046875)/r**3 + 1.75 + (0.71875*nu - 0.09375)/r + (0.567708333333333*nu**2 - 5.53125*nu - 0.078125)/r**2

        cdef double Hodd = (Ga3 + L*delta*gam*lam + L*gap*lap + SOcalib)/(ap2*(r + 2.0) + r**3)

        cdef double pr = prst/xi

        cdef double Heff = Heven + Hodd

        # Evaluate H_real/nu
        cdef double H = M * sqrt(1+2*nu*(Heff-1)) / nu
        return H,xi, A, Bnp, Bnpa, Qq, Heven, Hodd, Bp


    cpdef grad(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL1,double chiL2):
        """
        Compute the gradient of the Hamiltonian in polar coordinates.

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1_v (double[:]): Dimensionless spin vector of the primary.
          chi2_v (double[:]): Dimensionless spin vector of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.
          chi_1 (double): Projection of chi1_v onto the Newtonian orbital angular momentum unit vector (lN).
          chi_2 (double): Projection of chi2_v onto the Newtonian orbital angular momentum unit vector (lN).
          chiL1 (double): Projection of chi1_v onto the orbital angular momentum unit vector (l).
          chiL2 (double): Projection of chi2_v onto the orbital angular momentum unit vector (l).

        Returns:
           (tuple) dHdr, dHdphi, dHdpr, dHdpphi

        """

        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double chix1 = chi1_v[0]
        cdef double chiy1 = chi1_v[1]
        cdef double chiz1 = chi1_v[2]

        cdef double chix2 = chi2_v[0]
        cdef double chiy2 = chi2_v[1]
        cdef double chiz2 = chi2_v[2]

        cdef double pphi = L

        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs['a6']
        cdef double dSO = c_coeffs['dSO']


        # Extra quantities used in the Jacobian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double delta = self.EOBpars.p_params.delta
        cdef double x0 = r**3
        cdef double x1 = r + 2.0
        cdef double x2 = X_1*(2.0*chix1*chix2 + 2.0*chiy1*chiy2 + 2.0*chiz1*chiz2)
        cdef double x3 = X_2*x2
        cdef double x4 = X_1**2
        cdef double x5 = chix1**2 + chiy1**2 + chiz1**2
        cdef double x6 = x4*x5
        cdef double x7 = X_2**2*(chix2**2 + chiy2**2 + chiz2**2)
        cdef double x8 = x6 + x7
        cdef double x9 = x3 + x8
        cdef double x10 = x1*x9
        cdef double x11 = x0 + x10
        cdef double x12 = x11**(-1)
        cdef double x13 = r**4
        cdef double x14 = x13**(-1)
        cdef double x15 = 3.0*nu
        cdef double x16 = X_1*chiL1
        cdef double x17 = X_2*chiL2
        cdef double x18 = x16 + x17
        cdef double x19 = pphi*x18
        cdef double x20 = r**2
        cdef double x21 = x20**(-1)
        cdef double x22 = 0.71875*nu - 0.09375
        cdef double x23 = -1.40625*nu - 0.46875
        cdef double x24 = x0**(-1)
        cdef double x25 = pphi**2
        cdef double x26 = x24*x25
        cdef double x27 = 2.0*x26
        cdef double x28 = nu**2
        cdef double x29 = -2.0859375*nu - 2.07161458333333*x28 + 0.23046875
        cdef double x30 = 3.0*x14
        cdef double x31 = x25*x30
        cdef double x32 = 0.5859375*nu + 1.34765625*x28 + 0.41015625
        cdef double x33 = pphi**4
        cdef double x34 = r**5
        cdef double x35 = x34**(-1)
        cdef double x36 = 4.0*x35
        cdef double x37 = x33*x36
        cdef double x38 = 0.34375*nu + 0.09375
        cdef double x39 = 0.46875 - 0.28125*nu
        cdef double x40 = -0.2734375*nu - 0.798177083333333*x28 - 0.23046875
        cdef double x41 = -0.3515625*nu + 0.29296875*x28 - 0.41015625
        cdef double x42 = x16 - x17
        cdef double x43 = pphi*x42
        cdef double x44 = delta*x43
        cdef double x45 = chi_1*X_1
        cdef double x46 = chi_2*X_2
        cdef double x47 = x45 + x46
        cdef double x48 = x47**2
        cdef double x49 = -x48 + x9
        cdef double x50 = delta*x25
        cdef double x51 = delta*x9
        cdef double x52 = x49*(0.416666666666667 - 0.833333333333333*X_2)
        cdef double x53 = x45 - x46
        cdef double x54 = x4*x5 - x47*x53 - x7
        cdef double x55 = -0.125*x3 + 0.125*x48 + 0.5*x54*(0.5 - X_2) - 0.125*x6 - 0.125*x7
        cdef double x56 = x6 - x7
        cdef double x57 = delta*x56
        cdef double x58 = delta*x54
        cdef double x59 = nu*dSO*x24
        cdef double x60 = r**(-1)
        cdef double x61 = x21*x25
        cdef double x62 = x14*x33
        cdef double x63 = x18*(x21*(-5.53125*nu + 0.567708333333333*x28 - 0.078125) + x22*x60 + x23*x61 + x26*x29 + x32*x62 + 1.75)
        cdef double x64 = delta*(x21*(-0.03125*nu + 0.536458333333333*x28 + 0.078125) + x26*x40 + x38*x60 + x39*x61 + x41*x62 + 0.25)
        cdef double x65 = x24*x49
        cdef double x66 = x50*x65
        cdef double x67 = x42*(x21*(0.0416666666666667*x51 + 0.5*x52) - 0.125*x66)
        cdef double x68 = x26*x55
        cdef double x69 = x18*(x21*(0.208333333333333*delta*x56 - 0.625*x3 + 0.375*x48 - 0.833333333333333*x58 - 0.625*x6 - 0.625*x7) + x68)
        cdef double x70 = prst**4
        cdef double x71 = r**(-4.5)
        cdef double x72 = nu*x71
        cdef double x73 = prst**6
        cdef double x74 = r**(-3.5)
        cdef double x75 = nu*x74
        cdef double x76 = r**(-2.5)
        cdef double x77 = prst**8
        cdef double x78 = nu*x77
        cdef double x79 = nu*x60
        cdef double x80 = 0.121954868780449*x77
        cdef double x81 = 8.0*nu - 6.0*x28
        cdef double x82 = x21*x81
        cdef double x83 = nu**3
        cdef double x84 = 92.7110442849544*nu - 131.0*x28 + 10.0*x83
        cdef double x85 = x24*x70
        cdef double x86 = -2.78300763695006*nu - 5.4*x28 + 6.0*x83
        cdef double x87 = x21*x86
        cdef double x88 = nu**4
        cdef double x89 = -33.9782122170436*nu - 89.5298327361234*x28 + 188.0*x83 - 14.0*x88
        cdef double x90 = x24*x73
        cdef double x91 = 1.38977750996128*nu + 3.33842023648322*x28 + 3.42857142857143*x83 - 6.0*x88
        cdef double x92 = x77*x91
        cdef double x93 = log(r)
        cdef double x94 = nu*(452.542166996693 - 51.6952380952381*x93) + x28*(118.4*x93 - 1796.13660498019) + 602.318540416564*x83
        cdef double x95 = x14*x70
        cdef double x96 = r*x10 + x13
        cdef double x97 = x96**(-1)
        cdef double x98 = x18**2
        cdef double x99 = x97*x98
        cdef double x100 = x25*x60*x99
        cdef double x101 = 0.5*x49
        cdef double x102 = 0.5*x54
        cdef double x103 = -x3 + x8
        cdef double x104 = x103 - x53**2
        cdef double x105 = 0.5*x104
        cdef double x106 = x58*(98.0*nu + 43.0)
        cdef double x107 = 264.0*x28
        cdef double x108 = -1610.0*nu + x107 + 375.0
        cdef double x109 = 0.00260416666666667*x49
        cdef double x110 = -1090.0*nu + 152.0*x28 + 219.0
        cdef double x111 = 0.0078125*x104
        cdef double x112 = -x101*x21 + x14*(-0.0625*x106 + x108*x109 + x110*x111) + x24*(x101*(-1.75*nu - 0.9375) + x102*(0.75 - 1.5*X_2) + x105*(0.1875 - 0.75*nu)) + 1.0
        cdef double x113 = r**(-13)
        cdef double x114 = x20 + x9
        cdef double x115 = x114**4
        cdef double x116 = x21*x9
        cdef double x117 = 756.0*nu
        cdef double x118 = 336.0*r + x117 + 407.0
        cdef double x119 = 2048.0*nu*x118*x93 + 28.0*nu*(1920.0*a6 + 733955.307463037) - 7.0*r*(938918.400156317*nu - 185763.092693281*x28 - 245760.0) - 5416406.59541186*x28 - 3440640.0
        cdef double x120 = x93**2
        cdef double x121 = x120*x28
        cdef double x122 = x83*(-163683964.822551*r - 17833256.898555*x20 - 1188987459.03162)
        cdef double x123 = x28*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 122635399361.987*r - 6178501407.53758*x0 - 3089250703.76879*x13 + 2064783811.32587*x20 + 1426660551.8844*x34 + 276057889687.011)
        cdef double x124 = 588.0*nu + 1079.0
        cdef double x125 = x117 + 1079.0
        cdef double x126 = x0*x125
        cdef double x127 = -38842241.4769507*nu + 240.0*r*(-7466.27061066206*nu - 3024.0*x28 + 17264.0) + 1920.0*x124*x20 + 480.0*x125*x13 + 960.0*x126 - 1882456.23663972*x28 + 161280.0*x34 + 13447680.0
        cdef double x128 = nu*x93
        cdef double x129 = x127*x128
        cdef double x130 = 8.0*r
        cdef double x131 = 2.0*x0 + x130 + 4.0*x20 + 16.0
        cdef double x132 = 7680.0*a6
        cdef double x133 = 128.0*r
        cdef double x134 = 7704.0*r
        cdef double x135 = 148.04406601634*r
        cdef double x136 = 113485.217444961*r
        cdef double x137 = nu*(x132*(x13 + x131) + x133*(13218.7851094412*r + 4264.6962773603*x0 - 6852.34813868015*x13 + 8529.39255472061*x20 - 33722.4297811176) + x135*(1926.0*x0 + 349.0*x13 + x134 + 3852.0*x20 + 36400.0) + x136*(-x13 + x131))
        cdef double x138 = (67645734912.0*x121 + 1120.0*x122 + 7.0*x123 + 32768.0*x129 + 53760.0*x137 + 13212057600.0*x34 + 241555486248.807*x88)**(-1)
        cdef double x139 = x13*x138
        cdef double x140 = x119*x139
        cdef double x141 = 0.000130208333333333*x116 + x140
        cdef double x142 = x141**(-4)
        cdef double x143 = r*x120
        cdef double x144 = -630116198.873299*nu - 197773496.793534*x28 + 5805304367.87913
        cdef double x145 = x144*x20
        cdef double x146 = -2675575.66847905*nu - 138240.0*x28 - 5278341.3229329
        cdef double x147 = x0*x146
        cdef double x148 = nu*(-2510664218.28128*nu - 42636451.6032331*x28 + 14515200.0*x83 + 1002013764.01019)
        cdef double x149 = 43393301259014.8*nu + 43133561885859.3*x28 + 5927865218923.02*x83 + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0
        cdef double x150 = r*x149
        cdef double x151 = 14700.0*nu + 42911.0
        cdef double x152 = 283115520.0*x151
        cdef double x153 = -1698693120.0*nu*(11592.0*nu + 69847.0) + 49152.0*r*(409207698.136075*nu + 102574080.0*x28 - 2119671837.36038) + 879923036160.0*x0 + x152*x20
        cdef double x154 = x153*x93
        cdef double x155 = (x143 - 1.59227685093395e-9*x145 - 1.67189069348064e-7*x147 + 9.55366110560367e-9*x148 + 1.72773095804465e-13*x150 + 1.72773095804465e-13*x154)**2
        cdef double x156 = nu*r
        cdef double x157 = nu*x20
        cdef double x158 = r*x28
        cdef double x159 = r*x151
        cdef double x160 = x20*x28
        cdef double x161 = 5041721180160.0*x28 - 104186110149937.0
        cdef double x162 = -25392914995744.3*nu - r*x152 - x161 - 879923036160.0*x20
        cdef double x163 = x162*x93
        cdef double x164 = -0.0438084424460039*nu - 0.143521050466841*r + 0.0210425293255724*x120 + 0.0385795738434214*x156 + 0.00941289164152486*x157 + 0.00662650629087394*x158 - 1.18824456940711e-6*x159 + 0.000486339502879429*x160 - 3.63558293513537e-15*x163 + 0.0185696317637669*x20 + 0.291062041428379*x28 + 0.0244692826489756*x83 + 1
        cdef double x165 = x164**(-2)
        cdef double x166 = 4.0*x28
        cdef double x167 = 0.3125*delta*x56*(18.0*nu - 1.0) - 0.03125*(-33.0*nu + 32.0*x28 - 5.0)*(5.0*x3 + 5.0*x6 + 5.0*x7) - 0.03125*(-5.0*nu + x166 + 1.0)*(-15.0*x3 + 15.0*x6 + 15.0*x7)
        cdef double x168 = x115*x142*x155*x165*x167*x70
        cdef double x169 = x114**2
        cdef double x170 = prst**2
        cdef double x171 = x141**(-2)
        cdef double x172 = 1822680546449.21*x28
        cdef double x173 = 5787938193408.0*x120
        cdef double x174 = -12049908701745.2*nu + r*x172 - 39476764256925.6*r + 10611661054566.2*x156 + 2589101062873.81*x157 - 326837426.241486*x159 + 133772083200.0*x160 - x163 + x173 + 5107745331375.71*x20 + 80059249540278.2*x28 + 6730497718123.02*x83 + 275059053208689.0
        cdef double x175 = x174**(-1)
        cdef double x176 = 4.0*nu
        cdef double x177 = 3.0*x3
        cdef double x178 = 3.0*x6 + 3.0*x7
        cdef double x179 = -x177 + x178
        cdef double x180 = 0.0625*x179
        cdef double x181 = x57*(26.0*nu + 449.0)
        cdef double x182 = x9*(-1171.0*nu - 861.0)
        cdef double x183 = x103*(115.0*nu + x166 - 37.0)
        cdef double x184 = 5787938193408.0*x143 - 9216.0*x145 - 967680.0*x147 + 55296.0*x148 + x150 + x154
        cdef double x185 = x184**(-1)
        cdef double x186 = x174*x185
        cdef double x187 = x186*x34
        cdef double x188 = x119*x138
        cdef double x189 = 7680.0*x188
        cdef double x190 = x116 + x14*(0.03125*x181 + 0.015625*x182 + 0.015625*x183) + x187*x189 + x24*(x180*(x176 - 1.0) - 2.625*x57 + x9*(x15 + 2.8125))
        cdef double x191 = x169*x170*x171*x175*x184*x190
        cdef double x192 = -x1*x100 + x112*x61 + 1.27277314139085e-19*x113*x168 + 1.69542100694444e-8*x191*x35 + x21*x92 + 147.443752990146*x70*x72 + x70*x82 - 11.3175085791863*x73*x75 + x73*x87 + 1.48275342024365*x76*x78 + x79*x80 + x84*x85 + x89*x90 + x94*x95 + 1.0
        cdef double x193 = 2.0*x60
        cdef double x194 = x116*(x193 + 1.0) + 1.0
        cdef double x195 = x194**(-1)
        cdef double x196 = x103*(x176 + 1.0)
        cdef double x197 = x57*(2.0*nu - 3.0)
        cdef double x198 = 7.0*nu
        cdef double x199 = 0.390625*x3 + 0.390625*x6 + 0.390625*x7
        cdef double x200 = x49*(x198 - 31.0)
        cdef double x201 = x104*(-x176 - 3.0)
        cdef double x202 = x58*(68.0*nu - 1.0)
        cdef double x203 = x116 + x14*(0.125*x200 + 0.0625*x201 + 2.0625*x58) + x14*(0.125*x196 + 1.125*x3 - 1.25*x57 + 1.125*x6 + 1.125*x7) + 7680.0*x140 + x35*(x109*(2870.0*nu - x107 + 561.0) + x111*(1166.0*nu - 328.0*x28 - 171.0) + 0.125*x202) + x35*(0.015625*x179*(-27.0*nu + 28.0*x28 - 3.0) - 1.21875*x197 - x199*(x198 + 9.0)) + x65
        cdef double x204 = x195*x203
        cdef double x205 = (x192*x204)**(-0.5)
        cdef double x206 = -2.0*x3 - 2.0*x6 - 2.0*x7
        cdef double x207 = 4.0*x60 + 2.0
        cdef double x208 = 4.5*X_2
        cdef double x209 = r**(-6)
        cdef double x210 = -6572428.80109422*nu + 2048.0*x118*x79 + 688128.0*x128 + 1300341.64885296*x28 + 1720320.0
        cdef double x211 = x28*x60
        cdef double x212 = 4.0*x0
        cdef double x213 = x130 + 6.0*x20 + 8.0
        cdef double x214 = 1.31621673590926e-19*x119*(53760.0*nu*(3740417.71815805*r + 1057984.42953951*x0 - 938918.400156317*x13 + x132*(x212 + x213) + x133*(17058.7851094412*r - 27409.3925547206*x0 + 12794.0888320809*x20 + 13218.7851094412) + x135*(1396.0*x0 + x134 + 5778.0*x20 + 7704.0) + x136*(-x212 + x213) + 2115968.85907902*x20 + 2888096.47013111) + 32768.0*x127*x79 + 32768.0*x128*(-1791904.9465589*nu + 3840.0*r*x124 + 2880.0*x125*x20 + 1920.0*x126 + 806400.0*x13 - 725760.0*x28 + 4143360.0) + 66060288000.0*x13 + 135291469824.0*x211*x93 + 7.0*x28*(-117964800.0*a6 + 4129567622.65173*r - 12357002815.0752*x0 + 7133302759.42198*x13 - 18535504222.6128*x20 + 122635399361.987) + 7.0*x83*(-5706642207.53758*r - 26189434371.6082))/(0.28004222119933*x121 + 4.63661586574928e-9*x122 + 2.8978849160933e-11*x123 + 1.35654132757922e-7*x129 + 2.22557561555966e-7*x137 + 0.0546957463279941*x34 + x88)**2
        cdef double x215 = x13*x214
        cdef double x216 = 2.0*x24
        cdef double x217 = 11575876386816.0*x93
        cdef double x218 = 5807150888816.34*nu + 10215490662751.4*r + 5178202125747.62*x156 + 267544166400.0*x158 - x162*x60 + x172 + x217*x60 + x93*(4161798144000.0*nu + 1759846072320.0*r + 12148770078720.0) - 53501685054374.1
        cdef double x219 = x113*x115*x155*x167*x70
        cdef double x220 = -18432.0*r*x144 - 2903040.0*x146*x20 + x149 + x153*x60 + x173 + x217 + x93*(20113376778784.3*nu + 566231040.0*x159 + x161 + 2639769108480.0*x20)
        cdef double x221 = x170*x190
        cdef double x222 = x206*x24
        cdef double x223 = 7680.0*x210
        cdef double x224 = 30720.0*x0*x188 + x139*x223 - x215 + x222
        cdef double x225 = x184*x35
        cdef double x226 = prst**3
        cdef double x227 = prst**5
        cdef double x228 = prst**7
        cdef double x229 = 4.0*x226
        cdef double x230 = 6.0*x227
        cdef double x231 = x203*x205/(x116*x207 + 2.0)
        cdef double x232 = 2.0*pphi*x21
        cdef double x233 = pphi*x216
        cdef double x234 = 4.0*pphi**3*x14


        # Evaluate Hamiltonian
        cdef double H
        H,_,_,_,_,_,_,_,_ = self._call(q,p,chi1_v,chi2_v,m_1,m_2,chi_1,chi_2,chiL1,chiL2)

        # Heff Jacobian expressions
        cdef double dHeffdr = x12*(-dSO*x14*x15*x19 + x19*(x24*(1.25*x3 - 0.75*x48 - 0.416666666666667*x57 + 1.66666666666667*x58 + 1.25*x6 + 1.25*x7) - x31*x55) + x19*(-x21*x22 - x23*x27 - x24*(-11.0625*nu + 1.13541666666667*x28 - 0.15625) - x29*x31 - x32*x37) + x43*(0.375*x14*x49*x50 - x24*(0.0833333333333333*x51 + x52)) + x44*(-x21*x38 - x24*(-0.0625*nu + 1.07291666666667*x28 + 0.15625) - x27*x39 - x31*x40 - x37*x41)) + 0.5*x205*(x192*x195*(30720.0*x0*x119*x138 + 7680.0*x13*x138*x210 - x14*(x177 + x178 - 3.0*x48) + x206*x24 - x209*(x109*(14350.0*nu - 1320.0*x28 + 2805.0) + x111*(5830.0*nu - 1640.0*x28 - 855.0) + 0.625*x202) - x209*(0.015625*x179*(-135.0*nu + 140.0*x28 - 15.0) - 6.09375*x197 - x199*(35.0*nu + 45.0)) - x215 - x35*(0.5*x200 + 0.25*x201 + 8.25*x58) - x35*(0.5*x196 + x2*x208 - 5.0*x57 + 4.5*x6 + 4.5*x7)) - x192*x203*(x14*x206 - x207*x24*x9)/x194**2 + x204*(-663.496888455656*nu*r**(-5.5)*x70 - nu*x21*x80 + 39.6112800271521*nu*x71*x73 + x1*x21*x25*x97*x98 + x1*x25*x60*x98*(r*x9 + x10 + x212)/x96**2 - x100 - x112*x27 + 7.59859378406358e-45*x113*x115*x142*x165*x167*x184*x220*x70 + 6.78168402777778e-8*x114*x14*x170*x171*x175*x184*x190 + x14*x70*(118.4*x211 - 51.6952380952381*x79) - 9.25454462627843e-34*x142*x218*x219/x164**3 - 2.24091649004576e-37*x165*x169*x171*x184*x218*x221*x35 + 1.69542100694444e-8*x169*x170*x171*x175*x184*x35*(x138*x187*x223 - x14*(x180*(12.0*nu - 3.0) - 7.875*x57 + x9*(9.0*nu + 8.4375)) + 38400.0*x140*x186 + x185*x189*x218*x34 - x187*x214 + x222 - x35*(0.125*x181 + 0.0625*x182 + 0.0625*x183) - 2.29252167428035e-22*x174*x188*x220*x34/x155) + 1.69542100694444e-8*x169*x170*x171*x175*x190*x220*x35 - 8.47710503472222e-8*x191*x209 + x21*x25*(-x14*(x101*(-5.25*nu - 2.8125) + x102*(2.25 - x208) + x105*(0.5625 - 2.25*nu)) + x24*x49 - x35*(0.03125*x104*x110 - 0.25*x106 + 0.0104166666666667*x108*x49)) - x216*x92 - x30*x73*x89 - x36*x70*x94 - 3.70688355060912*x74*x78 - 2.0*x81*x85 - 3.0*x84*x95 - 2.0*x86*x90 - 4.41515887225116e-12*x169*x175*x221*x224*x225/x141**3 - 6.62902677807736e-23*x165*x219*x224/x141**5 + 1.01821851311268e-18*x114**3*x142*x155*x165*x167*x70/r**12 - 1.65460508380811e-18*x168/r**14)) - (3.0*x20 + x9)*(pphi*x63 + pphi*x67 + pphi*x69 + x19*x59 + x43*x64)/x11**2

        cdef double dHeffdphi = 0

        cdef double dHeffdpr = x231*(11.8620273619492*nu*x228*x76 + 3.39084201388889e-8*prst*x169*x171*x175*x190*x225 + 5.09109256556341e-19*x113*x115*x142*x155*x165*x167*x226 + x14*x229*x94 + 8.0*x21*x228*x91 + 589.775011960583*x226*x72 - 67.9050514751178*x227*x75 + 0.975638950243592*x228*x79 + x229*x24*x84 + x229*x82 + x230*x24*x89 + x230*x87)

        cdef double dHeffdpphi = x12*(x18*x59 + 2.0*x18*x68 + x19*(x23*x232 + x233*x29 + x234*x32) + x42*x64 - 0.25*x42*x66 + x44*(x232*x39 + x233*x40 + x234*x41) + x63 + x67 + x69) + x231*(-pphi*x1*x193*x99 + 2.0*pphi*x112*x21)

        # Compute H Jacobian
        cdef double dHdr = M * M * dHeffdr / (nu*H)
        cdef double dHdphi = M * M * dHeffdphi / (nu*H)
        cdef double dHdpr = M * M * dHeffdpr / (nu*H)
        cdef double dHdpphi = M * M * dHeffdpphi / (nu*H)

        return dHdr, dHdphi, dHdpr, dHdpphi

    cpdef hessian(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL1,double chiL2):

        """
        Evaluate the Hessian of the Hamiltonian.

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1_v (double[:]): Dimensionless spin vector of the primary.
          chi2_v (double[:]): Dimensionless spin vector of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.
          chi_1 (double): Projection of chi1_v onto the Newtonian orbital angular momentum unit vector (lN).
          chi_2 (double): Projection of chi2_v onto the Newtonian orbital angular momentum unit vector (lN).
          chiL1 (double): Projection of chi1_v onto the orbital angular momentum unit vector (l).
          chiL2 (double): Projection of chi2_v onto the orbital angular momentum unit vector (l).

        Returns:
           (np.array)  d2Hdr2, d2Hdrdphi, d2Hdrdpr, d2Hdrdpphi, d2Hdrdphi, d2Hdphi2, d2Hdphidpr, d2Hdphidpphi, d2Hdrdpr, d2Hdphidpr, d2Hdpr2, d2Hdprdpphi, d2Hdrdpphi, d2Hdphidpphi, d2Hdprdpphi, d2Hdpphi2

        """


        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double chix1 = chi1_v[0]
        cdef double chiy1 = chi1_v[1]
        cdef double chiz1 = chi1_v[2]

        cdef double chix2 = chi2_v[0]
        cdef double chiy2 = chi2_v[1]
        cdef double chiz2 = chi2_v[2]

        cdef double pphi = L

        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs['a6']
        cdef double dSO = c_coeffs['dSO']


        # Extra quantities used in the Jacobian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double delta = self.EOBpars.p_params.delta
        cdef double x0 = r**3
        cdef double x1 = r + 2.0
        cdef double x2 = X_1*(2.0*chix1*chix2 + 2.0*chiy1*chiy2 + 2.0*chiz1*chiz2)
        cdef double x3 = X_2*x2
        cdef double x4 = X_1**2
        cdef double x5 = chix1**2 + chiy1**2 + chiz1**2
        cdef double x6 = x4*x5
        cdef double x7 = X_2**2*(chix2**2 + chiy2**2 + chiz2**2)
        cdef double x8 = x6 + x7
        cdef double x9 = x3 + x8
        cdef double x10 = x1*x9
        cdef double x11 = x0 + x10
        cdef double x12 = x11**(-1)
        cdef double x13 = r**4
        cdef double x14 = x13**(-1)
        cdef double x15 = 3.0*nu
        cdef double x16 = X_1*chiL1
        cdef double x17 = X_2*chiL2
        cdef double x18 = x16 + x17
        cdef double x19 = pphi*x18
        cdef double x20 = r**2
        cdef double x21 = x20**(-1)
        cdef double x22 = 0.71875*nu - 0.09375
        cdef double x23 = -1.40625*nu - 0.46875
        cdef double x24 = x0**(-1)
        cdef double x25 = pphi**2
        cdef double x26 = x24*x25
        cdef double x27 = 2.0*x26
        cdef double x28 = nu**2
        cdef double x29 = -2.0859375*nu - 2.07161458333333*x28 + 0.23046875
        cdef double x30 = 3.0*x14
        cdef double x31 = x25*x30
        cdef double x32 = 0.5859375*nu + 1.34765625*x28 + 0.41015625
        cdef double x33 = pphi**4
        cdef double x34 = r**5
        cdef double x35 = x34**(-1)
        cdef double x36 = 4.0*x35
        cdef double x37 = x33*x36
        cdef double x38 = 0.34375*nu + 0.09375
        cdef double x39 = 0.46875 - 0.28125*nu
        cdef double x40 = -0.2734375*nu - 0.798177083333333*x28 - 0.23046875
        cdef double x41 = -0.3515625*nu + 0.29296875*x28 - 0.41015625
        cdef double x42 = x16 - x17
        cdef double x43 = pphi*x42
        cdef double x44 = delta*x43
        cdef double x45 = chi_1*X_1
        cdef double x46 = chi_2*X_2
        cdef double x47 = x45 + x46
        cdef double x48 = x47**2
        cdef double x49 = -x48 + x9
        cdef double x50 = delta*x25
        cdef double x51 = delta*x9
        cdef double x52 = x49*(0.416666666666667 - 0.833333333333333*X_2)
        cdef double x53 = x45 - x46
        cdef double x54 = x4*x5 - x47*x53 - x7
        cdef double x55 = -0.125*x3 + 0.125*x48 + 0.5*x54*(0.5 - X_2) - 0.125*x6 - 0.125*x7
        cdef double x56 = x6 - x7
        cdef double x57 = delta*x56
        cdef double x58 = delta*x54
        cdef double x59 = nu*dSO*x24
        cdef double x60 = r**(-1)
        cdef double x61 = x21*x25
        cdef double x62 = x14*x33
        cdef double x63 = x18*(x21*(-5.53125*nu + 0.567708333333333*x28 - 0.078125) + x22*x60 + x23*x61 + x26*x29 + x32*x62 + 1.75)
        cdef double x64 = delta*(x21*(-0.03125*nu + 0.536458333333333*x28 + 0.078125) + x26*x40 + x38*x60 + x39*x61 + x41*x62 + 0.25)
        cdef double x65 = x24*x49
        cdef double x66 = x50*x65
        cdef double x67 = x42*(x21*(0.0416666666666667*x51 + 0.5*x52) - 0.125*x66)
        cdef double x68 = x26*x55
        cdef double x69 = x18*(x21*(0.208333333333333*delta*x56 - 0.625*x3 + 0.375*x48 - 0.833333333333333*x58 - 0.625*x6 - 0.625*x7) + x68)
        cdef double x70 = prst**4
        cdef double x71 = r**(-4.5)
        cdef double x72 = nu*x71
        cdef double x73 = prst**6
        cdef double x74 = r**(-3.5)
        cdef double x75 = nu*x74
        cdef double x76 = r**(-2.5)
        cdef double x77 = prst**8
        cdef double x78 = nu*x77
        cdef double x79 = nu*x60
        cdef double x80 = 0.121954868780449*x77
        cdef double x81 = 8.0*nu - 6.0*x28
        cdef double x82 = x21*x81
        cdef double x83 = nu**3
        cdef double x84 = 92.7110442849544*nu - 131.0*x28 + 10.0*x83
        cdef double x85 = x24*x70
        cdef double x86 = -2.78300763695006*nu - 5.4*x28 + 6.0*x83
        cdef double x87 = x21*x86
        cdef double x88 = nu**4
        cdef double x89 = -33.9782122170436*nu - 89.5298327361234*x28 + 188.0*x83 - 14.0*x88
        cdef double x90 = x24*x73
        cdef double x91 = 1.38977750996128*nu + 3.33842023648322*x28 + 3.42857142857143*x83 - 6.0*x88
        cdef double x92 = x77*x91
        cdef double x93 = log(r)
        cdef double x94 = nu*(452.542166996693 - 51.6952380952381*x93) + x28*(118.4*x93 - 1796.13660498019) + 602.318540416564*x83
        cdef double x95 = x14*x70
        cdef double x96 = r*x10 + x13
        cdef double x97 = x96**(-1)
        cdef double x98 = x18**2
        cdef double x99 = x97*x98
        cdef double x100 = x25*x60*x99
        cdef double x101 = 0.5*x49
        cdef double x102 = 0.5*x54
        cdef double x103 = -x3 + x8
        cdef double x104 = x103 - x53**2
        cdef double x105 = 0.5*x104
        cdef double x106 = x58*(98.0*nu + 43.0)
        cdef double x107 = 264.0*x28
        cdef double x108 = -1610.0*nu + x107 + 375.0
        cdef double x109 = 0.00260416666666667*x49
        cdef double x110 = -1090.0*nu + 152.0*x28 + 219.0
        cdef double x111 = 0.0078125*x104
        cdef double x112 = -x101*x21 + x14*(-0.0625*x106 + x108*x109 + x110*x111) + x24*(x101*(-1.75*nu - 0.9375) + x102*(0.75 - 1.5*X_2) + x105*(0.1875 - 0.75*nu)) + 1.0
        cdef double x113 = r**(-13)
        cdef double x114 = x20 + x9
        cdef double x115 = x114**4
        cdef double x116 = x21*x9
        cdef double x117 = 756.0*nu
        cdef double x118 = 336.0*r + x117 + 407.0
        cdef double x119 = 2048.0*nu*x118*x93 + 28.0*nu*(1920.0*a6 + 733955.307463037) - 7.0*r*(938918.400156317*nu - 185763.092693281*x28 - 245760.0) - 5416406.59541186*x28 - 3440640.0
        cdef double x120 = x93**2
        cdef double x121 = x120*x28
        cdef double x122 = x83*(-163683964.822551*r - 17833256.898555*x20 - 1188987459.03162)
        cdef double x123 = x28*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 122635399361.987*r - 6178501407.53758*x0 - 3089250703.76879*x13 + 2064783811.32587*x20 + 1426660551.8844*x34 + 276057889687.011)
        cdef double x124 = 588.0*nu + 1079.0
        cdef double x125 = x117 + 1079.0
        cdef double x126 = x0*x125
        cdef double x127 = -38842241.4769507*nu + 240.0*r*(-7466.27061066206*nu - 3024.0*x28 + 17264.0) + 1920.0*x124*x20 + 480.0*x125*x13 + 960.0*x126 - 1882456.23663972*x28 + 161280.0*x34 + 13447680.0
        cdef double x128 = nu*x93
        cdef double x129 = x127*x128
        cdef double x130 = 8.0*r
        cdef double x131 = 2.0*x0 + x130 + 4.0*x20 + 16.0
        cdef double x132 = 7680.0*a6
        cdef double x133 = 128.0*r
        cdef double x134 = 7704.0*r
        cdef double x135 = 148.04406601634*r
        cdef double x136 = 113485.217444961*r
        cdef double x137 = nu*(x132*(x13 + x131) + x133*(13218.7851094412*r + 4264.6962773603*x0 - 6852.34813868015*x13 + 8529.39255472061*x20 - 33722.4297811176) + x135*(1926.0*x0 + 349.0*x13 + x134 + 3852.0*x20 + 36400.0) + x136*(-x13 + x131))
        cdef double x138 = (67645734912.0*x121 + 1120.0*x122 + 7.0*x123 + 32768.0*x129 + 53760.0*x137 + 13212057600.0*x34 + 241555486248.807*x88)**(-1)
        cdef double x139 = x13*x138
        cdef double x140 = x119*x139
        cdef double x141 = 0.000130208333333333*x116 + x140
        cdef double x142 = x141**(-4)
        cdef double x143 = r*x120
        cdef double x144 = -630116198.873299*nu - 197773496.793534*x28 + 5805304367.87913
        cdef double x145 = x144*x20
        cdef double x146 = -2675575.66847905*nu - 138240.0*x28 - 5278341.3229329
        cdef double x147 = x0*x146
        cdef double x148 = nu*(-2510664218.28128*nu - 42636451.6032331*x28 + 14515200.0*x83 + 1002013764.01019)
        cdef double x149 = 43393301259014.8*nu + 43133561885859.3*x28 + 5927865218923.02*x83 + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0
        cdef double x150 = r*x149
        cdef double x151 = 14700.0*nu + 42911.0
        cdef double x152 = 283115520.0*x151
        cdef double x153 = -1698693120.0*nu*(11592.0*nu + 69847.0) + 49152.0*r*(409207698.136075*nu + 102574080.0*x28 - 2119671837.36038) + 879923036160.0*x0 + x152*x20
        cdef double x154 = x153*x93
        cdef double x155 = (x143 - 1.59227685093395e-9*x145 - 1.67189069348064e-7*x147 + 9.55366110560367e-9*x148 + 1.72773095804465e-13*x150 + 1.72773095804465e-13*x154)**2
        cdef double x156 = nu*r
        cdef double x157 = nu*x20
        cdef double x158 = r*x28
        cdef double x159 = r*x151
        cdef double x160 = x20*x28
        cdef double x161 = 5041721180160.0*x28 - 104186110149937.0
        cdef double x162 = -25392914995744.3*nu - r*x152 - x161 - 879923036160.0*x20
        cdef double x163 = x162*x93
        cdef double x164 = -0.0438084424460039*nu - 0.143521050466841*r + 0.0210425293255724*x120 + 0.0385795738434214*x156 + 0.00941289164152486*x157 + 0.00662650629087394*x158 - 1.18824456940711e-6*x159 + 0.000486339502879429*x160 - 3.63558293513537e-15*x163 + 0.0185696317637669*x20 + 0.291062041428379*x28 + 0.0244692826489756*x83 + 1
        cdef double x165 = x164**(-2)
        cdef double x166 = 4.0*x28
        cdef double x167 = 0.3125*delta*x56*(18.0*nu - 1.0) - 0.03125*(-33.0*nu + 32.0*x28 - 5.0)*(5.0*x3 + 5.0*x6 + 5.0*x7) - 0.03125*(-5.0*nu + x166 + 1.0)*(-15.0*x3 + 15.0*x6 + 15.0*x7)
        cdef double x168 = x115*x142*x155*x165*x167*x70
        cdef double x169 = x114**2
        cdef double x170 = prst**2
        cdef double x171 = x141**(-2)
        cdef double x172 = 1822680546449.21*x28
        cdef double x173 = 5787938193408.0*x120
        cdef double x174 = -12049908701745.2*nu + r*x172 - 39476764256925.6*r + 10611661054566.2*x156 + 2589101062873.81*x157 - 326837426.241486*x159 + 133772083200.0*x160 - x163 + x173 + 5107745331375.71*x20 + 80059249540278.2*x28 + 6730497718123.02*x83 + 275059053208689.0
        cdef double x175 = x174**(-1)
        cdef double x176 = 4.0*nu
        cdef double x177 = 3.0*x3
        cdef double x178 = 3.0*x6 + 3.0*x7
        cdef double x179 = -x177 + x178
        cdef double x180 = 0.0625*x179
        cdef double x181 = x57*(26.0*nu + 449.0)
        cdef double x182 = x9*(-1171.0*nu - 861.0)
        cdef double x183 = x103*(115.0*nu + x166 - 37.0)
        cdef double x184 = 5787938193408.0*x143 - 9216.0*x145 - 967680.0*x147 + 55296.0*x148 + x150 + x154
        cdef double x185 = x184**(-1)
        cdef double x186 = x174*x185
        cdef double x187 = x186*x34
        cdef double x188 = x119*x138
        cdef double x189 = 7680.0*x188
        cdef double x190 = x116 + x14*(0.03125*x181 + 0.015625*x182 + 0.015625*x183) + x187*x189 + x24*(x180*(x176 - 1.0) - 2.625*x57 + x9*(x15 + 2.8125))
        cdef double x191 = x169*x170*x171*x175*x184*x190
        cdef double x192 = -x1*x100 + x112*x61 + 1.27277314139085e-19*x113*x168 + 1.69542100694444e-8*x191*x35 + x21*x92 + 147.443752990146*x70*x72 + x70*x82 - 11.3175085791863*x73*x75 + x73*x87 + 1.48275342024365*x76*x78 + x79*x80 + x84*x85 + x89*x90 + x94*x95 + 1.0
        cdef double x193 = 2.0*x60
        cdef double x194 = x116*(x193 + 1.0) + 1.0
        cdef double x195 = x194**(-1)
        cdef double x196 = x103*(x176 + 1.0)
        cdef double x197 = x57*(2.0*nu - 3.0)
        cdef double x198 = 7.0*nu
        cdef double x199 = 0.390625*x3 + 0.390625*x6 + 0.390625*x7
        cdef double x200 = x49*(x198 - 31.0)
        cdef double x201 = x104*(-x176 - 3.0)
        cdef double x202 = x58*(68.0*nu - 1.0)
        cdef double x203 = x116 + x14*(0.125*x200 + 0.0625*x201 + 2.0625*x58) + x14*(0.125*x196 + 1.125*x3 - 1.25*x57 + 1.125*x6 + 1.125*x7) + 7680.0*x140 + x35*(x109*(2870.0*nu - x107 + 561.0) + x111*(1166.0*nu - 328.0*x28 - 171.0) + 0.125*x202) + x35*(0.015625*x179*(-27.0*nu + 28.0*x28 - 3.0) - 1.21875*x197 - x199*(x198 + 9.0)) + x65
        cdef double x204 = x195*x203
        cdef double x205 = (x192*x204)**(-0.5)
        cdef double x206 = -2.0*x3 - 2.0*x6 - 2.0*x7
        cdef double x207 = 4.0*x60 + 2.0
        cdef double x208 = 4.5*X_2
        cdef double x209 = r**(-6)
        cdef double x210 = -6572428.80109422*nu + 2048.0*x118*x79 + 688128.0*x128 + 1300341.64885296*x28 + 1720320.0
        cdef double x211 = x28*x60
        cdef double x212 = 4.0*x0
        cdef double x213 = x130 + 6.0*x20 + 8.0
        cdef double x214 = 1.31621673590926e-19*x119*(53760.0*nu*(3740417.71815805*r + 1057984.42953951*x0 - 938918.400156317*x13 + x132*(x212 + x213) + x133*(17058.7851094412*r - 27409.3925547206*x0 + 12794.0888320809*x20 + 13218.7851094412) + x135*(1396.0*x0 + x134 + 5778.0*x20 + 7704.0) + x136*(-x212 + x213) + 2115968.85907902*x20 + 2888096.47013111) + 32768.0*x127*x79 + 32768.0*x128*(-1791904.9465589*nu + 3840.0*r*x124 + 2880.0*x125*x20 + 1920.0*x126 + 806400.0*x13 - 725760.0*x28 + 4143360.0) + 66060288000.0*x13 + 135291469824.0*x211*x93 + 7.0*x28*(-117964800.0*a6 + 4129567622.65173*r - 12357002815.0752*x0 + 7133302759.42198*x13 - 18535504222.6128*x20 + 122635399361.987) + 7.0*x83*(-5706642207.53758*r - 26189434371.6082))/(0.28004222119933*x121 + 4.63661586574928e-9*x122 + 2.8978849160933e-11*x123 + 1.35654132757922e-7*x129 + 2.22557561555966e-7*x137 + 0.0546957463279941*x34 + x88)**2
        cdef double x215 = x13*x214
        cdef double x216 = 2.0*x24
        cdef double x217 = 11575876386816.0*x93
        cdef double x218 = 5807150888816.34*nu + 10215490662751.4*r + 5178202125747.62*x156 + 267544166400.0*x158 - x162*x60 + x172 + x217*x60 + x93*(4161798144000.0*nu + 1759846072320.0*r + 12148770078720.0) - 53501685054374.1
        cdef double x219 = x113*x115*x155*x167*x70
        cdef double x220 = -18432.0*r*x144 - 2903040.0*x146*x20 + x149 + x153*x60 + x173 + x217 + x93*(20113376778784.3*nu + 566231040.0*x159 + x161 + 2639769108480.0*x20)
        cdef double x221 = x170*x190
        cdef double x222 = x206*x24
        cdef double x223 = 7680.0*x210
        cdef double x224 = 30720.0*x0*x188 + x139*x223 - x215 + x222
        cdef double x225 = x184*x35
        cdef double x226 = prst**3
        cdef double x227 = prst**5
        cdef double x228 = prst**7
        cdef double x229 = 4.0*x226
        cdef double x230 = 6.0*x227
        cdef double x231 = x203*x205/(x116*x207 + 2.0)
        cdef double x232 = 2.0*pphi*x21
        cdef double x233 = pphi*x216
        cdef double x234 = 4.0*pphi**3*x14
        cdef double y0 = r**3
        cdef double y1 = r + 2.0
        cdef double y2 = chix1*chix2
        cdef double y3 = chiy1*chiy2
        cdef double y4 = chiz1*chiz2
        cdef double y5 = 2.0*y2 + 2.0*y3 + 2.0*y4
        cdef double y6 = X_1*X_2
        cdef double y7 = y5*y6
        cdef double y8 = X_1**2
        cdef double y9 = chix1**2
        cdef double y10 = chiy1**2
        cdef double y11 = chiz1**2
        cdef double y12 = y10 + y11 + y9
        cdef double y13 = y12*y8
        cdef double y14 = X_2**2
        cdef double y15 = chix2**2
        cdef double y16 = chiy2**2
        cdef double y17 = chiz2**2
        cdef double y18 = y14*(y15 + y16 + y17)
        cdef double y19 = y13 + y18
        cdef double y20 = y19 + y7
        cdef double y21 = y1*y20
        cdef double y22 = y0 + y21
        cdef double y23 = y22**(-1)
        cdef double y24 = r**5
        cdef double y25 = y24**(-1)
        cdef double y26 = 12.0*nu
        cdef double y27 = X_1*chiL1
        cdef double y28 = X_2*chiL2
        cdef double y29 = y27 + y28
        cdef double y30 = pphi*y29
        cdef double y31 = dSO*y30
        cdef double y32 = y0**(-1)
        cdef double y33 = -1.40625*nu - 0.46875
        cdef double y34 = pphi**2
        cdef double y35 = r**4
        cdef double y36 = y35**(-1)
        cdef double y37 = 6.0*y36
        cdef double y38 = y34*y37
        cdef double y39 = nu**2
        cdef double y40 = -2.0859375*nu - 2.07161458333333*y39 + 0.23046875
        cdef double y41 = 12.0*y25
        cdef double y42 = y34*y41
        cdef double y43 = r**(-6)
        cdef double y44 = 20.0*y43
        cdef double y45 = pphi**4
        cdef double y46 = 0.5859375*nu + 1.34765625*y39 + 0.41015625
        cdef double y47 = y45*y46
        cdef double y48 = 0.46875 - 0.28125*nu
        cdef double y49 = -0.2734375*nu - 0.798177083333333*y39 - 0.23046875
        cdef double y50 = -0.3515625*nu + 0.29296875*y39 - 0.41015625
        cdef double y51 = y45*y50
        cdef double y52 = y27 - y28
        cdef double y53 = pphi*y52
        cdef double y54 = delta*y53
        cdef double y55 = chi_1*X_1
        cdef double y56 = chi_2*X_2
        cdef double y57 = y55 + y56
        cdef double y58 = y57**2
        cdef double y59 = y20 - y58
        cdef double y60 = delta*y34
        cdef double y61 = y59*y60
        cdef double y62 = delta*y20
        cdef double y63 = y55 - y56
        cdef double y64 = y12*y8 - y18 - y57*y63
        cdef double y65 = -0.125*y13 - 0.125*y18 + 0.125*y58 + 0.5*y64*(0.5 - X_2) - 0.125*y7
        cdef double y66 = y34*y65
        cdef double y67 = y13 - y18
        cdef double y68 = delta*y67
        cdef double y69 = -1.25*y68
        cdef double y70 = delta*y64
        cdef double y71 = r**2
        cdef double y72 = 6.0*y71
        cdef double y73 = 2.0*y13 + 2.0*y18 + 2.0*y7
        cdef double y74 = y22**(-2)
        cdef double y75 = 3.0*nu
        cdef double y76 = y36*y75
        cdef double y77 = y71**(-1)
        cdef double y78 = 0.71875*nu - 0.09375
        cdef double y79 = y32*y34
        cdef double y80 = 2.0*y79
        cdef double y81 = 3.0*y36
        cdef double y82 = y34*y81
        cdef double y83 = 4.0*y25
        cdef double y84 = y29*(-y32*(-11.0625*nu + 1.13541666666667*y39 - 0.15625) - y33*y80 - y40*y82 - y47*y83 - y77*y78)
        cdef double y85 = 0.34375*nu + 0.09375
        cdef double y86 = -y32*(-0.0625*nu + 1.07291666666667*y39 + 0.15625) - y48*y80 - y49*y82 - y51*y83 - y77*y85
        cdef double y87 = y59*(0.416666666666667 - 0.833333333333333*X_2)
        cdef double y88 = y52*(-y32*(0.0833333333333333*y62 + y87) + 0.375*y36*y61)
        cdef double y89 = y29*(y32*(1.25*y13 + 1.25*y18 - 0.75*y58 - 0.416666666666667*y68 + 1.25*y7 + 1.66666666666667*y70) - y66*y81)
        cdef double y90 = nu*y32
        cdef double y91 = r**(-1)
        cdef double y92 = y34*y77
        cdef double y93 = y36*y45
        cdef double y94 = y29*(y33*y92 + y40*y79 + y46*y93 + y77*(-5.53125*nu + 0.567708333333333*y39 - 0.078125) + y78*y91 + 1.75)
        cdef double y95 = y48*y92 + y49*y79 + y50*y93 + y77*(-0.03125*nu + 0.536458333333333*y39 + 0.078125) + y85*y91 + 0.25
        cdef double y96 = y32*y59
        cdef double y97 = y52*(-0.125*y60*y96 + y77*(0.0416666666666667*y62 + 0.5*y87))
        cdef double y98 = y65*y79
        cdef double y99 = y29*(y77*(0.208333333333333*delta*y67 - 0.625*y13 - 0.625*y18 + 0.375*y58 - 0.625*y7 - 0.833333333333333*y70) + y98)
        cdef double y100 = prst**4
        cdef double y101 = r**(-4.5)
        cdef double y102 = nu*y101
        cdef double y103 = r**(-3.5)
        cdef double y104 = prst**6
        cdef double y105 = nu*y104
        cdef double y106 = r**(-2.5)
        cdef double y107 = prst**8
        cdef double y108 = nu*y107
        cdef double y109 = nu*y91
        cdef double y110 = 0.121954868780449*y107
        cdef double y111 = 8.0*nu - 6.0*y39
        cdef double y112 = y100*y111
        cdef double y113 = nu**3
        cdef double y114 = 92.7110442849544*nu + 10.0*y113 - 131.0*y39
        cdef double y115 = y100*y114
        cdef double y116 = -2.78300763695006*nu + 6.0*y113 - 5.4*y39
        cdef double y117 = y104*y116
        cdef double y118 = nu**4
        cdef double y119 = -33.9782122170436*nu + 188.0*y113 - 14.0*y118 - 89.5298327361234*y39
        cdef double y120 = y104*y119
        cdef double y121 = 1.38977750996128*nu + 3.42857142857143*y113 - 6.0*y118 + 3.33842023648322*y39
        cdef double y122 = y107*y121
        cdef double y123 = log(r)
        cdef double y124 = nu*(452.542166996693 - 51.6952380952381*y123) + 602.318540416564*y113 + y39*(118.4*y123 - 1796.13660498019)
        cdef double y125 = y100*y36
        cdef double y126 = r*y21 + y35
        cdef double y127 = y126**(-1)
        cdef double y128 = y29**2
        cdef double y129 = y127*y128
        cdef double y130 = y129*y91
        cdef double y131 = y130*y34
        cdef double y132 = y59*y77
        cdef double y133 = y64*(0.75 - 1.5*X_2)
        cdef double y134 = y59*(-1.75*nu - 0.9375)
        cdef double y135 = y19 - y7
        cdef double y136 = y135 - y63**2
        cdef double y137 = y136*(0.1875 - 0.75*nu)
        cdef double y138 = y70*(98.0*nu + 43.0)
        cdef double y139 = 264.0*y39
        cdef double y140 = -1610.0*nu + y139 + 375.0
        cdef double y141 = 0.00260416666666667*y59
        cdef double y142 = -1090.0*nu + 152.0*y39 + 219.0
        cdef double y143 = 0.0078125*y136
        cdef double y144 = -0.5*y132 + y32*(0.5*y133 + 0.5*y134 + 0.5*y137) + y36*(-0.0625*y138 + y140*y141 + y142*y143) + 1.0
        cdef double y145 = y20*y77
        cdef double y146 = y123**2
        cdef double y147 = y146*y39
        cdef double y148 = y113*(-163683964.822551*r - 17833256.898555*y71 - 1188987459.03162)
        cdef double y149 = y39*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 122635399361.987*r - 6178501407.53758*y0 + 1426660551.8844*y24 - 3089250703.76879*y35 + 2064783811.32587*y71 + 276057889687.011)
        cdef double y150 = 588.0*nu + 1079.0
        cdef double y151 = 756.0*nu
        cdef double y152 = y151 + 1079.0
        cdef double y153 = y0*y152
        cdef double y154 = -38842241.4769507*nu + 240.0*r*(-7466.27061066206*nu - 3024.0*y39 + 17264.0) + 1920.0*y150*y71 + 480.0*y152*y35 + 960.0*y153 + 161280.0*y24 - 1882456.23663972*y39 + 13447680.0
        cdef double y155 = nu*y123
        cdef double y156 = y154*y155
        cdef double y157 = 8.0*r
        cdef double y158 = 2.0*y0 + y157 + 4.0*y71 + 16.0
        cdef double y159 = 7680.0*a6
        cdef double y160 = 128.0*r
        cdef double y161 = 7704.0*r
        cdef double y162 = 148.04406601634*r
        cdef double y163 = 113485.217444961*r
        cdef double y164 = nu*(y159*(y158 + y35) + y160*(13218.7851094412*r + 4264.6962773603*y0 - 6852.34813868015*y35 + 8529.39255472061*y71 - 33722.4297811176) + y162*(1926.0*y0 + y161 + 349.0*y35 + 3852.0*y71 + 36400.0) + y163*(y158 - y35))
        cdef double y165 = (241555486248.807*y118 + 67645734912.0*y147 + 1120.0*y148 + 7.0*y149 + 32768.0*y156 + 53760.0*y164 + 13212057600.0*y24)**(-1)
        cdef double y166 = 336.0*r + y151 + 407.0
        cdef double y167 = 2048.0*nu*y123*y166 + 28.0*nu*(1920.0*a6 + 733955.307463037) - 7.0*r*(938918.400156317*nu - 185763.092693281*y39 - 245760.0) - 5416406.59541186*y39 - 3440640.0
        cdef double y168 = y165*y167
        cdef double y169 = y168*y35
        cdef double y170 = 0.000130208333333333*y145 + y169
        cdef double y171 = y170**(-4)
        cdef double y172 = r*y146
        cdef double y173 = -630116198.873299*nu - 197773496.793534*y39 + 5805304367.87913
        cdef double y174 = y173*y71
        cdef double y175 = -2675575.66847905*nu - 138240.0*y39 - 5278341.3229329
        cdef double y176 = y0*y175
        cdef double y177 = nu*(-2510664218.28128*nu + 14515200.0*y113 - 42636451.6032331*y39 + 1002013764.01019)
        cdef double y178 = (1 - 0.496948781616935*nu)**2
        cdef double y179 = 43393301259014.8*nu + 5927865218923.02*y113 + 86618264430493.3*y178 + 43133561885859.3*y39 + 188440788778196.0
        cdef double y180 = r*y179
        cdef double y181 = 14700.0*nu + 42911.0
        cdef double y182 = 283115520.0*y181
        cdef double y183 = -1698693120.0*nu*(11592.0*nu + 69847.0) + 49152.0*r*(409207698.136075*nu + 102574080.0*y39 - 2119671837.36038) + 879923036160.0*y0 + y182*y71
        cdef double y184 = y123*y183
        cdef double y185 = y172 - 1.59227685093395e-9*y174 - 1.67189069348064e-7*y176 + 9.55366110560367e-9*y177 + 1.72773095804465e-13*y180 + 1.72773095804465e-13*y184
        cdef double y186 = y185**2
        cdef double y187 = nu*r
        cdef double y188 = nu*y71
        cdef double y189 = r*y39
        cdef double y190 = r*y181
        cdef double y191 = y39*y71
        cdef double y192 = 5041721180160.0*y39 - 104186110149937.0
        cdef double y193 = -25392914995744.3*nu - r*y182 - y192 - 879923036160.0*y71
        cdef double y194 = y123*y193
        cdef double y195 = -0.0438084424460039*nu - 0.143521050466841*r + 0.0244692826489756*y113 + 0.0210425293255724*y146 + 0.0385795738434214*y187 + 0.00941289164152486*y188 + 0.00662650629087394*y189 - 1.18824456940711e-6*y190 + 0.000486339502879429*y191 - 3.63558293513537e-15*y194 + 0.291062041428379*y39 + 0.0185696317637669*y71 + 1
        cdef double y196 = y195**(-2)
        cdef double y197 = 4.0*y39
        cdef double y198 = 0.3125*delta*y67*(18.0*nu - 1.0) - 0.03125*(-33.0*nu + 32.0*y39 - 5.0)*(5.0*y13 + 5.0*y18 + 5.0*y7) - 0.03125*(-5.0*nu + y197 + 1.0)*(15.0*y13 + 15.0*y18 - 15.0*y7)
        cdef double y199 = y171*y186*y196*y198
        cdef double y200 = y100*y199
        cdef double y201 = r**(-13)
        cdef double y202 = y20 + y71
        cdef double y203 = y202**4
        cdef double y204 = y201*y203
        cdef double y205 = prst**2
        cdef double y206 = y170**(-2)
        cdef double y207 = 1822680546449.21*y39
        cdef double y208 = 5787938193408.0*y146
        cdef double y209 = -12049908701745.2*nu + r*y207 - 39476764256925.6*r + 6730497718123.02*y113 + 10611661054566.2*y187 + 2589101062873.81*y188 - 326837426.241486*y190 + 133772083200.0*y191 - y194 + y208 + 80059249540278.2*y39 + 5107745331375.71*y71 + 275059053208689.0
        cdef double y210 = y209**(-1)
        cdef double y211 = y205*y206*y210
        cdef double y212 = 4.0*nu
        cdef double y213 = 3.0*y7
        cdef double y214 = 3.0*y13 + 3.0*y18
        cdef double y215 = -y213 + y214
        cdef double y216 = 0.0625*y215
        cdef double y217 = y68*(26.0*nu + 449.0)
        cdef double y218 = y20*(-1171.0*nu - 861.0)
        cdef double y219 = y135*(115.0*nu + y197 - 37.0)
        cdef double y220 = 5787938193408.0*y172 - 9216.0*y174 - 967680.0*y176 + 55296.0*y177 + y180 + y184
        cdef double y221 = y220**(-1)
        cdef double y222 = y221*y24
        cdef double y223 = y209*y222
        cdef double y224 = 7680.0*y168
        cdef double y225 = y145 + y223*y224 + y32*(y20*(y75 + 2.8125) + y216*(y212 - 1.0) - 2.625*y68) + y36*(0.03125*y217 + 0.015625*y218 + 0.015625*y219)
        cdef double y226 = y220*y225
        cdef double y227 = y211*y226
        cdef double y228 = y202**2
        cdef double y229 = y228*y25
        cdef double y230 = 1.69542100694444e-8*y229
        cdef double y231 = -y1*y131 + 147.443752990146*y100*y102 - 11.3175085791863*y103*y105 + 1.48275342024365*y106*y108 + y109*y110 + y112*y77 + y115*y32 + y117*y77 + y120*y32 + y122*y77 + y124*y125 + y144*y92 + 1.27277314139085e-19*y200*y204 + y227*y230 + 1.0
        cdef double y232 = 2.0*y91
        cdef double y233 = y145*(y232 + 1.0) + 1.0
        cdef double y234 = y233**(-1)
        cdef double y235 = y135*(y212 + 1.0)
        cdef double y236 = y36*(1.125*y13 + 1.125*y18 + 0.125*y235 + y69 + 1.125*y7)
        cdef double y237 = y68*(2.0*nu - 3.0)
        cdef double y238 = 7.0*nu
        cdef double y239 = 25.0*y13 + 25.0*y18 + 25.0*y7
        cdef double y240 = 0.015625*y239
        cdef double y241 = y25*(0.015625*y215*(-27.0*nu + 28.0*y39 - 3.0) - 1.21875*y237 - y240*(y238 + 9.0))
        cdef double y242 = y59*(y238 - 31.0)
        cdef double y243 = y136*(-y212 - 3.0)
        cdef double y244 = y36*(0.125*y242 + 0.0625*y243 + 2.0625*y70)
        cdef double y245 = y70*(68.0*nu - 1.0)
        cdef double y246 = y25*(y141*(2870.0*nu - y139 + 561.0) + y143*(1166.0*nu - 328.0*y39 - 171.0) + 0.125*y245)
        cdef double y247 = y145 + 7680.0*y169 + y236 + y241 + y244 + y246 + y96
        cdef double y248 = y234*y247
        cdef double y249 = y231*y248
        cdef double y250 = y249**(-1.5)
        cdef double y251 = -y73
        cdef double y252 = 4.0*y91 + 2.0
        cdef double y253 = y20*y32
        cdef double y254 = y251*y36 - y252*y253
        cdef double y255 = y233**(-2)
        cdef double y256 = y247*y255
        cdef double y257 = y231*y256
        cdef double y258 = y213 + y214 - 3.0*y58
        cdef double y259 = 4.5*X_2
        cdef double y260 = X_1*y5
        cdef double y261 = 35.0*nu
        cdef double y262 = y261 + 45.0
        cdef double y263 = -135.0*nu + 140.0*y39 - 15.0
        cdef double y264 = 1320.0*y39
        cdef double y265 = 14350.0*nu - y264 + 2805.0
        cdef double y266 = 5830.0*nu - 1640.0*y39 - 855.0
        cdef double y267 = 2048.0*y166
        cdef double y268 = -6572428.80109422*nu + y109*y267 + 688128.0*y155 + 1300341.64885296*y39 + 1720320.0
        cdef double y269 = y167*y35
        cdef double y270 = y118 + 0.28004222119933*y147 + 4.63661586574928e-9*y148 + 2.8978849160933e-11*y149 + 1.35654132757922e-7*y156 + 2.22557561555966e-7*y164 + 0.0546957463279941*y24
        cdef double y271 = y270**(-2)
        cdef double y272 = y113*(-5706642207.53758*r - 26189434371.6082)
        cdef double y273 = y123*y91
        cdef double y274 = y273*y39
        cdef double y275 = y39*(-117964800.0*a6 + 4129567622.65173*r - 12357002815.0752*y0 + 7133302759.42198*y35 - 18535504222.6128*y71 + 122635399361.987)
        cdef double y276 = y152*y71
        cdef double y277 = -1791904.9465589*nu + 3840.0*r*y150 + 1920.0*y153 + 2880.0*y276 + 806400.0*y35 - 725760.0*y39 + 4143360.0
        cdef double y278 = y155*y277
        cdef double y279 = y109*y154
        cdef double y280 = 4.0*y0
        cdef double y281 = y157 + y72 + 8.0
        cdef double y282 = nu*(3740417.71815805*r + 1057984.42953951*y0 + y159*(y280 + y281) + y160*(17058.7851094412*r - 27409.3925547206*y0 + 12794.0888320809*y71 + 13218.7851094412) + y162*(1396.0*y0 + y161 + 5778.0*y71 + 7704.0) + y163*(-y280 + y281) - 938918.400156317*y35 + 2115968.85907902*y71 + 2888096.47013111)
        cdef double y283 = y271*(7.0*y272 + 135291469824.0*y274 + 7.0*y275 + 32768.0*y278 + 32768.0*y279 + 53760.0*y282 + 66060288000.0*y35)
        cdef double y284 = y269*y283
        cdef double y285 = 1.31621673590926e-19*y284
        cdef double y286 = 30720.0*y0*y165*y167 + 7680.0*y165*y268*y35 - y25*(0.5*y242 + 0.25*y243 + 8.25*y70) - y25*(4.5*y13 + 4.5*y18 + 0.5*y235 + y259*y260 - 5.0*y68) + y251*y32 - y258*y36 - y285 - y43*(y141*y265 + y143*y266 + 0.625*y245) - y43*(0.015625*y215*y263 - 6.09375*y237 - y240*y262)
        cdef double y287 = y231*y234
        cdef double y288 = r**(-5.5)
        cdef double y289 = nu*y100
        cdef double y290 = nu*y77
        cdef double y291 = 2.0*y32
        cdef double y292 = 118.4*y39
        cdef double y293 = -51.6952380952381*nu*y91 + y292*y91
        cdef double y294 = y100*y124
        cdef double y295 = y126**(-2)
        cdef double y296 = r*y20
        cdef double y297 = y21 + y280 + y296
        cdef double y298 = 0.5*y59
        cdef double y299 = 0.0104166666666667*y59
        cdef double y300 = 0.03125*y136
        cdef double y301 = -y25*(-0.25*y138 + y140*y299 + y142*y300) + y32*y59 - y36*(0.5*y136*(0.5625 - 2.25*nu) + y298*(-5.25*nu - 2.8125) + 0.5*y64*(2.25 - y259))
        cdef double y302 = r**(-14)
        cdef double y303 = y200*y203
        cdef double y304 = r**(-12)
        cdef double y305 = y202**3
        cdef double y306 = 5178202125747.62*nu
        cdef double y307 = 267544166400.0*y39
        cdef double y308 = 11575876386816.0*y91
        cdef double y309 = y123*y308
        cdef double y310 = y123*(4161798144000.0*nu + 1759846072320.0*r + 12148770078720.0)
        cdef double y311 = y193*y91
        cdef double y312 = 5807150888816.34*nu + r*y306 + r*y307 + 10215490662751.4*r + y207 + y309 + y310 - y311 - 53501685054374.1
        cdef double y313 = y195**(-3)
        cdef double y314 = y312*y313
        cdef double y315 = y100*y171*y198
        cdef double y316 = y186*y315
        cdef double y317 = y204*y316
        cdef double y318 = 9.25454462627843e-34*y317
        cdef double y319 = y183*y91
        cdef double y320 = y123*(20113376778784.3*nu + 566231040.0*y190 + y192 + 2639769108480.0*y71)
        cdef double y321 = r*y173
        cdef double y322 = y175*y71
        cdef double y323 = 11575876386816.0*y123 + y179 + y208 + y319 + y320 - 18432.0*y321 - 2903040.0*y322
        cdef double y324 = y227*y228
        cdef double y325 = y206*y226
        cdef double y326 = y205*y325
        cdef double y327 = y196*y326
        cdef double y328 = y312*y327
        cdef double y329 = 2.24091649004576e-37*y229
        cdef double y330 = y170**(-5)
        cdef double y331 = y251*y32
        cdef double y332 = y165*y35
        cdef double y333 = y268*y332
        cdef double y334 = y0*y168
        cdef double y335 = -y285 + y331 + 7680.0*y333 + 30720.0*y334
        cdef double y336 = y330*y335
        cdef double y337 = y196*y204
        cdef double y338 = y198*y337
        cdef double y339 = y100*y186
        cdef double y340 = y338*y339
        cdef double y341 = 6.62902677807736e-23*y340
        cdef double y342 = y205*y210*y226
        cdef double y343 = y170**(-3)
        cdef double y344 = y335*y343
        cdef double y345 = y342*y344
        cdef double y346 = y26 - 3.0
        cdef double y347 = y222*y224
        cdef double y348 = y165*y268
        cdef double y349 = y209*y221
        cdef double y350 = y186**(-1)
        cdef double y351 = y323*y350
        cdef double y352 = y209*y24
        cdef double y353 = y168*y352
        cdef double y354 = 2.29252167428035e-22*y353
        cdef double y355 = y167*y283
        cdef double y356 = 38400.0*y169*y349 + 7680.0*y223*y348 - 1.31621673590926e-19*y223*y355 - y25*(0.125*y217 + 0.0625*y218 + 0.0625*y219) + y312*y347 + y331 - y351*y354 - y36*(y20*(9.0*nu + 8.4375) + y216*y346 - 7.875*y68)
        cdef double y357 = 39.6112800271521*nu*y101*y104 + y1*y127*y128*y34*y77 + y1*y128*y295*y297*y34*y91 + 1.01821851311268e-18*y100*y171*y186*y196*y198*y304*y305 + 7.59859378406358e-45*y100*y171*y196*y198*y201*y203*y220*y323 + y100*y293*y36 - 3.70688355060912*y103*y108 - y110*y290 - y112*y291 - 3.0*y114*y125 - y117*y291 - y120*y81 - y122*y291 - y131 - y144*y80 + 6.78168402777778e-8*y202*y205*y206*y210*y220*y225*y36 + 1.69542100694444e-8*y205*y206*y210*y220*y228*y25*y356 + 1.69542100694444e-8*y205*y206*y210*y225*y228*y25*y323 - 4.41515887225116e-12*y229*y345 - 663.496888455656*y288*y289 - y294*y83 + y301*y34*y77 - 1.65460508380811e-18*y302*y303 - y314*y318 - 8.47710503472222e-8*y324*y43 - y328*y329 - y336*y341
        cdef double y358 = y248*y357 - y254*y257 + y286*y287
        cdef double y359 = 12.0*y13 + 12.0*y18 + 12.0*y7
        cdef double y360 = -4.0*y13 - 4.0*y18 - 4.0*y7
        cdef double y361 = 8.0*y91 + 4.0
        cdef double y362 = -y253*y361 + y36*y360
        cdef double y363 = r**(-7)
        cdef double y364 = y265*y59
        cdef double y365 = 0.015625*y136
        cdef double y366 = 0.03125*y239
        cdef double y367 = 20.0*nu
        cdef double y368 = 6.0*y13 + 6.0*y18 + 6.0*y7
        cdef double y369 = y36*y368
        cdef double y370 = 10569646080.0*y109 - 7680.0*y267*y290
        cdef double y371 = 2.63243347181853e-19*y283
        cdef double y372 = y268*y371
        cdef double y373 = 1.99471718230171e-8*(5.17401430341932e-11*y272 + y274 + 5.17401430341932e-11*y275 + 2.42203000992063e-7*y278 + 2.42203000992063e-7*y279 + 3.97364298502604e-7*y282 + 0.48828125*y35)**2/y270**3
        cdef double y374 = 135291469824.0*y39*y77
        cdef double y375 = 12.0*y71
        cdef double y376 = 12.0*r + 8.0
        cdef double y377 = 1.31621673590926e-19*y271*(53760.0*nu*(8463875.43631609*r - 7511347.20125054*y0 + y159*(y375 + y376) + y160*(25588.1776641618*r - 82228.1776641618*y71 + 17058.7851094412) + y162*(11556.0*r + 4188.0*y71 + 7704.0) + y163*(-y375 + y376) + 6347906.57723707*y71 + 7480835.43631609) + 264241152000.0*y0 + 65536.0*y109*y277 - 39946495452.7631*y113 - y123*y374 - 32768.0*y154*y290 + 32768.0*y155*(2257920.0*nu + 5760.0*r*y152 + 3225600.0*y0 + 5760.0*y276 + 4143360.0) + y374 + 7.0*y39*(-37071008445.2255*r + 28533211037.6879*y0 - 37071008445.2255*y71 + 4129567622.65173))
        cdef double y378 = 61440.0*y0*y348 - 1.05297338872741e-18*y0*y355 + 92160.0*y168*y71 + y269*y373 - y269*y377 + y332*y370 - y35*y372 + y369
        cdef double y379 = 9.0*X_2
        cdef double y380 = y1*y129
        cdef double y381 = y128*y34
        cdef double y382 = y295*y381
        cdef double y383 = y1*y91
        cdef double y384 = y128*y295*y297
        cdef double y385 = (0.230275523363951*nu + 0.031457442188381*y113 + 0.0614297810037367*y123 + 0.0307148905018684*y146 + 0.459657725867658*y178 + 5.30670671930294e-15*y319 + 5.30670671930294e-15*y320 - 9.78132182501918e-11*y321 - 1.54055818744052e-8*y322 + 0.228897162687159*y39 + 1)**2
        cdef double y386 = y315*y337
        cdef double y387 = 11575876386816.0*y77
        cdef double y388 = 8323596288000.0*nu + 24297540157440.0
        cdef double y389 = -y123*y387 + 1759846072320.0*y123 + y193*y77 + y306 + y307 + y387 + y91*(3519692144640.0*r + y388) + 10215490662751.4
        cdef double y390 = y203*y302
        cdef double y391 = y314*y316
        cdef double y392 = y304*y305
        cdef double y393 = (0.108541457767442*nu + 0.190937736865098*r + 0.0967857763822762*y187 + 0.00500066803742898*y189 + 0.216364706551791*y273 + 1.86910000868887e-14*y310 - 1.86910000868887e-14*y311 + 0.0340677222520525*y39 - 1)**2
        cdef double y394 = 11614301777632.7*nu - 5806080.0*r*y175 + y123*(5279538216960.0*r + y388) - y183*y77 + y308 + y309 + 3645361092898.41*y39 + y91*(40226753557568.7*nu + 1132462080.0*y190 + 10083442360320.0*y39 + 5279538216960.0*y71 - 208372220299875.0) - 107003370108748.0
        cdef double y395 = y196*y390
        cdef double y396 = y220*y323
        cdef double y397 = y315*y396
        cdef double y398 = y196*y392
        cdef double y399 = y204*y314
        cdef double y400 = y211*y230
        cdef double y401 = y211*y323
        cdef double y402 = y225*y401
        cdef double y403 = y202*y36
        cdef double y404 = 1.35633680555556e-7*y403
        cdef double y405 = y228*y43
        cdef double y406 = 1.69542100694444e-7*y405
        cdef double y407 = y205*y229
        cdef double y408 = y225*y323*y407
        cdef double y409 = 4.48183298009152e-37*y196*y312
        cdef double y410 = y206*y409
        cdef double y411 = y198*y336*y339
        cdef double y412 = (-4.28455968720463e-24*y284 + 3.25520833333333e-5*y331 + 0.25*y333 + y334)**2
        cdef double y413 = y229*y342
        cdef double y414 = 8.83031774450231e-12*y344
        cdef double y415 = y210*y414
        cdef double y416 = y220*y356
        cdef double y417 = y211*y416
        cdef double y418 = 3.39084201388889e-8*y229
        cdef double y419 = y407*y416
        cdef double y420 = y222*y312
        cdef double y421 = 76800.0*y221
        cdef double y422 = 4.5850433485607e-22*y351
        cdef double y423 = y167*y223
        cdef double y424 = y249**(-0.5)
        cdef double y425 = (y145*y252 + 2.0)**(-1)
        cdef double y426 = 12.0*y205
        cdef double y427 = y111*y77
        cdef double y428 = y114*y32
        cdef double y429 = 30.0*y100
        cdef double y430 = y116*y77
        cdef double y431 = y119*y32
        cdef double y432 = y121*y77
        cdef double y433 = y124*y36
        cdef double y434 = y199*y204
        cdef double y435 = y210*y325
        cdef double y436 = y418*y435
        cdef double y437 = prst**3
        cdef double y438 = y102*y437
        cdef double y439 = prst**5
        cdef double y440 = nu*y103*y439
        cdef double y441 = prst**7
        cdef double y442 = nu*y441
        cdef double y443 = y106*y442
        cdef double y444 = y109*y441
        cdef double y445 = 0.00678224732971111*y437
        cdef double y446 = 0.0101733709945667*y439
        cdef double y447 = y432*y441
        cdef double y448 = y434*y437
        cdef double y449 = prst*y435
        cdef double y450 = y250*y255*(y170 + 0.000130208333333333*y236 + 0.000130208333333333*y241 + 0.000130208333333333*y244 + 0.000130208333333333*y246 + 0.000130208333333333*y96)**2
        cdef double y451 = y36*y46
        cdef double y452 = 12.0*y34
        cdef double y453 = y36*y50
        cdef double y454 = 2.0*pphi*y77
        cdef double y455 = pphi*y291
        cdef double y456 = pphi**3
        cdef double y457 = 4.0*y456
        cdef double y458 = y33*y454 + y40*y455 + y451*y457
        cdef double y459 = y453*y457 + y454*y48 + y455*y49
        cdef double y460 = delta*y52
        cdef double y461 = pphi*y1
        cdef double y462 = 4.0*y437
        cdef double y463 = 6.0*y439
        cdef double y464 = prst*y436 + y427*y462 + y428*y462 + y430*y463 + y431*y463 + y433*y462 + 589.775011960583*y438 - 67.9050514751178*y440 + 11.8620273619492*y443 + 0.975638950243592*y444 + 8.0*y447 + 5.09109256556341e-19*y448
        cdef double y465 = y254*y256
        cdef double y466 = y234*y286
        cdef double y467 = 16.0*y25
        cdef double y468 = y247*y250*y358/(y145*y361 + 4.0)
        cdef double y469 = dSO*y29
        cdef double y470 = 4.0*pphi*y32
        cdef double y471 = pphi*y37
        cdef double y472 = y456*y467
        cdef double y473 = y34*y460
        cdef double y474 = y129*y232
        cdef double y475 = -2.0*pphi*y144*y77 + y461*y474
        cdef double y476 = -y475

        # Evaluate Hamiltonian
        cdef double H
        H,_,_,_,_,_,_,_,_ = self._call(q,p,chi1_v,chi2_v,m_1,m_2,chi_1,chi_2,chiL1,chiL2)

        # Evaluate Heff Jacobian
        cdef double dHeffdr = x12*(-dSO*x14*x15*x19 + x19*(x24*(1.25*x3 - 0.75*x48 - 0.416666666666667*x57 + 1.66666666666667*x58 + 1.25*x6 + 1.25*x7) - x31*x55) + x19*(-x21*x22 - x23*x27 - x24*(-11.0625*nu + 1.13541666666667*x28 - 0.15625) - x29*x31 - x32*x37) + x43*(0.375*x14*x49*x50 - x24*(0.0833333333333333*x51 + x52)) + x44*(-x21*x38 - x24*(-0.0625*nu + 1.07291666666667*x28 + 0.15625) - x27*x39 - x31*x40 - x37*x41)) + 0.5*x205*(x192*x195*(30720.0*x0*x119*x138 + 7680.0*x13*x138*x210 - x14*(x177 + x178 - 3.0*x48) + x206*x24 - x209*(x109*(14350.0*nu - 1320.0*x28 + 2805.0) + x111*(5830.0*nu - 1640.0*x28 - 855.0) + 0.625*x202) - x209*(0.015625*x179*(-135.0*nu + 140.0*x28 - 15.0) - 6.09375*x197 - x199*(35.0*nu + 45.0)) - x215 - x35*(0.5*x200 + 0.25*x201 + 8.25*x58) - x35*(0.5*x196 + x2*x208 - 5.0*x57 + 4.5*x6 + 4.5*x7)) - x192*x203*(x14*x206 - x207*x24*x9)/x194**2 + x204*(-663.496888455656*nu*r**(-5.5)*x70 - nu*x21*x80 + 39.6112800271521*nu*x71*x73 + x1*x21*x25*x97*x98 + x1*x25*x60*x98*(r*x9 + x10 + x212)/x96**2 - x100 - x112*x27 + 7.59859378406358e-45*x113*x115*x142*x165*x167*x184*x220*x70 + 6.78168402777778e-8*x114*x14*x170*x171*x175*x184*x190 + x14*x70*(118.4*x211 - 51.6952380952381*x79) - 9.25454462627843e-34*x142*x218*x219/x164**3 - 2.24091649004576e-37*x165*x169*x171*x184*x218*x221*x35 + 1.69542100694444e-8*x169*x170*x171*x175*x184*x35*(x138*x187*x223 - x14*(x180*(12.0*nu - 3.0) - 7.875*x57 + x9*(9.0*nu + 8.4375)) + 38400.0*x140*x186 + x185*x189*x218*x34 - x187*x214 + x222 - x35*(0.125*x181 + 0.0625*x182 + 0.0625*x183) - 2.29252167428035e-22*x174*x188*x220*x34/x155) + 1.69542100694444e-8*x169*x170*x171*x175*x190*x220*x35 - 8.47710503472222e-8*x191*x209 + x21*x25*(-x14*(x101*(-5.25*nu - 2.8125) + x102*(2.25 - x208) + x105*(0.5625 - 2.25*nu)) + x24*x49 - x35*(0.03125*x104*x110 - 0.25*x106 + 0.0104166666666667*x108*x49)) - x216*x92 - x30*x73*x89 - x36*x70*x94 - 3.70688355060912*x74*x78 - 2.0*x81*x85 - 3.0*x84*x95 - 2.0*x86*x90 - 4.41515887225116e-12*x169*x175*x221*x224*x225/x141**3 - 6.62902677807736e-23*x165*x219*x224/x141**5 + 1.01821851311268e-18*x114**3*x142*x155*x165*x167*x70/r**12 - 1.65460508380811e-18*x168/r**14)) - (3.0*x20 + x9)*(pphi*x63 + pphi*x67 + pphi*x69 + x19*x59 + x43*x64)/x11**2

        cdef double dHeffdphi = 0

        cdef double dHeffdpr = x231*(11.8620273619492*nu*x228*x76 + 3.39084201388889e-8*prst*x169*x171*x175*x190*x225 + 5.09109256556341e-19*x113*x115*x142*x155*x165*x167*x226 + x14*x229*x94 + 8.0*x21*x228*x91 + 589.775011960583*x226*x72 - 67.9050514751178*x227*x75 + 0.975638950243592*x228*x79 + x229*x24*x84 + x229*x82 + x230*x24*x89 + x230*x87)

        cdef double dHeffdpphi = x12*(x18*x59 + 2.0*x18*x68 + x19*(x23*x232 + x233*x29 + x234*x32) + x42*x64 - 0.25*x42*x66 + x44*(x232*x39 + x233*x40 + x234*x41) + x63 + x67 + x69) + x231*(-pphi*x1*x193*x99 + 2.0*pphi*x112*x21)

        # Evaluate Heff Hessian
        cdef double d2Heffdr2 = y23*(y25*y26*y31 + y30*(y36*(-3.75*y13 - 3.75*y18 + 2.25*y58 - y69 - 3.75*y7 - 5.0*y70) + y41*y66) + y30*(y32*(1.4375*nu - 0.1875) + y33*y38 + y36*(-33.1875*nu + 3.40625*y39 - 0.46875) + y40*y42 + y44*y47) + y53*(-1.5*y25*y61 + y36*(y59*(1.25 - 2.5*X_2) + 0.25*y62)) + y54*(y32*(0.6875*nu + 0.1875) + y36*(-0.1875*nu + 3.21875*y39 + 0.46875) + y38*y48 + y42*y49 + y44*y51)) - 0.25*y250*y358**2 + 0.5*y424*(-y231*y255*y286*y362 + 2.0*y231*y247*y254**2/y233**3 + y234*y357*(61440.0*y0*y165*y167 + 15360.0*y165*y268*y35 - y25*(y242 + 0.5*y243 + 16.5*y70) - y25*(9.0*y13 + 9.0*y18 + y235 + y260*y379 - 10.0*y68) - 2.63243347181853e-19*y284 + y32*y360 - y36*(y368 - 6.0*y58) - y43*(1.25*y245 + y266*y365 + 0.00520833333333333*y364) - y43*(0.03125*y215*y263 - 12.1875*y237 - y262*y366)) + y248*(3649.23288650611*r**(-6.5)*y289 - 2.0*y1*y384*y92 - 8.0*y100*y25*y293 - 7.91520185839956e-48*y100*y336*y338*y396 + 12.9740924271319*y102*y107 - 178.250760122184*y105*y288 + 0.243909737560898*y108*y32 + 6.0*y111*y125 + y115*y41 + y117*y37 + y120*y41 + y122*y37 + y125*(51.6952380952381*nu*y77 - y292*y77) + 2.0*y129*y92 + y144*y38 + 1.62760416666667e-6*y171*y412*y413 + 1.16714400523217e-40*y196*y226*y312*y344*y407 - 2.5455462827817e-17*y200*y201*y305 - 6.103515625e-7*y202*y227*y25 + 7.59859378406358e-45*y220*y386*y394 + y220*y400*(y165*y223*y370 - y167*y371*y420 - y168*y24*y312*y422 - 2.29252167428035e-21*y169*y209*y351 + y169*y312*y421 + y209*y333*y421 - y223*y372 + y25*(y20*(36.0*nu + 33.75) + 0.25*y215*y346 - 31.5*y68) - 1.31621673590926e-18*y284*y349 + 153600.0*y334*y349 + y347*y389 - y348*y352*y422 + 15360.0*y348*y420 - y350*y354*y394 + 7.85795675813156e-45*y351*y352*y355 + y369 + y373*y423 - y377*y423 + y43*(0.0625*y135*(575.0*nu + 20.0*y39 - 185.0) + 0.0625*y20*(-5855.0*nu - 4305.0) + 0.625*y217) + 2.81299777100766e-6*y353*y385/y185**3) + y225*y394*y400 + 1.35633680555556e-7*y227*y32 + 4.66406554828496e-24*y229*y313*y326*y393 + y232*y297*y382 + y294*y44 - 4.0*y301*y79 - y313*y318*y389 + 5.08626302083333e-7*y324*y363 - y327*y329*y389 - 1.79273319203661e-36*y328*y403 + 2.24091649004576e-36*y328*y405 - y330*y341*y378 - 4.41515887225116e-12*y343*y378*y413 - 3.53212709780093e-11*y345*y403 + 4.41515887225116e-11*y345*y405 + y356*y401*y418 - y380*y80 + y382*y383*(y14*(2.0*y15 + 2.0*y16 + 2.0*y17) + y375 + y6*(4.0*y2 + 4.0*y3 + 4.0*y4) + y8*(2.0*y10 + 2.0*y11 + 2.0*y9)) + 2.69825540021952e-16*y385*y386 + 2.40618160283239e-32*y390*y391 - 1.48072714020455e-32*y391*y392 - 1.97563438385653e-43*y395*y397 + 1.72354696230011e-21*y395*y411 + 1.21577500545017e-43*y397*y398 - 1.10501271569469e-58*y397*y399 - 1.06064428449238e-21*y398*y411 + 9.64015065237337e-37*y399*y411 + y402*y404 - y402*y406 + y404*y417 - y406*y417 - y408*y410 - y408*y415 - y410*y419 - y415*y419 + y92*(y25*(y136*(1.125 - 4.5*nu) + y59*(-10.5*nu - 5.625) + y64*(4.5 - y379)) - y258*y36 + y43*(-1.25*y138 + y299*(-8050.0*nu + y264 + 1875.0) + y300*(-5450.0*nu + 760.0*y39 + 1095.0))) + 2.88925109089694e-20*y317*y393/y195**4 + 4.07287405245073e-17*y340*y412/y170**6 - 32.0*y381*y383*(y0 + 0.25*y21 + 0.25*y296)**2/y126**3 + 6.10931107867609e-18*y200*y228/r**11 + 2.31644711733135e-17*y303/r**15) - y256*y357*y362 - y257*(y20*y36*(12.0*y91 + 6.0) + y25*y359) + y287*(y25*(y359 - 12.0*y58) + y363*(3.75*y245 + 0.015625*y364 + y365*(17490.0*nu - 4920.0*y39 - 2565.0)) + y363*(0.03125*y215*(-405.0*nu + 420.0*y39 - 45.0) - 36.5625*y237 - y366*(105.0*nu + 135.0)) + y378 + y43*(0.25*y136*(-y367 - 15.0) + y298*(y261 - 155.0) + 41.25*y70) + y43*(22.5*y13 + 0.5*y135*(y367 + 5.0) + 22.5*y18 - 25.0*y68 + 22.5*y7))) - y74*(y72 + y73)*(pphi*y84 + pphi*y88 + pphi*y89 - y31*y76 + y54*y86) + (-6.0*r*y74 + 18.0*(0.333333333333333*y13 + 0.333333333333333*y18 + 0.333333333333333*y7 + y71)**2/y22**3)*(pphi*y94 + pphi*y97 + pphi*y99 + y31*y90 + y54*y95)

        cdef double d2Heffdphi2 = 0

        cdef double d2Heffdpr2 = y247*y424*y425*(1769.32503588175*y102*y205 - 339.525257375589*y103*y289 + 6.82947265356779*y104*y109 + 56.0*y104*y432 + 83.0341915336443*y105*y106 + 1.52732776966902e-18*y205*y434 + y426*y427 + y426*y428 + y426*y433 + y429*y430 + y429*y431 + y436) - 5129029357728.48*y450*(5.74938229854254e-11*y229*y449 + y427*y445 + y428*y445 + y430*y446 + y431*y446 + y433*y445 + y438 - 0.115137213510253*y440 + 0.020112800850135*y443 + 0.00165425616626294*y444 + 0.0135644946594222*y447 + 8.63226223952612e-22*y448)**2

        cdef double d2Heffdpphi2 = y23*(6.0*y30*y32*y65 + y30*(y32*(-4.171875*nu - 4.14322916666667*y39 + 0.4609375) + y451*y452 + y77*(-2.8125*nu - 0.9375)) + y458*(2.0*y27 + 2.0*y28) + 2.0*y459*y460 - 0.75*y54*y96 + y54*(y32*(-0.546875*nu - 1.59635416666667*y39 - 0.4609375) + y452*y453 + y77*(0.9375 - 0.5625*nu))) + y247*y424*y425*(y130*(-2.0*r - 4.0) + y77*(-y132 + y32*(y133 + y134 + y137) + y36*(-0.125*y138 + 0.00520833333333333*y140*y59 + y142*y365) + 2.0)) - 58982400.0*y450*(pphi*y144*y77 - y130*y461)**2

        cdef double  d2Heffdrdphi = 0

        cdef double d2Heffdrdpr = 0.5*y424*(y248*(237.667680162912*nu*y101*y439 - 2653.98755382262*nu*y288*y437 + 1.35633680555556e-7*prst*y202*y206*y210*y220*y225*y36 + 3.39084201388889e-8*prst*y206*y210*y220*y228*y25*y356 + 3.39084201388889e-8*prst*y206*y210*y225*y228*y25*y323 - prst*y210*y226*y229*y414 - prst*y229*y325*y409 - 29.655068404873*y103*y442 - 8.0*y111*y32*y437 - 12.0*y114*y36*y437 - 12.0*y116*y32*y439 - 18.0*y119*y36*y439 - 16.0*y121*y32*y441 - y124*y437*y467 + 4.07287405245073e-18*y171*y186*y196*y198*y304*y305*y437 - 3.70181785051137e-33*y171*y186*y198*y399*y437 + 3.03943751362543e-44*y171*y196*y198*y201*y203*y220*y323*y437 - 2.65161071123094e-22*y186*y336*y338*y437 - 6.61842033523243e-18*y199*y390*y437 - 0.975638950243592*y290*y441 + 4.0*y293*y36*y437 - y406*y449) - y464*y465 + y464*y466) - y464*y468

        cdef double d2Heffdrdpphi = y23*(-y29*y37*y66 + y30*(-y33*y470 - y40*y471 - y46*y472) + 0.75*y36*y473*y59 + y460*y86 - y469*y76 + y54*(-y470*y48 - y471*y49 - y472*y50) + y84 + y88 + y89) + 0.5*y424*(y248*(-pphi*y474 - y144*y470 + y232*y384*y461 + y301*y454 + y380*y454) - y465*y476 + y466*y476) - y468*y476 - y74*(y20 + 3.0*y71)*(2.0*y29*y98 + y30*y458 + y459*y54 + y460*y95 + y469*y90 - 0.25*y473*y96 + y94 + y97 + y99)

        cdef double d2Heffdphidpr = 0

        cdef double d2Heffdphidpphi = 0

        cdef double d2Heffdprdpphi = 14745600.0*y450*y464*y475

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

    cdef double xi(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL1,double chiL2):

        """
        Compute the tortoise coordinate conversion factor.

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1_v (double[:]): Dimensionless spin vector of the primary.
          chi2_v (double[:]): Dimensionless spin vector of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.
          chi_1 (double): Projection of chi1_v onto the Newtonian orbital angular momentum unit vector (lN).
          chi_2 (double): Projection of chi2_v onto the Newtonian orbital angular momentum unit vector (lN).
          chiL1 (double): Projection of chi1_v onto the orbital angular momentum unit vector (l).
          chiL2 (double): Projection of chi2_v onto the orbital angular momentum unit vector (l).

        Returns:
           (double) xi
        """

        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double chix1 = chi1_v[0]
        cdef double chiy1 = chi1_v[1]
        cdef double chiz1 = chi1_v[2]

        cdef double chix2 = chi2_v[0]
        cdef double chiy2 = chi2_v[1]
        cdef double chiz2 = chi2_v[2]

        cdef double pphi = L

        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs['a6']
        cdef double dSO = c_coeffs['dSO']



        # Extra quantities used in the Jacobian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double delta = self.EOBpars.p_params.delta

        # Actual Hamiltonian expressions
        cdef double Dbpm = r*(6730497718123.02*nu**3 + 133772083200.0*nu**2*r**2 + 1822680546449.21*nu**2*r + 80059249540278.2*nu**2 + 2589101062873.81*nu*r**2 + 10611661054566.2*nu*r - 12049908701745.2*nu + 5107745331375.71*r**2 - 326837426.241486*r*(14700.0*nu + 42911.0) - 39476764256925.6*r - (-5041721180160.0*nu**2 - 25392914995744.3*nu - 879923036160.0*r**2 - 283115520.0*r*(14700.0*nu + 42911.0) + 104186110149937.0)*log(r) + 5787938193408.0*log(r)**2 + 275059053208689.0)/(55296.0*nu*(14515200.0*nu**3 - 42636451.6032331*nu**2 - 2510664218.28128*nu + 1002013764.01019) - 967680.0*r**3*(-138240.0*nu**2 - 2675575.66847905*nu - 5278341.3229329) - 9216.0*r**2*(-197773496.793534*nu**2 - 630116198.873299*nu + 5805304367.87913) + r*(5927865218923.02*nu**3 + 43133561885859.3*nu**2 + 43393301259014.8*nu + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0) + 5787938193408.0*r*log(r)**2 + (-1698693120.0*nu*(11592.0*nu + 69847.0) + 879923036160.0*r**3 + 283115520.0*r**2*(14700.0*nu + 42911.0) + 49152.0*r*(102574080.0*nu**2 + 409207698.136075*nu - 2119671837.36038))*log(r))

        cdef double Apm = 7680.0*r**4*(-5416406.59541186*nu**2 + 28.0*nu*(1920.0*a6 + 733955.307463037) + 2048.0*nu*(756.0*nu + 336.0*r + 407.0)*log(r) - 7.0*r*(-185763.092693281*nu**2 + 938918.400156317*nu - 245760.0) - 3440640.0)/(241555486248.807*nu**4 + 1120.0*nu**3*(-17833256.898555*r**2 - 163683964.822551*r - 1188987459.03162) + 7.0*nu**2*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 1426660551.8844*r**5 - 3089250703.76879*r**4 - 6178501407.53758*r**3 + 2064783811.32587*r**2 + 122635399361.987*r + 276057889687.011) + 67645734912.0*nu**2*log(r)**2 + 53760.0*nu*(7680.0*a6*(r**4 + 2.0*r**3 + 4.0*r**2 + 8.0*r + 16.0) + 128.0*r*(-6852.34813868015*r**4 + 4264.6962773603*r**3 + 8529.39255472061*r**2 + 13218.7851094412*r - 33722.4297811176) + 113485.217444961*r*(-r**4 + 2.0*r**3 + 4.0*r**2 + 8.0*r + 16.0) + 148.04406601634*r*(349.0*r**4 + 1926.0*r**3 + 3852.0*r**2 + 7704.0*r + 36400.0)) + 32768.0*nu*(-1882456.23663972*nu**2 - 38842241.4769507*nu + 161280.0*r**5 + 480.0*r**4*(756.0*nu + 1079.0) + 960.0*r**3*(756.0*nu + 1079.0) + 1920.0*r**2*(588.0*nu + 1079.0) + 240.0*r*(-3024.0*nu**2 - 7466.27061066206*nu + 17264.0) + 13447680.0)*log(r) + 13212057600.0*r**5)

        cdef double t2 = chix2**2 + chiy2**2 + chiz2**2

        cdef double t1 = chix1**2 + chiy1**2 + chiz1**2

        cdef double ap2 = X_1**2*t1 + X_1*X_2*(2.0*chix1*chix2 + 2.0*chiy1*chiy2 + 2.0*chiz1*chiz2) + X_2**2*t2

        cdef double xi = Dbpm**0.5*r**2*(Apm + ap2/r**2)/(ap2 + r**2)

        return xi

    cpdef dynamics(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL1,double chiL2):

        """
        Compute the dynamics from the Hamiltonian,i.e., dHdr, dHdphi, dHdpr, dHdpphi,H and xi.

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1_v (double[:]): Dimensionless spin vector of the primary.
          chi2_v (double[:]): Dimensionless spin vector of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.
          chi_1 (double): Projection of chi1_v onto the Newtonian orbital angular momentum unit vector (lN).
          chi_2 (double): Projection of chi2_v onto the Newtonian orbital angular momentum unit vector (lN).
          chiL1 (double): Projection of chi1_v onto the orbital angular momentum unit vector (l).
          chiL2 (double): Projection of chi2_v onto the orbital angular momentum unit vector (l).

        Returns:
           (tuple) dHdr, dHdphi, dHdpr, dHdpphi,H and xi
        """

        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double chix1 = chi1_v[0]
        cdef double chiy1 = chi1_v[1]
        cdef double chiz1 = chi1_v[2]

        cdef double chix2 = chi2_v[0]
        cdef double chiy2 = chi2_v[1]
        cdef double chiz2 = chi2_v[2]

        cdef double pphi = L

        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs['a6']
        cdef double dSO = c_coeffs['dSO']


        # Extra quantities used in the Jacobian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double delta = self.EOBpars.p_params.delta
        cdef double x0 = r**3
        cdef double x1 = r + 2.0
        cdef double x2 = X_1*(2.0*chix1*chix2 + 2.0*chiy1*chiy2 + 2.0*chiz1*chiz2)
        cdef double x3 = X_2*x2
        cdef double x4 = X_1**2
        cdef double x5 = chix1**2 + chiy1**2 + chiz1**2
        cdef double x6 = x4*x5
        cdef double x7 = X_2**2*(chix2**2 + chiy2**2 + chiz2**2)
        cdef double x8 = x6 + x7
        cdef double x9 = x3 + x8
        cdef double x10 = x1*x9
        cdef double x11 = x0 + x10
        cdef double x12 = x11**(-1)
        cdef double x13 = r**4
        cdef double x14 = x13**(-1)
        cdef double x15 = 3.0*nu
        cdef double x16 = X_1*chiL1
        cdef double x17 = X_2*chiL2
        cdef double x18 = x16 + x17
        cdef double x19 = pphi*x18
        cdef double x20 = r**2
        cdef double x21 = x20**(-1)
        cdef double x22 = 0.71875*nu - 0.09375
        cdef double x23 = -1.40625*nu - 0.46875
        cdef double x24 = x0**(-1)
        cdef double x25 = pphi**2
        cdef double x26 = x24*x25
        cdef double x27 = 2.0*x26
        cdef double x28 = nu**2
        cdef double x29 = -2.0859375*nu - 2.07161458333333*x28 + 0.23046875
        cdef double x30 = 3.0*x14
        cdef double x31 = x25*x30
        cdef double x32 = 0.5859375*nu + 1.34765625*x28 + 0.41015625
        cdef double x33 = pphi**4
        cdef double x34 = r**5
        cdef double x35 = x34**(-1)
        cdef double x36 = 4.0*x35
        cdef double x37 = x33*x36
        cdef double x38 = 0.34375*nu + 0.09375
        cdef double x39 = 0.46875 - 0.28125*nu
        cdef double x40 = -0.2734375*nu - 0.798177083333333*x28 - 0.23046875
        cdef double x41 = -0.3515625*nu + 0.29296875*x28 - 0.41015625
        cdef double x42 = x16 - x17
        cdef double x43 = pphi*x42
        cdef double x44 = delta*x43
        cdef double x45 = chi_1*X_1
        cdef double x46 = chi_2*X_2
        cdef double x47 = x45 + x46
        cdef double x48 = x47**2
        cdef double x49 = -x48 + x9
        cdef double x50 = delta*x25
        cdef double x51 = delta*x9
        cdef double x52 = x49*(0.416666666666667 - 0.833333333333333*X_2)
        cdef double x53 = x45 - x46
        cdef double x54 = x4*x5 - x47*x53 - x7
        cdef double x55 = -0.125*x3 + 0.125*x48 + 0.5*x54*(0.5 - X_2) - 0.125*x6 - 0.125*x7
        cdef double x56 = x6 - x7
        cdef double x57 = delta*x56
        cdef double x58 = delta*x54
        cdef double x59 = nu*dSO*x24
        cdef double x60 = r**(-1)
        cdef double x61 = x21*x25
        cdef double x62 = x14*x33
        cdef double x63 = x18*(x21*(-5.53125*nu + 0.567708333333333*x28 - 0.078125) + x22*x60 + x23*x61 + x26*x29 + x32*x62 + 1.75)
        cdef double x64 = delta*(x21*(-0.03125*nu + 0.536458333333333*x28 + 0.078125) + x26*x40 + x38*x60 + x39*x61 + x41*x62 + 0.25)
        cdef double x65 = x24*x49
        cdef double x66 = x50*x65
        cdef double x67 = x42*(x21*(0.0416666666666667*x51 + 0.5*x52) - 0.125*x66)
        cdef double x68 = x26*x55
        cdef double x69 = x18*(x21*(0.208333333333333*delta*x56 - 0.625*x3 + 0.375*x48 - 0.833333333333333*x58 - 0.625*x6 - 0.625*x7) + x68)
        cdef double x70 = prst**4
        cdef double x71 = r**(-4.5)
        cdef double x72 = nu*x71
        cdef double x73 = prst**6
        cdef double x74 = r**(-3.5)
        cdef double x75 = nu*x74
        cdef double x76 = r**(-2.5)
        cdef double x77 = prst**8
        cdef double x78 = nu*x77
        cdef double x79 = nu*x60
        cdef double x80 = 0.121954868780449*x77
        cdef double x81 = 8.0*nu - 6.0*x28
        cdef double x82 = x21*x81
        cdef double x83 = nu**3
        cdef double x84 = 92.7110442849544*nu - 131.0*x28 + 10.0*x83
        cdef double x85 = x24*x70
        cdef double x86 = -2.78300763695006*nu - 5.4*x28 + 6.0*x83
        cdef double x87 = x21*x86
        cdef double x88 = nu**4
        cdef double x89 = -33.9782122170436*nu - 89.5298327361234*x28 + 188.0*x83 - 14.0*x88
        cdef double x90 = x24*x73
        cdef double x91 = 1.38977750996128*nu + 3.33842023648322*x28 + 3.42857142857143*x83 - 6.0*x88
        cdef double x92 = x77*x91
        cdef double x93 = log(r)
        cdef double x94 = nu*(452.542166996693 - 51.6952380952381*x93) + x28*(118.4*x93 - 1796.13660498019) + 602.318540416564*x83
        cdef double x95 = x14*x70
        cdef double x96 = r*x10 + x13
        cdef double x97 = x96**(-1)
        cdef double x98 = x18**2
        cdef double x99 = x97*x98
        cdef double x100 = x25*x60*x99
        cdef double x101 = 0.5*x49
        cdef double x102 = 0.5*x54
        cdef double x103 = -x3 + x8
        cdef double x104 = x103 - x53**2
        cdef double x105 = 0.5*x104
        cdef double x106 = x58*(98.0*nu + 43.0)
        cdef double x107 = 264.0*x28
        cdef double x108 = -1610.0*nu + x107 + 375.0
        cdef double x109 = 0.00260416666666667*x49
        cdef double x110 = -1090.0*nu + 152.0*x28 + 219.0
        cdef double x111 = 0.0078125*x104
        cdef double x112 = -x101*x21 + x14*(-0.0625*x106 + x108*x109 + x110*x111) + x24*(x101*(-1.75*nu - 0.9375) + x102*(0.75 - 1.5*X_2) + x105*(0.1875 - 0.75*nu)) + 1.0
        cdef double x113 = r**(-13)
        cdef double x114 = x20 + x9
        cdef double x115 = x114**4
        cdef double x116 = x21*x9
        cdef double x117 = 756.0*nu
        cdef double x118 = 336.0*r + x117 + 407.0
        cdef double x119 = 2048.0*nu*x118*x93 + 28.0*nu*(1920.0*a6 + 733955.307463037) - 7.0*r*(938918.400156317*nu - 185763.092693281*x28 - 245760.0) - 5416406.59541186*x28 - 3440640.0
        cdef double x120 = x93**2
        cdef double x121 = x120*x28
        cdef double x122 = x83*(-163683964.822551*r - 17833256.898555*x20 - 1188987459.03162)
        cdef double x123 = x28*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 122635399361.987*r - 6178501407.53758*x0 - 3089250703.76879*x13 + 2064783811.32587*x20 + 1426660551.8844*x34 + 276057889687.011)
        cdef double x124 = 588.0*nu + 1079.0
        cdef double x125 = x117 + 1079.0
        cdef double x126 = x0*x125
        cdef double x127 = -38842241.4769507*nu + 240.0*r*(-7466.27061066206*nu - 3024.0*x28 + 17264.0) + 1920.0*x124*x20 + 480.0*x125*x13 + 960.0*x126 - 1882456.23663972*x28 + 161280.0*x34 + 13447680.0
        cdef double x128 = nu*x93
        cdef double x129 = x127*x128
        cdef double x130 = 8.0*r
        cdef double x131 = 2.0*x0 + x130 + 4.0*x20 + 16.0
        cdef double x132 = 7680.0*a6
        cdef double x133 = 128.0*r
        cdef double x134 = 7704.0*r
        cdef double x135 = 148.04406601634*r
        cdef double x136 = 113485.217444961*r
        cdef double x137 = nu*(x132*(x13 + x131) + x133*(13218.7851094412*r + 4264.6962773603*x0 - 6852.34813868015*x13 + 8529.39255472061*x20 - 33722.4297811176) + x135*(1926.0*x0 + 349.0*x13 + x134 + 3852.0*x20 + 36400.0) + x136*(-x13 + x131))
        cdef double x138 = (67645734912.0*x121 + 1120.0*x122 + 7.0*x123 + 32768.0*x129 + 53760.0*x137 + 13212057600.0*x34 + 241555486248.807*x88)**(-1)
        cdef double x139 = x13*x138
        cdef double x140 = x119*x139
        cdef double x141 = 0.000130208333333333*x116 + x140
        cdef double x142 = x141**(-4)
        cdef double x143 = r*x120
        cdef double x144 = -630116198.873299*nu - 197773496.793534*x28 + 5805304367.87913
        cdef double x145 = x144*x20
        cdef double x146 = -2675575.66847905*nu - 138240.0*x28 - 5278341.3229329
        cdef double x147 = x0*x146
        cdef double x148 = nu*(-2510664218.28128*nu - 42636451.6032331*x28 + 14515200.0*x83 + 1002013764.01019)
        cdef double x149 = 43393301259014.8*nu + 43133561885859.3*x28 + 5927865218923.02*x83 + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0
        cdef double x150 = r*x149
        cdef double x151 = 14700.0*nu + 42911.0
        cdef double x152 = 283115520.0*x151
        cdef double x153 = -1698693120.0*nu*(11592.0*nu + 69847.0) + 49152.0*r*(409207698.136075*nu + 102574080.0*x28 - 2119671837.36038) + 879923036160.0*x0 + x152*x20
        cdef double x154 = x153*x93
        cdef double x155 = (x143 - 1.59227685093395e-9*x145 - 1.67189069348064e-7*x147 + 9.55366110560367e-9*x148 + 1.72773095804465e-13*x150 + 1.72773095804465e-13*x154)**2
        cdef double x156 = nu*r
        cdef double x157 = nu*x20
        cdef double x158 = r*x28
        cdef double x159 = r*x151
        cdef double x160 = x20*x28
        cdef double x161 = 5041721180160.0*x28 - 104186110149937.0
        cdef double x162 = -25392914995744.3*nu - r*x152 - x161 - 879923036160.0*x20
        cdef double x163 = x162*x93
        cdef double x164 = -0.0438084424460039*nu - 0.143521050466841*r + 0.0210425293255724*x120 + 0.0385795738434214*x156 + 0.00941289164152486*x157 + 0.00662650629087394*x158 - 1.18824456940711e-6*x159 + 0.000486339502879429*x160 - 3.63558293513537e-15*x163 + 0.0185696317637669*x20 + 0.291062041428379*x28 + 0.0244692826489756*x83 + 1
        cdef double x165 = x164**(-2)
        cdef double x166 = 4.0*x28
        cdef double x167 = 0.3125*delta*x56*(18.0*nu - 1.0) - 0.03125*(-33.0*nu + 32.0*x28 - 5.0)*(5.0*x3 + 5.0*x6 + 5.0*x7) - 0.03125*(-5.0*nu + x166 + 1.0)*(-15.0*x3 + 15.0*x6 + 15.0*x7)
        cdef double x168 = x115*x142*x155*x165*x167*x70
        cdef double x169 = x114**2
        cdef double x170 = prst**2
        cdef double x171 = x141**(-2)
        cdef double x172 = 1822680546449.21*x28
        cdef double x173 = 5787938193408.0*x120
        cdef double x174 = -12049908701745.2*nu + r*x172 - 39476764256925.6*r + 10611661054566.2*x156 + 2589101062873.81*x157 - 326837426.241486*x159 + 133772083200.0*x160 - x163 + x173 + 5107745331375.71*x20 + 80059249540278.2*x28 + 6730497718123.02*x83 + 275059053208689.0
        cdef double x175 = x174**(-1)
        cdef double x176 = 4.0*nu
        cdef double x177 = 3.0*x3
        cdef double x178 = 3.0*x6 + 3.0*x7
        cdef double x179 = -x177 + x178
        cdef double x180 = 0.0625*x179
        cdef double x181 = x57*(26.0*nu + 449.0)
        cdef double x182 = x9*(-1171.0*nu - 861.0)
        cdef double x183 = x103*(115.0*nu + x166 - 37.0)
        cdef double x184 = 5787938193408.0*x143 - 9216.0*x145 - 967680.0*x147 + 55296.0*x148 + x150 + x154
        cdef double x185 = x184**(-1)
        cdef double x186 = x174*x185
        cdef double x187 = x186*x34
        cdef double x188 = x119*x138
        cdef double x189 = 7680.0*x188
        cdef double x190 = x116 + x14*(0.03125*x181 + 0.015625*x182 + 0.015625*x183) + x187*x189 + x24*(x180*(x176 - 1.0) - 2.625*x57 + x9*(x15 + 2.8125))
        cdef double x191 = x169*x170*x171*x175*x184*x190
        cdef double x192 = -x1*x100 + x112*x61 + 1.27277314139085e-19*x113*x168 + 1.69542100694444e-8*x191*x35 + x21*x92 + 147.443752990146*x70*x72 + x70*x82 - 11.3175085791863*x73*x75 + x73*x87 + 1.48275342024365*x76*x78 + x79*x80 + x84*x85 + x89*x90 + x94*x95 + 1.0
        cdef double x193 = 2.0*x60
        cdef double x194 = x116*(x193 + 1.0) + 1.0
        cdef double x195 = x194**(-1)
        cdef double x196 = x103*(x176 + 1.0)
        cdef double x197 = x57*(2.0*nu - 3.0)
        cdef double x198 = 7.0*nu
        cdef double x199 = 0.390625*x3 + 0.390625*x6 + 0.390625*x7
        cdef double x200 = x49*(x198 - 31.0)
        cdef double x201 = x104*(-x176 - 3.0)
        cdef double x202 = x58*(68.0*nu - 1.0)
        cdef double x203 = x116 + x14*(0.125*x200 + 0.0625*x201 + 2.0625*x58) + x14*(0.125*x196 + 1.125*x3 - 1.25*x57 + 1.125*x6 + 1.125*x7) + 7680.0*x140 + x35*(x109*(2870.0*nu - x107 + 561.0) + x111*(1166.0*nu - 328.0*x28 - 171.0) + 0.125*x202) + x35*(0.015625*x179*(-27.0*nu + 28.0*x28 - 3.0) - 1.21875*x197 - x199*(x198 + 9.0)) + x65
        cdef double x204 = x195*x203
        cdef double x205 = (x192*x204)**(-0.5)
        cdef double x206 = -2.0*x3 - 2.0*x6 - 2.0*x7
        cdef double x207 = 4.0*x60 + 2.0
        cdef double x208 = 4.5*X_2
        cdef double x209 = r**(-6)
        cdef double x210 = -6572428.80109422*nu + 2048.0*x118*x79 + 688128.0*x128 + 1300341.64885296*x28 + 1720320.0
        cdef double x211 = x28*x60
        cdef double x212 = 4.0*x0
        cdef double x213 = x130 + 6.0*x20 + 8.0
        cdef double x214 = 1.31621673590926e-19*x119*(53760.0*nu*(3740417.71815805*r + 1057984.42953951*x0 - 938918.400156317*x13 + x132*(x212 + x213) + x133*(17058.7851094412*r - 27409.3925547206*x0 + 12794.0888320809*x20 + 13218.7851094412) + x135*(1396.0*x0 + x134 + 5778.0*x20 + 7704.0) + x136*(-x212 + x213) + 2115968.85907902*x20 + 2888096.47013111) + 32768.0*x127*x79 + 32768.0*x128*(-1791904.9465589*nu + 3840.0*r*x124 + 2880.0*x125*x20 + 1920.0*x126 + 806400.0*x13 - 725760.0*x28 + 4143360.0) + 66060288000.0*x13 + 135291469824.0*x211*x93 + 7.0*x28*(-117964800.0*a6 + 4129567622.65173*r - 12357002815.0752*x0 + 7133302759.42198*x13 - 18535504222.6128*x20 + 122635399361.987) + 7.0*x83*(-5706642207.53758*r - 26189434371.6082))/(0.28004222119933*x121 + 4.63661586574928e-9*x122 + 2.8978849160933e-11*x123 + 1.35654132757922e-7*x129 + 2.22557561555966e-7*x137 + 0.0546957463279941*x34 + x88)**2
        cdef double x215 = x13*x214
        cdef double x216 = 2.0*x24
        cdef double x217 = 11575876386816.0*x93
        cdef double x218 = 5807150888816.34*nu + 10215490662751.4*r + 5178202125747.62*x156 + 267544166400.0*x158 - x162*x60 + x172 + x217*x60 + x93*(4161798144000.0*nu + 1759846072320.0*r + 12148770078720.0) - 53501685054374.1
        cdef double x219 = x113*x115*x155*x167*x70
        cdef double x220 = -18432.0*r*x144 - 2903040.0*x146*x20 + x149 + x153*x60 + x173 + x217 + x93*(20113376778784.3*nu + 566231040.0*x159 + x161 + 2639769108480.0*x20)
        cdef double x221 = x170*x190
        cdef double x222 = x206*x24
        cdef double x223 = 7680.0*x210
        cdef double x224 = 30720.0*x0*x188 + x139*x223 - x215 + x222
        cdef double x225 = x184*x35
        cdef double x226 = prst**3
        cdef double x227 = prst**5
        cdef double x228 = prst**7
        cdef double x229 = 4.0*x226
        cdef double x230 = 6.0*x227
        cdef double x231 = x203*x205/(x116*x207 + 2.0)
        cdef double x232 = 2.0*pphi*x21
        cdef double x233 = pphi*x216
        cdef double x234 = 4.0*pphi**3*x14


        # Evaluate Hamiltonian
        cdef double H,xi
        H,xi,_,_,_,_,_,_,_  = self._call(q,p,chi1_v,chi2_v,m_1,m_2,chi_1,chi_2,chiL1,chiL2)

        # Heff Jacobian expressions
        cdef double dHeffdr = x12*(-dSO*x14*x15*x19 + x19*(x24*(1.25*x3 - 0.75*x48 - 0.416666666666667*x57 + 1.66666666666667*x58 + 1.25*x6 + 1.25*x7) - x31*x55) + x19*(-x21*x22 - x23*x27 - x24*(-11.0625*nu + 1.13541666666667*x28 - 0.15625) - x29*x31 - x32*x37) + x43*(0.375*x14*x49*x50 - x24*(0.0833333333333333*x51 + x52)) + x44*(-x21*x38 - x24*(-0.0625*nu + 1.07291666666667*x28 + 0.15625) - x27*x39 - x31*x40 - x37*x41)) + 0.5*x205*(x192*x195*(30720.0*x0*x119*x138 + 7680.0*x13*x138*x210 - x14*(x177 + x178 - 3.0*x48) + x206*x24 - x209*(x109*(14350.0*nu - 1320.0*x28 + 2805.0) + x111*(5830.0*nu - 1640.0*x28 - 855.0) + 0.625*x202) - x209*(0.015625*x179*(-135.0*nu + 140.0*x28 - 15.0) - 6.09375*x197 - x199*(35.0*nu + 45.0)) - x215 - x35*(0.5*x200 + 0.25*x201 + 8.25*x58) - x35*(0.5*x196 + x2*x208 - 5.0*x57 + 4.5*x6 + 4.5*x7)) - x192*x203*(x14*x206 - x207*x24*x9)/x194**2 + x204*(-663.496888455656*nu*r**(-5.5)*x70 - nu*x21*x80 + 39.6112800271521*nu*x71*x73 + x1*x21*x25*x97*x98 + x1*x25*x60*x98*(r*x9 + x10 + x212)/x96**2 - x100 - x112*x27 + 7.59859378406358e-45*x113*x115*x142*x165*x167*x184*x220*x70 + 6.78168402777778e-8*x114*x14*x170*x171*x175*x184*x190 + x14*x70*(118.4*x211 - 51.6952380952381*x79) - 9.25454462627843e-34*x142*x218*x219/x164**3 - 2.24091649004576e-37*x165*x169*x171*x184*x218*x221*x35 + 1.69542100694444e-8*x169*x170*x171*x175*x184*x35*(x138*x187*x223 - x14*(x180*(12.0*nu - 3.0) - 7.875*x57 + x9*(9.0*nu + 8.4375)) + 38400.0*x140*x186 + x185*x189*x218*x34 - x187*x214 + x222 - x35*(0.125*x181 + 0.0625*x182 + 0.0625*x183) - 2.29252167428035e-22*x174*x188*x220*x34/x155) + 1.69542100694444e-8*x169*x170*x171*x175*x190*x220*x35 - 8.47710503472222e-8*x191*x209 + x21*x25*(-x14*(x101*(-5.25*nu - 2.8125) + x102*(2.25 - x208) + x105*(0.5625 - 2.25*nu)) + x24*x49 - x35*(0.03125*x104*x110 - 0.25*x106 + 0.0104166666666667*x108*x49)) - x216*x92 - x30*x73*x89 - x36*x70*x94 - 3.70688355060912*x74*x78 - 2.0*x81*x85 - 3.0*x84*x95 - 2.0*x86*x90 - 4.41515887225116e-12*x169*x175*x221*x224*x225/x141**3 - 6.62902677807736e-23*x165*x219*x224/x141**5 + 1.01821851311268e-18*x114**3*x142*x155*x165*x167*x70/r**12 - 1.65460508380811e-18*x168/r**14)) - (3.0*x20 + x9)*(pphi*x63 + pphi*x67 + pphi*x69 + x19*x59 + x43*x64)/x11**2

        cdef double  dHeffdphi = 0

        cdef double  dHeffdpr = x231*(11.8620273619492*nu*x228*x76 + 3.39084201388889e-8*prst*x169*x171*x175*x190*x225 + 5.09109256556341e-19*x113*x115*x142*x155*x165*x167*x226 + x14*x229*x94 + 8.0*x21*x228*x91 + 589.775011960583*x226*x72 - 67.9050514751178*x227*x75 + 0.975638950243592*x228*x79 + x229*x24*x84 + x229*x82 + x230*x24*x89 + x230*x87)

        cdef double  dHeffdpphi = x12*(x18*x59 + 2.0*x18*x68 + x19*(x23*x232 + x233*x29 + x234*x32) + x42*x64 - 0.25*x42*x66 + x44*(x232*x39 + x233*x40 + x234*x41) + x63 + x67 + x69) + x231*(-pphi*x1*x193*x99 + 2.0*pphi*x112*x21)
        cdef double  M2 = M*M
        cdef double  nuH = nu*H
        # Compute H Jacobian
        cdef double  dHdr = M2 * dHeffdr / nuH
        cdef double  dHdphi = M2 * dHeffdphi / nuH
        cdef double  dHdpr = M2 * dHeffdpr / nuH
        cdef double  dHdpphi = M2 * dHeffdpphi / nuH

        return dHdr, dHdphi, dHdpr, dHdpphi,H,xi

    cpdef double omega(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL1,double chiL2):

        """
        Compute the orbital frequency from the Hamiltonian.

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1_v (double[:]): Dimensionless spin vector of the primary.
          chi2_v (double[:]): Dimensionless spin vector of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.
          chi_1 (double): Projection of chi1_v onto the Newtonian orbital angular momentum unit vector (lN).
          chi_2 (double): Projection of chi2_v onto the Newtonian orbital angular momentum unit vector (lN).
          chiL1 (double): Projection of chi1_v onto the orbital angular momentum unit vector (l).
          chiL2 (double): Projection of chi2_v onto the orbital angular momentum unit vector (l).

        Returns:
           (double) dHdpphi
        """

        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double chix1 = chi1_v[0]
        cdef double chiy1 = chi1_v[1]
        cdef double chiz1 = chi1_v[2]

        cdef double chix2 = chi2_v[0]
        cdef double chiy2 = chi2_v[1]
        cdef double chiz2 = chi2_v[2]

        cdef double pphi = L

        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs['a6']
        cdef double dSO = c_coeffs['dSO']


        # Extra quantities used in the Jacobian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double delta = self.EOBpars.p_params.delta
        cdef double z0 = r**3
        cdef double z1 = r + 2.0
        cdef double z2 = X_1*X_2*(2.0*chix1*chix2 + 2.0*chiy1*chiy2 + 2.0*chiz1*chiz2)
        cdef double z3 = X_1**2
        cdef double z4 = chix1**2 + chiy1**2 + chiz1**2
        cdef double z5 = z3*z4
        cdef double z6 = X_2**2*(chix2**2 + chiy2**2 + chiz2**2)
        cdef double z7 = z5 + z6
        cdef double z8 = z2 + z7
        cdef double z9 = z1*z8
        cdef double z10 = z0**(-1)
        cdef double z11 = X_1*chiL1
        cdef double z12 = X_2*chiL2
        cdef double z13 = z11 + z12
        cdef double z14 = -1.40625*nu - 0.46875
        cdef double z15 = r**2
        cdef double z16 = z15**(-1)
        cdef double z17 = 2.0*pphi
        cdef double z18 = z16*z17
        cdef double z19 = nu**2
        cdef double z20 = -2.0859375*nu - 2.07161458333333*z19 + 0.23046875
        cdef double z21 = z10*z17
        cdef double z22 = 0.5859375*nu + 1.34765625*z19 + 0.41015625
        cdef double z23 = r**4
        cdef double z24 = z23**(-1)
        cdef double z25 = 4.0*pphi**3*z24
        cdef double z26 = 0.46875 - 0.28125*nu
        cdef double z27 = -0.2734375*nu - 0.798177083333333*z19 - 0.23046875
        cdef double z28 = -0.3515625*nu + 0.29296875*z19 - 0.41015625
        cdef double z29 = z11 - z12
        cdef double z30 = delta*z29
        cdef double z31 = pphi**2
        cdef double z32 = chi_1*X_1
        cdef double z33 = chi_2*X_2
        cdef double z34 = z32 + z33
        cdef double z35 = z34**2
        cdef double z36 = -z35 + z8
        cdef double z37 = z10*z36
        cdef double z38 = z31*z37
        cdef double z39 = r**(-1)
        cdef double z40 = z16*z31
        cdef double z41 = z10*z31
        cdef double z42 = pphi**4*z24
        cdef double z43 = z32 - z33
        cdef double z44 = z3*z4 - z34*z43 - z6
        cdef double z45 = z41*(-0.125*z2 + 0.125*z35 + 0.5*z44*(0.5 - X_2) - 0.125*z5 - 0.125*z6)
        cdef double z46 = 0.5*z36
        cdef double z47 = z5 - z6
        cdef double z48 = delta*z44
        cdef double z49 = z16*z8
        cdef double z50 = 2.0*z39
        cdef double z51 = z1*z13**2/(r*z9 + z23)
        cdef double z52 = -z2 + z7
        cdef double z53 = -z43**2 + z52
        cdef double z54 = 264.0*z19
        cdef double z55 = 0.00260416666666667*z36
        cdef double z56 = 0.0078125*z53
        cdef double z57 = z10*(0.5*z44*(0.75 - 1.5*X_2) + z46*(-1.75*nu - 0.9375) + 0.5*z53*(0.1875 - 0.75*nu)) - z16*z46 + z24*(-0.0625*z48*(98.0*nu + 43.0) + z55*(-1610.0*nu + z54 + 375.0) + z56*(-1090.0*nu + 152.0*z19 + 219.0)) + 1.0
        cdef double z58 = delta*z47
        cdef double z59 = 4.0*nu
        cdef double z60 = r**5
        cdef double z61 = z60**(-1)
        cdef double z62 = 7.0*nu
        cdef double z63 = -3.0*z2 + 3.0*z5 + 3.0*z6
        cdef double z64 = nu**4
        cdef double z65 = log(r)
        cdef double z66 = z65**2
        cdef double z67 = nu**3
        cdef double z68 = 756.0*nu
        cdef double z69 = z68 + 1079.0
        cdef double z70 = 8.0*r + 2.0*z0 + 4.0*z15 + 16.0
        cdef double z71 = (2048.0*nu*z65*(336.0*r + z68 + 407.0) + 28.0*nu*(1920.0*a6 + 733955.307463037) - 7.0*r*(938918.400156317*nu - 185763.092693281*z19 - 245760.0) - 5416406.59541186*z19 - 3440640.0)/(32768.0*nu*z65*(-38842241.4769507*nu + 240.0*r*(-7466.27061066206*nu - 3024.0*z19 + 17264.0) + 960.0*z0*z69 + 1920.0*z15*(588.0*nu + 1079.0) - 1882456.23663972*z19 + 480.0*z23*z69 + 161280.0*z60 + 13447680.0) + 53760.0*nu*(7680.0*a6*(z23 + z70) + 113485.217444961*r*(-z23 + z70) + 148.04406601634*r*(7704.0*r + 1926.0*z0 + 3852.0*z15 + 349.0*z23 + 36400.0) + 128.0*r*(13218.7851094412*r + 4264.6962773603*z0 + 8529.39255472061*z15 - 6852.34813868015*z23 - 33722.4297811176)) + 67645734912.0*z19*z66 + 7.0*z19*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 122635399361.987*r - 6178501407.53758*z0 + 2064783811.32587*z15 - 3089250703.76879*z23 + 1426660551.8844*z60 + 276057889687.011) + 13212057600.0*z60 + 241555486248.807*z64 + 1120.0*z67*(-163683964.822551*r - 17833256.898555*z15 - 1188987459.03162))
        cdef double z72 = z23*z71
        cdef double z73 = z24*(0.125*z36*(z62 - 31.0) + 2.0625*z48 + 0.0625*z53*(-z59 - 3.0)) + z24*(1.125*z2 + 1.125*z5 + 0.125*z52*(z59 + 1.0) - 1.25*z58 + 1.125*z6) + z37 + z49 + z61*(0.125*z48*(68.0*nu - 1.0) + z55*(2870.0*nu - z54 + 561.0) + z56*(1166.0*nu - 328.0*z19 - 171.0)) + z61*(-1.21875*z58*(2.0*nu - 3.0) + 0.015625*z63*(-27.0*nu + 28.0*z19 - 3.0) - 0.015625*(z62 + 9.0)*(25.0*z2 + 25.0*z5 + 25.0*z6)) + 7680.0*z72
        cdef double z74 = prst**4
        cdef double z75 = prst**6
        cdef double z76 = prst**8
        cdef double z77 = nu*z76
        cdef double z78 = z15 + z8
        cdef double z79 = nu*r
        cdef double z80 = nu*z15
        cdef double z81 = r*z19
        cdef double z82 = 14700.0*nu + 42911.0
        cdef double z83 = r*z82
        cdef double z84 = z15*z19
        cdef double z85 = 283115520.0*z82
        cdef double z86 = z65*(-25392914995744.3*nu - r*z85 - 879923036160.0*z15 - 5041721180160.0*z19 + 104186110149937.0)
        cdef double z87 = r*z66
        cdef double z88 = z15*(-630116198.873299*nu - 197773496.793534*z19 + 5805304367.87913)
        cdef double z89 = z0*(-2675575.66847905*nu - 138240.0*z19 - 5278341.3229329)
        cdef double z90 = nu*(-2510664218.28128*nu - 42636451.6032331*z19 + 14515200.0*z67 + 1002013764.01019)
        cdef double z91 = r*(43393301259014.8*nu + 43133561885859.3*z19 + 5927865218923.02*z67 + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0)
        cdef double z92 = z65*(-1698693120.0*nu*(11592.0*nu + 69847.0) + 49152.0*r*(409207698.136075*nu + 102574080.0*z19 - 2119671837.36038) + 879923036160.0*z0 + z15*z85)
        cdef double z93 = 4.0*z19
        cdef double z94 = 0.000130208333333333*z49 + z72
        cdef double z95 = -12049908701745.2*nu - 39476764256925.6*r + 5107745331375.71*z15 + 80059249540278.2*z19 + 5787938193408.0*z66 + 6730497718123.02*z67 + 10611661054566.2*z79 + 2589101062873.81*z80 + 1822680546449.21*z81 - 326837426.241486*z83 + 133772083200.0*z84 - z86 + 275059053208689.0
        cdef double z96 = 5787938193408.0*z87 - 9216.0*z88 - 967680.0*z89 + 55296.0*z90 + z91 + z92


        # Evaluate Hamiltonian
        cdef double H
        H,_,_,_,_,_,_,_,_ = self._call(q,p,chi1_v,chi2_v,m_1,m_2,chi_1,chi_2,chiL1,chiL2)

        # Heff Jacobian expressions

        cdef double dHeffdpphi = z73*(z73*(147.443752990146*nu*r**(-4.5)*z74 - 11.3175085791863*nu*r**(-3.5)*z75 + 1.69542100694444e-8*prst**2*z61*z78**2*z96*(z10*(-2.625*z58 + 0.0625*z63*(z59 - 1.0) + z8*(3.0*nu + 2.8125)) + z24*(0.015625*z52*(115.0*nu + z93 - 37.0) + 0.03125*z58*(26.0*nu + 449.0) + 0.015625*z8*(-1171.0*nu - 861.0)) + z49 + 7680.0*z60*z71*z95/z96)/(z94**2*z95) + 1.48275342024365*r**(-2.5)*z77 + z10*z74*(92.7110442849544*nu - 131.0*z19 + 10.0*z67) + z10*z75*(-33.9782122170436*nu - 89.5298327361234*z19 - 14.0*z64 + 188.0*z67) + z16*z74*(8.0*nu - 6.0*z19) + z16*z75*(-2.78300763695006*nu - 5.4*z19 + 6.0*z67) + z16*z76*(1.38977750996128*nu + 3.33842023648322*z19 - 6.0*z64 + 3.42857142857143*z67) + z24*z74*(nu*(452.542166996693 - 51.6952380952381*z65) + z19*(118.4*z65 - 1796.13660498019) + 602.318540416564*z67) - z31*z39*z51 + 0.121954868780449*z39*z77 + z40*z57 + 1.0 + 1.27277314139085e-19*z74*z78**4*(0.3125*delta*z47*(18.0*nu - 1.0) - 0.03125*(-33.0*nu + 32.0*z19 - 5.0)*(5.0*z2 + 5.0*z5 + 5.0*z6) - 0.03125*(-5.0*nu + z93 + 1.0)*(-15.0*z2 + 15.0*z5 + 15.0*z6))*(z87 - 1.59227685093395e-9*z88 - 1.67189069348064e-7*z89 + 9.55366110560367e-9*z90 + 1.72773095804465e-13*z91 + 1.72773095804465e-13*z92)**2/(r**13*z94**4*(-0.0438084424460039*nu - 0.143521050466841*r + 0.0185696317637669*z15 + 0.291062041428379*z19 + 0.0210425293255724*z66 + 0.0244692826489756*z67 + 0.0385795738434214*z79 + 0.00941289164152486*z80 + 0.00662650629087394*z81 - 1.18824456940711e-6*z83 + 0.000486339502879429*z84 - 3.63558293513537e-15*z86 + 1)**2))/(z49*(z50 + 1.0) + 1.0))**(-0.5)*(2.0*pphi*z16*z57 - pphi*z50*z51)/(z49*(4.0*z39 + 2.0) + 2.0) + (nu*dSO*z10*z13 + pphi*z13*(z14*z18 + z20*z21 + z22*z25) + pphi*z30*(z18*z26 + z21*z27 + z25*z28) + 2.0*z13*z45 + z13*(z16*(0.208333333333333*delta*z47 - 0.625*z2 + 0.375*z35 - 0.833333333333333*z48 - 0.625*z5 - 0.625*z6) + z45) + z13*(z14*z40 + z16*(-5.53125*nu + 0.567708333333333*z19 - 0.078125) + z20*z41 + z22*z42 + z39*(0.71875*nu - 0.09375) + 1.75) + z29*(-0.125*delta*z38 + z16*(0.0416666666666667*delta*z8 + z46*(0.416666666666667 - 0.833333333333333*X_2))) - 0.25*z30*z38 + z30*(z16*(-0.03125*nu + 0.536458333333333*z19 + 0.078125) + z26*z40 + z27*z41 + z28*z42 + z39*(0.34375*nu + 0.09375) + 0.25))/(z0 + z9)

        # Compute H Jacobian

        cdef double omega = M * M * dHeffdpphi / (nu*H)

        return omega

    cpdef auxderivs(self, double[:]q,double[:]p,double[:]chi1_v,double[:]chi2_v,double m_1,double m_2,double chi_1,double chi_2,double chiL1,double chiL2):

        """
        Compute derivatives of the potentials which are used in the post-adiabatic approximation.

        Args:
          q (double[:]): Canonical positions (r,phi).
          p (double[:]): Canonical momenta  (prstar,pphi).
          chi1_v (double[:]): Dimensionless spin vector of the primary.
          chi2_v (double[:]): Dimensionless spin vector of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.
          chi_1 (double): Projection of chi1_v onto the Newtonian orbital angular momentum unit vector (lN).
          chi_2 (double): Projection of chi2_v onto the Newtonian orbital angular momentum unit vector (lN).
          chiL1 (double): Projection of chi1_v onto the orbital angular momentum unit vector (l).
          chiL2 (double): Projection of chi2_v onto the orbital angular momentum unit vector (l).

        Returns:
           (tuple) dAdr, dBnpdr, dBnpadr, dxidr, dQdr, dQdprst, dHodddr, dBpdr, dHevendr

        """

        # Coordinate definitions

        cdef double r = q[0]
        cdef double phi = q[1]

        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double chix1 = chi1_v[0]
        cdef double chiy1 = chi1_v[1]
        cdef double chiz1 = chi1_v[2]

        cdef double chix2 = chi2_v[0]
        cdef double chiy2 = chi2_v[1]
        cdef double chiz2 = chi2_v[2]

        cdef double pphi = L

        cdef CalibCoeffs c_coeffs = self.calibration_coeffs
        cdef double a6 = c_coeffs['a6']
        cdef double dSO = c_coeffs['dSO']


        # Extra quantities used in the auxiliary derivatives
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double delta = self.EOBpars.p_params.delta
        cdef double w0 = r**(-1)
        cdef double w1 = r**2
        cdef double w2 = w1**(-1)
        cdef double w3 = X_1*(2.0*chix1*chix2 + 2.0*chiy1*chiy2 + 2.0*chiz1*chiz2)
        cdef double w4 = X_2*w3
        cdef double w5 = X_1**2
        cdef double w6 = chix1**2 + chiy1**2 + chiz1**2
        cdef double w7 = w5*w6
        cdef double w8 = X_2**2*(chix2**2 + chiy2**2 + chiz2**2)
        cdef double w9 = w7 + w8
        cdef double w10 = w4 + w9
        cdef double w11 = w10*w2
        cdef double w12 = w11*(2.0*w0 + 1.0) + 1.0
        cdef double w13 = r**4
        cdef double w14 = w13**(-1)
        cdef double w15 = 2.0*w4 + 2.0*w7 + 2.0*w8
        cdef double w16 = -w15
        cdef double w17 = r**3
        cdef double w18 = w17**(-1)
        cdef double w19 = r**5
        cdef double w20 = w19**(-1)
        cdef double w21 = chi_1*X_1
        cdef double w22 = chi_2*X_2
        cdef double w23 = w21 + w22
        cdef double w24 = w21 - w22
        cdef double w25 = -w23*w24 + w5*w6 - w8
        cdef double w26 = delta*w25
        cdef double w27 = w26*(68.0*nu - 1.0)
        cdef double w28 = nu**2
        cdef double w29 = 264.0*w28
        cdef double w30 = w23**2
        cdef double w31 = w10 - w30
        cdef double w32 = 0.00260416666666667*w31
        cdef double w33 = -w4 + w9
        cdef double w34 = -w24**2 + w33
        cdef double w35 = 0.0078125*w34
        cdef double w36 = w7 - w8
        cdef double w37 = delta*w36
        cdef double w38 = w37*(2.0*nu - 3.0)
        cdef double w39 = 7.0*nu
        cdef double w40 = 0.390625*w4 + 0.390625*w7 + 0.390625*w8
        cdef double w41 = 3.0*w4
        cdef double w42 = 3.0*w7 + 3.0*w8
        cdef double w43 = -w41 + w42
        cdef double w44 = w31*(w39 - 31.0)
        cdef double w45 = 4.0*nu
        cdef double w46 = w34*(-w45 - 3.0)
        cdef double w47 = w33*(w45 + 1.0)
        cdef double w48 = w18*w31
        cdef double w49 = log(r)
        cdef double w50 = 756.0*nu
        cdef double w51 = 336.0*r + w50 + 407.0
        cdef double w52 = 2048.0*nu*w49*w51 + 28.0*nu*(1920.0*a6 + 733955.307463037) - 7.0*r*(938918.400156317*nu - 185763.092693281*w28 - 245760.0) - 5416406.59541186*w28 - 3440640.0
        cdef double w53 = nu**4
        cdef double w54 = w49**2
        cdef double w55 = w28*w54
        cdef double w56 = nu**3
        cdef double w57 = w56*(-163683964.822551*r - 17833256.898555*w1 - 1188987459.03162)
        cdef double w58 = w28*(-39321600.0*a6*(3.0*r + 59.0) + 745857848.115604*a6 + 122635399361.987*r + 2064783811.32587*w1 - 3089250703.76879*w13 - 6178501407.53758*w17 + 1426660551.8844*w19 + 276057889687.011)
        cdef double w59 = 588.0*nu + 1079.0
        cdef double w60 = w50 + 1079.0
        cdef double w61 = w17*w60
        cdef double w62 = -38842241.4769507*nu + 240.0*r*(-7466.27061066206*nu - 3024.0*w28 + 17264.0) + 1920.0*w1*w59 + 480.0*w13*w60 + 161280.0*w19 - 1882456.23663972*w28 + 960.0*w61 + 13447680.0
        cdef double w63 = nu*w49
        cdef double w64 = w62*w63
        cdef double w65 = 8.0*r
        cdef double w66 = 2.0*w17
        cdef double w67 = 4.0*w1 + w65 + w66 + 16.0
        cdef double w68 = 7680.0*a6
        cdef double w69 = 128.0*r
        cdef double w70 = 7704.0*r
        cdef double w71 = 148.04406601634*r
        cdef double w72 = 113485.217444961*r
        cdef double w73 = nu*(w68*(w13 + w67) + w69*(13218.7851094412*r + 8529.39255472061*w1 - 6852.34813868015*w13 + 4264.6962773603*w17 - 33722.4297811176) + w71*(3852.0*w1 + 349.0*w13 + 1926.0*w17 + w70 + 36400.0) + w72*(-w13 + w67))
        cdef double w74 = (13212057600.0*w19 + 241555486248.807*w53 + 67645734912.0*w55 + 1120.0*w57 + 7.0*w58 + 32768.0*w64 + 53760.0*w73)**(-1)
        cdef double w75 = w13*w74
        cdef double w76 = w52*w75
        cdef double w77 = w11 + 7680.0*w76
        cdef double w78 = w14*(2.0625*w26 + 0.125*w44 + 0.0625*w46) + w14*(-1.25*w37 + 1.125*w4 + 0.125*w47 + 1.125*w7 + 1.125*w8) + w20*(0.125*w27 + w32*(2870.0*nu - w29 + 561.0) + w35*(1166.0*nu - 328.0*w28 - 171.0)) + w20*(-1.21875*w38 - w40*(w39 + 9.0) + 0.015625*w43*(-27.0*nu + 28.0*w28 - 3.0)) + w48 + w77
        cdef double w79 = w78*(w10*w18*(-4.0*w0 - 2.0) + w14*w16)/w12**2
        cdef double w80 = w12**(-1)
        cdef double w81 = 4.5*X_2
        cdef double w82 = r**(-6)
        cdef double w83 = nu*w0
        cdef double w84 = -6572428.80109422*nu + 1300341.64885296*w28 + 2048.0*w51*w83 + 688128.0*w63 + 1720320.0
        cdef double w85 = w0*w28
        cdef double w86 = 4.0*w17
        cdef double w87 = 6.0*w1 + w65 + 8.0
        cdef double w88 = 1.31621673590926e-19*w52*(53760.0*nu*(3740417.71815805*r + 2115968.85907902*w1 - 938918.400156317*w13 + 1057984.42953951*w17 + w68*(w86 + w87) + w69*(17058.7851094412*r + 12794.0888320809*w1 - 27409.3925547206*w17 + 13218.7851094412) + w71*(5778.0*w1 + 1396.0*w17 + w70 + 7704.0) + w72*(-w86 + w87) + 2888096.47013111) + 66060288000.0*w13 + 7.0*w28*(-117964800.0*a6 + 4129567622.65173*r - 18535504222.6128*w1 + 7133302759.42198*w13 - 12357002815.0752*w17 + 122635399361.987) + 135291469824.0*w49*w85 + 7.0*w56*(-5706642207.53758*r - 26189434371.6082) + 32768.0*w62*w83 + 32768.0*w63*(-1791904.9465589*nu + 3840.0*r*w59 + 2880.0*w1*w60 + 806400.0*w13 - 725760.0*w28 + 1920.0*w61 + 4143360.0))/(0.0546957463279941*w19 + w53 + 0.28004222119933*w55 + 4.63661586574928e-9*w57 + 2.8978849160933e-11*w58 + 1.35654132757922e-7*w64 + 2.22557561555966e-7*w73)**2
        cdef double w89 = w13*w88
        cdef double w90 = w80*(7680.0*w13*w74*w84 - w14*(-3.0*w30 + w41 + w42) + w16*w18 + 30720.0*w17*w52*w74 - w20*(8.25*w26 + 0.5*w44 + 0.25*w46) - w20*(w3*w81 - 5.0*w37 + 0.5*w47 + 4.5*w7 + 4.5*w8) - w82*(0.625*w27 + w32*(14350.0*nu - 1320.0*w28 + 2805.0) + w35*(5830.0*nu - 1640.0*w28 - 855.0)) - w82*(-6.09375*w38 - w40*(35.0*nu + 45.0) + 0.015625*w43*(-135.0*nu + 140.0*w28 - 15.0)) - w89)
        cdef double w91 = w16*w18
        cdef double w92 = 0.0625*w43
        cdef double w93 = w37*(26.0*nu + 449.0)
        cdef double w94 = w10*(-1171.0*nu - 861.0)
        cdef double w95 = 4.0*w28
        cdef double w96 = w33*(115.0*nu + w95 - 37.0)
        cdef double w97 = 5787938193408.0*w54
        cdef double w98 = -630116198.873299*nu - 197773496.793534*w28 + 5805304367.87913
        cdef double w99 = w1*w98
        cdef double w100 = -2675575.66847905*nu - 138240.0*w28 - 5278341.3229329
        cdef double w101 = w100*w17
        cdef double w102 = nu*(-2510664218.28128*nu - 42636451.6032331*w28 + 14515200.0*w56 + 1002013764.01019)
        cdef double w103 = 43393301259014.8*nu + 43133561885859.3*w28 + 5927865218923.02*w56 + 86618264430493.3*(1 - 0.496948781616935*nu)**2 + 188440788778196.0
        cdef double w104 = r*w103
        cdef double w105 = 14700.0*nu + 42911.0
        cdef double w106 = 283115520.0*w105
        cdef double w107 = -1698693120.0*nu*(11592.0*nu + 69847.0) + 49152.0*r*(409207698.136075*nu + 102574080.0*w28 - 2119671837.36038) + w1*w106 + 879923036160.0*w17
        cdef double w108 = w107*w49
        cdef double w109 = r*w97 - 967680.0*w101 + 55296.0*w102 + w104 + w108 - 9216.0*w99
        cdef double w110 = w109**(-1)
        cdef double w111 = nu*r
        cdef double w112 = 1822680546449.21*w28
        cdef double w113 = r*w28
        cdef double w114 = 11575876386816.0*w49
        cdef double w115 = 5041721180160.0*w28 - 104186110149937.0
        cdef double w116 = -25392914995744.3*nu - r*w106 - 879923036160.0*w1 - w115
        cdef double w117 = 5807150888816.34*nu + 10215490662751.4*r + w0*w114 - w0*w116 + 5178202125747.62*w111 + w112 + 267544166400.0*w113 + w49*(4161798144000.0*nu + 1759846072320.0*r + 12148770078720.0) - 53501685054374.1
        cdef double w118 = w110*w117
        cdef double w119 = w52*w74
        cdef double w120 = 7680.0*w119
        cdef double w121 = 7680.0*w84
        cdef double w122 = nu*w1
        cdef double w123 = r*w105
        cdef double w124 = w1*w28
        cdef double w125 = w116*w49
        cdef double w126 = -12049908701745.2*nu + r*w112 - 39476764256925.6*r + 5107745331375.71*w1 + 10611661054566.2*w111 + 2589101062873.81*w122 - 326837426.241486*w123 + 133772083200.0*w124 - w125 + 80059249540278.2*w28 + 6730497718123.02*w56 + w97 + 275059053208689.0
        cdef double w127 = w110*w126
        cdef double w128 = w127*w19
        cdef double w129 = (r*w54 - 1.67189069348064e-7*w101 + 9.55366110560367e-9*w102 + 1.72773095804465e-13*w104 + 1.72773095804465e-13*w108 - 1.59227685093395e-9*w99)**2
        cdef double w130 = -18432.0*r*w98 + w0*w107 - 2903040.0*w1*w100 + w103 + w114 + w49*(20113376778784.3*nu + 2639769108480.0*w1 + w115 + 566231040.0*w123) + w97
        cdef double w131 = w126*w130/w129
        cdef double w132 = w118*w120*w19 - 2.29252167428035e-22*w119*w131*w19 + w121*w128*w74 + 38400.0*w127*w76 - w128*w88 - w14*(w10*(9.0*nu + 8.4375) - 7.875*w37 + w92*(12.0*nu - 3.0)) - w20*(0.125*w93 + 0.0625*w94 + 0.0625*w96) + w91
        cdef double w133 = r + 2.0
        cdef double w134 = r*w10
        cdef double w135 = w13 + w133*w134
        cdef double w136 = w135**(-1)
        cdef double w137 = w133*w136
        cdef double w138 = w135**(-2)
        cdef double w139 = w10*w133
        cdef double w140 = w134 + w139 + w86
        cdef double w141 = sqrt(r*w127)
        cdef double w142 = w1 + w10
        cdef double w143 = w141/w142
        cdef double w144 = w142**2
        cdef double w145 = 30720.0*w119*w17 + w121*w75 - w89 + w91
        cdef double w146 = prst**4
        cdef double w147 = r**(-4.5)
        cdef double w148 = prst**6
        cdef double w149 = r**(-3.5)
        cdef double w150 = prst**8
        cdef double w151 = nu*w150
        cdef double w152 = 8.0*nu - 6.0*w28
        cdef double w153 = w146*w18
        cdef double w154 = 92.7110442849544*nu - 131.0*w28 + 10.0*w56
        cdef double w155 = w14*w146
        cdef double w156 = -2.78300763695006*nu - 5.4*w28 + 6.0*w56
        cdef double w157 = w148*w18
        cdef double w158 = -33.9782122170436*nu - 89.5298327361234*w28 - 14.0*w53 + 188.0*w56
        cdef double w159 = 3.0*w14
        cdef double w160 = 1.38977750996128*nu + 3.33842023648322*w28 - 6.0*w53 + 3.42857142857143*w56
        cdef double w161 = w150*w160
        cdef double w162 = nu*(452.542166996693 - 51.6952380952381*w49) + w28*(118.4*w49 - 1796.13660498019) + 602.318540416564*w56
        cdef double w163 = 4.0*w162
        cdef double w164 = w142**4
        cdef double w165 = 0.000130208333333333*w11 + w76
        cdef double w166 = w165**(-4)
        cdef double w167 = -0.0438084424460039*nu - 0.143521050466841*r + 0.0185696317637669*w1 + 0.0385795738434214*w111 + 0.00662650629087394*w113 + 0.00941289164152486*w122 - 1.18824456940711e-6*w123 + 0.000486339502879429*w124 - 3.63558293513537e-15*w125 + 0.291062041428379*w28 + 0.0210425293255724*w54 + 0.0244692826489756*w56 + 1
        cdef double w168 = w167**(-2)
        cdef double w169 = 0.3125*delta*w36*(18.0*nu - 1.0) - 0.03125*(-33.0*nu + 32.0*w28 - 5.0)*(5.0*w4 + 5.0*w7 + 5.0*w8) - 0.03125*(-5.0*nu + w95 + 1.0)*(-15.0*w4 + 15.0*w7 + 15.0*w8)
        cdef double w170 = w129*w166*w168*w169
        cdef double w171 = w146*w164*w170
        cdef double w172 = r**(-13)
        cdef double w173 = w164*w172
        cdef double w174 = 663.496888455656*nu*r**(-5.5)*w146 - 39.6112800271521*nu*w147*w148 - 7.59859378406358e-45*w109*w130*w146*w164*w166*w168*w169*w172 + 9.25454462627843e-34*w117*w129*w146*w166*w169*w173/w167**3 + 6.62902677807736e-23*w129*w145*w146*w168*w169*w173/w165**5 - w14*w146*(-51.6952380952381*w83 + 118.4*w85) + w146*w163*w20 + w148*w158*w159 + 3.70688355060912*w149*w151 + 0.121954868780449*w151*w2 + 2.0*w152*w153 + 3.0*w154*w155 + 2.0*w156*w157 + 2.0*w161*w18 - 1.01821851311268e-18*w129*w142**3*w146*w166*w168*w169/r**12 + 1.65460508380811e-18*w171/r**14
        cdef double w175 = prst**3
        cdef double w176 = nu*w147
        cdef double w177 = prst**5
        cdef double w178 = nu*w149
        cdef double w179 = r**(-2.5)
        cdef double w180 = prst**7
        cdef double w181 = w152*w2
        cdef double w182 = 4.0*w175
        cdef double w183 = w156*w2
        cdef double w184 = 6.0*w177
        cdef double w185 = w139 + w17
        cdef double w186 = 3.0*nu
        cdef double w187 = X_1*chiL1
        cdef double w188 = X_2*chiL2
        cdef double w189 = w187 + w188
        cdef double w190 = pphi*w189
        cdef double w191 = dSO*w190
        cdef double w192 = 0.71875*nu - 0.09375
        cdef double w193 = -1.40625*nu - 0.46875
        cdef double w194 = pphi**2
        cdef double w195 = w18*w194
        cdef double w196 = 2.0*w195
        cdef double w197 = -2.0859375*nu - 2.07161458333333*w28 + 0.23046875
        cdef double w198 = w159*w194
        cdef double w199 = 0.5859375*nu + 1.34765625*w28 + 0.41015625
        cdef double w200 = pphi**4
        cdef double w201 = 4.0*w20*w200
        cdef double w202 = 0.34375*nu + 0.09375
        cdef double w203 = 0.46875 - 0.28125*nu
        cdef double w204 = -0.2734375*nu - 0.798177083333333*w28 - 0.23046875
        cdef double w205 = -0.3515625*nu + 0.29296875*w28 - 0.41015625
        cdef double w206 = pphi*(w187 - w188)
        cdef double w207 = delta*w206
        cdef double w208 = delta*w194
        cdef double w209 = delta*w10
        cdef double w210 = w31*(0.416666666666667 - 0.833333333333333*X_2)
        cdef double w211 = 0.5*w25*(0.5 - X_2) + 0.125*w30 - 0.125*w4 - 0.125*w7 - 0.125*w8
        cdef double w212 = w194*w2
        cdef double w213 = w14*w200
        cdef double w214 = 0.5*w25
        cdef double w215 = 0.5*w31
        cdef double w216 = 0.5*w34
        cdef double w217 = w26*(98.0*nu + 43.0)
        cdef double w218 = -1610.0*nu + w29 + 375.0
        cdef double w219 = -1090.0*nu + 152.0*w28 + 219.0
        cdef double w220 = -w14*(w214*(2.25 - w81) + w215*(-5.25*nu - 2.8125) + w216*(0.5625 - 2.25*nu)) + w18*w31 - w20*(-0.25*w217 + 0.0104166666666667*w218*w31 + 0.03125*w219*w34)
        cdef double w221 = w189**2
        cdef double w222 = w0*w194*w221
        cdef double w223 = w14*(-0.0625*w217 + w218*w32 + w219*w35) + w18*(w214*(0.75 - 1.5*X_2) + w215*(-1.75*nu - 0.9375) + w216*(0.1875 - 0.75*nu)) - w2*w215 + 1.0
        cdef double w224 = prst**2
        cdef double w225 = w165**(-2)
        cdef double w226 = w126**(-1)
        cdef double w227 = w11 + w120*w128 + w14*(0.03125*w93 + 0.015625*w94 + 0.015625*w96) + w18*(w10*(w186 + 2.8125) - 2.625*w37 + w92*(w45 - 1.0))
        cdef double w228 = w109*w144*w224*w225*w226*w227
        cdef double w229 = -w137*w222 + 147.443752990146*w146*w176 + w146*w181 - 11.3175085791863*w148*w178 + w148*w183 + 0.121954868780449*w150*w83 + 1.48275342024365*w151*w179 + w153*w154 + w155*w162 + w157*w158 + w161*w2 + 1.27277314139085e-19*w171*w172 + 1.69542100694444e-8*w20*w228 + w212*w223 + 1.0
        cdef double w230 = w78*w80

        cdef double dAdr = -w79 + w90
        cdef double dBnpdr = w132
        cdef double dBnpadr = r*w133*w138*w140 - r*w136 - w137
        cdef double dxidr = 2.0*r*w143*w77 + w1*w143*w145 + w1*w77*(r*w118 - 2.98505426338587e-26*r*w131 + w127)/(w141*(2.0*w1 + w15)) - w141*w66*w77/w144
        cdef double dQdr = -w174
        cdef double dQdprst = 11.8620273619492*nu*w179*w180 + w14*w163*w175 + w154*w18*w182 + w158*w18*w184 + 8.0*w160*w180*w2 + 5.09109256556341e-19*w170*w173*w175 + 589.775011960583*w175*w176 - 67.9050514751178*w177*w178 + 0.975638950243592*w180*w83 + w181*w182 + w183*w184
        cdef double dHodddr = (-w14*w186*w191 + w190*(w18*(1.66666666666667*w26 - 0.75*w30 - 0.416666666666667*w37 + 1.25*w4 + 1.25*w7 + 1.25*w8) - w198*w211) + w190*(-w18*(-11.0625*nu + 1.13541666666667*w28 - 0.15625) - w192*w2 - w193*w196 - w197*w198 - w199*w201) + w206*(0.375*w14*w208*w31 - w18*(0.0833333333333333*w209 + w210)) + w207*(-w18*(-0.0625*nu + 1.07291666666667*w28 + 0.15625) - w196*w203 - w198*w204 - w2*w202 - w201*w205))/w185 - (3.0*w1 + w10)*(nu*w18*w191 + w190*(w195*w211 + w2*(0.208333333333333*delta*w36 - 0.833333333333333*w26 + 0.375*w30 - 0.625*w4 - 0.625*w7 - 0.625*w8)) + w190*(w0*w192 + w193*w212 + w195*w197 + w199*w213 + w2*(-5.53125*nu + 0.567708333333333*w28 - 0.078125) + 1.75) + w206*(w2*(0.0416666666666667*w209 + 0.5*w210) - 0.125*w208*w48) + w207*(w0*w202 + w195*w204 + w2*(-0.03125*nu + 0.536458333333333*w28 + 0.078125) + w203*w212 + w205*w213 + 0.25))/w185**2
        cdef double dBpdr = w220
        cdef double dHevendr = 0.5*(w229*w230)**(-0.5)*(-w229*w79 + w229*w90 + w230*(w0*w133*w138*w140*w194*w221 - 2.24091649004576e-37*w109*w117*w144*w168*w20*w224*w225*w227 + 1.69542100694444e-8*w109*w132*w144*w20*w224*w225*w226 + 6.78168402777778e-8*w109*w14*w142*w224*w225*w226*w227 - 4.41515887225116e-12*w109*w144*w145*w20*w224*w226*w227/w165**3 + 1.69542100694444e-8*w130*w144*w20*w224*w225*w226*w227 + w133*w136*w194*w2*w221 - w136*w222 - w174 + w194*w2*w220 - w196*w223 - 8.47710503472222e-8*w228*w82))

        return dAdr, dBnpdr, dBnpadr, dxidr, dQdr, dQdprst, dHodddr, dBpdr, dHevendr
