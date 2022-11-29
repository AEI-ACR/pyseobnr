
# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

cimport cython

from libc.math cimport log, sqrt, pi

DEF euler_gamma=0.5772156649015329


"""
Coefficients entering the evolution equation for S1dot to 4PN order
"""
cdef class s1dotCoeffs:
    @cython.cdivision(True)
    @cython.embedsignature(True)
    def __cinit__(self, double nu, double delta):

        # coefficients entering s1dot
        self.asdotv110622 = nu*(103*nu + 417)/144. - 9/4.
        self.asdotv110621 = nu*(nu*(121. + 2310./(delta + 1)) - 819. + 486./(delta - 2.*nu + 1.) - 1746./(delta + 1.))/144.
        self.asdotv11061 = -nu*(nu + 147)/48. - 3/8.
        self.asdotv1950 = -delta*(nu*(5*nu - 156) + 27)/32. + nu*(-nu*(2*nu + 315) + 18)/96. + 27/32.
        self.asdotv18422 = nu*(-17. + 54./(delta - 2*nu + 1) - 90./(delta + 1))/12.
        self.asdotv18421 = nu/12. - 1/2.
        self.asdotv1841 = nu/4.
        self.asdotv1730 = delta*(10*nu - 9)/16. - nu*(2*nu - 60)/48. + 9/16.
        self.asdotv16222 = -3.*nu/(delta - 2*nu + 1)
        self.asdotv16221 = -3/2.
        self.asdotv1621 = -1/2.
        self.asdotv1510 = -3*delta/4. + nu/2. + 3/4.





"""
Coefficients entering the evolution equation for S2dot to 4PN order
"""
cdef class s2dotCoeffs:
    @cython.cdivision(True)
    @cython.embedsignature(True)
    def __cinit__(self, double nu, double delta):


        # coefficients entering s2dot
        self.asdotv210622 = nu*(delta*(121*nu + 336) - 2189.*nu + 2082.)/(144.*(delta + 1.)) - 31/4. + (243*delta + 243)/(288.*nu)
        self.asdotv210621 = nu*(103*nu + 417)/144. - 9/4.
        self.asdotv21061 = nu*(nu + 147)/48. + 3/8.
        self.asdotv2950 = delta*(nu*(5*nu - 156) + 27)/32. + nu*(-nu*(2.*nu + 315.) + 18)/96. + 27/32.
        self.asdotv28422 = nu*(73 - 17*delta)/(12.*delta + 12.) - 6. + (9.*delta + 9)/(8.*nu)
        self.asdotv28421 = nu/12. - 1/2.
        self.asdotv2841 = -nu/4.
        self.asdotv2730 = delta*(27 - 30*nu)/48. - nu*(2*nu - 60)/48. + 9/16.
        self.asdotv26222 = (-3*delta + 6*nu - 3)/(4.*nu)
        self.asdotv26221 = -3/2.
        self.asdotv2621 = 1/2.
        self.asdotv2510 = 3*delta/4. + nu/2. + 3/4.



"""
Coefficients entering the evolution equation for lNdot to 4PN order
"""
cdef class lNdotCoeffs:
    @cython.cdivision(True)
    @cython.embedsignature(True)
    def __cinit__(self, double nu, double delta):

        # coefficients entering lNdot
        self.alNdotv11642 = (delta*(nu*(439*nu - 192) - 279) - nu*(nu*(138*nu + 1381) + 366) + 279)/(96.*nu**2)
        self.alNdotv11641 = (-3*delta*(71*nu + 9) + nu*(1413 - 32*nu) + 207)/(192.*nu)
        self.alNdotv11632 = (delta*(nu*(192 - 439*nu) + 279) - nu*(nu*(138*nu + 1381) + 366) + 279)/(96.*nu**2)
        self.alNdotv11631 = (3*delta*(71*nu + 9) + nu*(1413 - 32*nu) + 207)/(192.*nu)
        self.alNdotv1162 = (-18*delta*nu - 9*delta)/(32.*nu)
        self.alNdotv1161 = delta*(89*nu - 27)/(96.*nu)
        self.alNdotv1052 = (9*delta*(7*nu**2 - 21*nu - 9) - 2*nu**3 + 423*nu**2 + 27*nu + 81)/(96.*nu)
        self.alNdotv1051 = (-9.*delta*(7*nu**2 - 21*nu - 9) - 2*nu**3 + 423*nu**2 + 27*nu + 81)/(96.*nu)
        self.alNdotv9442 = 2. + (3*delta*(nu - 3) + 21*nu - 9)/(2.*nu**2)
        self.alNdotv9441 = (-15*delta/16. - 5*nu/4. - 99/16.)/nu
        self.alNdotv9432 = (-3*delta*(nu - 3) + nu*(4*nu + 21) - 9)/(2.*nu**2)
        self.alNdotv9431 = -5/4. + (15*delta - 99)/(16.*nu)
        self.alNdotv942 = 3*delta/(8.*nu)
        self.alNdotv941 = -5*delta/(8.*nu)
        self.alNdotv832 = (9*delta*(nu + 1) + 2*nu**2 - 9*nu + 9)/(8.*nu)
        self.alNdotv831 = (-9*delta*(nu + 1) + 2*nu**2 - 9*nu + 9)/(8.*nu)
        self.alNdotv7222 = (-3*delta - 6*nu + 3)/(4.*nu**2)
        self.alNdotv7221 = 3./(2*nu)
        self.alNdotv7212 = 3./(2*nu)
        self.alNdotv7211 = (3.*delta - 6.*nu + 3.)/(4.*nu**2)
        self.alNdotv612 = (-3*delta/4. - nu/2. - 3/4.)/nu
        self.alNdotv611 = (3*delta/4. - nu/2. - 3/4.)/nu




"""
Coefficients entering the evolution equation for omegadot to 4PN order
"""
cdef class omegadotCoeffs:
    @cython.cdivision(True)
    @cython.embedsignature(True)
    def __cinit__(self, double nu, double delta):

        # coefficients entering omegadot
        self.aomegadotv8448 = 138323.*delta*nu/(6912*delta + 6912)
        self.aomegadotv8438 = 138323.*delta*nu/(6912*delta + 6912)
        self.aomegadotv8428 = 34303.*pi*delta*nu/(336*delta + 336)
        self.aomegadotv8418 = 34303.*pi*delta*nu/(336*delta + 336)
        self.aomegadotv8408 = -869429.*delta*nu/(6912*delta + 6912)
        self.aomegadotv8398 = -869429.*delta*nu/(6912*delta + 6912)
        self.aomegadotv8388 = -121021.*nu/(2304*delta + 2304)
        self.aomegadotv8378 = 639709.*nu/(6912*delta + 6912)
        self.aomegadotv8368 = -58573.*pi*nu/(168*delta + 168)
        self.aomegadotv8358 = 3317.*pi*nu/(6*delta + 6)
        self.aomegadotv8348 = 1305019.*nu/(2304*delta + 2304)
        self.aomegadotv8338 = -5653915.*nu/(6912*delta + 6912)
        self.aomegadotv8328 = -162541*nu/3456.
        self.aomegadotv8318 = 33163*nu/3456.
        self.aomegadotv8308 = -3431557.*delta/(nu*(36288*delta + 36288))
        self.aomegadotv8298 = 8207303.*delta/(nu*(145152*delta + 145152))
        self.aomegadotv8288 = -1665.*pi*delta/(nu*(28*delta + 28))
        self.aomegadotv8278 = 75409.*delta/(nu*(12096*delta + 12096))
        self.aomegadotv8268 = -11888267.*delta/(nu*(48384*delta + 48384))
        self.aomegadotv8258 = 64009./(nu*(3456*delta + 3456))
        self.aomegadotv8248 = -8207303./(nu*(145152*delta + 145152))
        self.aomegadotv8238 = -1665.*pi/(nu*(28*delta + 28))
        self.aomegadotv8228 = -1304161./(nu*(2688*delta + 2688))
        self.aomegadotv8218 = 11888267./(nu*(48384*delta + 48384))
        self.aomegadotv8208 = -9355721./(72576*nu)
        self.aomegadotv8198 = 21001565./(24192*nu)
        self.aomegadotv8188 = -8207303.*delta/(nu**2*(145152*delta + 145152))
        self.aomegadotv8178 = 11888267.*delta/(nu**2*(48384*delta + 48384))
        self.aomegadotv8168 = -8207303./(nu**2*(145152*delta + 145152))
        self.aomegadotv8158 = 11888267./(nu**2*(48384*delta + 48384))
        self.aomegadotv8148 = 2039635.*delta/(12096*delta + 12096)
        self.aomegadotv8138 = 711521.*delta/(5376*delta + 5376)
        self.aomegadotv8128 = 266519.*pi*delta/(2016*delta + 2016)
        self.aomegadotv8118 = -46957.*pi*delta/(504*delta + 504)
        self.aomegadotv8108 = -615605.*delta/(12096*delta + 12096)
        self.aomegadotv898 = 14283281.*delta/(48384*delta + 48384)
        self.aomegadotv888 = 11390447./(24192*delta + 24192)
        self.aomegadotv878 = -43485./(256*delta + 256)
        self.aomegadotv868 = 506279.*pi/(2016*delta + 2016)
        self.aomegadotv858 = -15271.*pi/(72*delta + 72)
        self.aomegadotv848 = -13421113./(24192*delta + 24192)
        self.aomegadotv838 = 38663087./(48384*delta + 48384)
        self.aomegadotv828 = -195697/896.
        self.aomegadotv818 = -10150387/24192.
        self.aomegadotv7317 = -10819.*delta*nu**2/(432*delta + 432)
        self.aomegadotv7307 = -10819.*delta*nu**2/(432*delta + 432)
        self.aomegadotv7297 = 33781.*nu**2/(216*delta + 216)
        self.aomegadotv7287 = -5575.*nu**2/(27*delta + 27)
        self.aomegadotv7277 = 91495.*pi*nu**2/1512
        self.aomegadotv7267 = 7081.*delta*nu/(144*delta + 144)
        self.aomegadotv7257 = 40289.*delta*nu/(288*delta + 288)
        self.aomegadotv7247 = -464479.*nu/(1008*delta + 1008)
        self.aomegadotv7237 = 436705.*nu/(672*delta + 672)
        self.aomegadotv7227 = 358675.*pi*nu/6048
        self.aomegadotv7217 = 6.*pi*delta/(nu*(delta + 1))
        self.aomegadotv7207 = 6.*pi*delta/(nu*(delta + 1))
        self.aomegadotv7197 = -209.*pi*delta/(nu*(8*delta + 8))
        self.aomegadotv7187 = -209.*pi*delta/(nu*(8*delta + 8))
        self.aomegadotv7177 = -1195759.*delta/(nu*(18144*delta + 18144))
        self.aomegadotv7167 = 18.*pi/(nu*(delta + 1))
        self.aomegadotv7157 = -6.*pi/(nu*(delta + 1))
        self.aomegadotv7147 = -627.*pi/(nu*(8*delta + 8))
        self.aomegadotv7137 = 209.*pi/(nu*(8*delta + 8))
        self.aomegadotv7127 = -1195759./(nu*(18144*delta + 18144))
        self.aomegadotv7117 = -12.*pi/nu
        self.aomegadotv7107 = 207.*pi/(4*nu)
        self.aomegadotv797 = -6.*pi*delta/(nu**2*(delta + 1))
        self.aomegadotv787 = 209.*pi*delta/(nu**2*(8*delta + 8))
        self.aomegadotv777 = -6.*pi/(nu**2*(delta + 1))
        self.aomegadotv767 = 209.*pi/(nu**2*(8*delta + 8))
        self.aomegadotv757 = 2694373.*delta/(18144*delta + 18144)
        self.aomegadotv747 = -1932041.*delta/(18144*delta + 18144)
        self.aomegadotv737 = 565099./(2016*delta + 2016)
        self.aomegadotv727 = -4323559./(18144*delta + 18144)
        self.aomegadotv717 = -4415*pi/4032.
        self.aomegadotvLog6 = -1712/105.
        self.aomegadotv6376 = -3424*log(2)/105.
        self.aomegadotv6366 = -5605*nu**3/2592.
        self.aomegadotv6356 = 541*nu**2/896.
        self.aomegadotv6346 = 451*pi**2*nu/48.
        self.aomegadotv6336 = -56198689.*nu/217728.
        self.aomegadotv6326 = -42383.*delta/(nu*(2016*delta + 2016))
        self.aomegadotv6316 = -8503.*delta/(nu*(448*delta + 448))
        self.aomegadotv6306 = 77813.*delta/(nu*(2016*delta + 2016))
        self.aomegadotv6296 = 2185.*delta/(nu*(448*delta + 448))
        self.aomegadotv6286 = -59455./(nu*(1008*delta + 1008))
        self.aomegadotv6276 = 8503./(nu*(448*delta + 448))
        self.aomegadotv6266 = 48739./(nu*(1008*delta + 1008))
        self.aomegadotv6256 = -2185./(nu*(448*delta + 448))
        self.aomegadotv6246 = 16255./(672*nu)
        self.aomegadotv6236 = -151.*pi/(6*nu)
        self.aomegadotv6226 = 14433./(224*nu)
        self.aomegadotv6216 = 8503.*delta/(nu**2*(448*delta + 448))
        self.aomegadotv6206 = -2185.*delta/(nu**2*(448*delta + 448))
        self.aomegadotv6196 = 8503./(nu**2*(448*delta + 448))
        self.aomegadotv6186 = -2185./(nu**2*(448*delta + 448))
        self.aomegadotv6176 = -6011.*delta/(576*delta + 576)
        self.aomegadotv6166 = -6011.*delta/(576*delta + 576)
        self.aomegadotv6156 = -37.*pi*delta/(3*delta + 3)
        self.aomegadotv6146 = 25373.*delta/(576*delta + 576)
        self.aomegadotv6136 = 25373.*delta/(576*delta + 576)
        self.aomegadotv6126 = -1219./(192*delta + 192)
        self.aomegadotv6116 = -8365./(576*delta + 576)
        self.aomegadotv6106 = 151.*pi/(3*delta + 3)
        self.aomegadotv696 = -188.*pi/(3*delta + 3)
        self.aomegadotv686 = -1497./(64*delta + 64)
        self.aomegadotv676 = 64219./(576*delta + 576)
        self.aomegadotv666 = 6373/288.
        self.aomegadotv656 = 16*pi**2/3.
        self.aomegadotv646 = -37*pi/3.
        self.aomegadotv636 = -11779/288.
        self.aomegadotv626 = -1712*euler_gamma/105.
        self.aomegadotv616 = 16447322263/139708800.
        self.aomegadotv5125 = 79.*delta*nu/(6*delta + 6)
        self.aomegadotv5115 = 79.*delta*nu/(6*delta + 6)
        self.aomegadotv5105 = -685.*nu/(12*delta + 12)
        self.aomegadotv595 = 1001.*nu/(12*delta + 12)
        self.aomegadotv585 = -189.*pi*nu/8
        self.aomegadotv575 = -809.*delta/(nu*(84*delta + 84))
        self.aomegadotv565 = -809./(nu*(84*delta + 84))
        self.aomegadotv555 = 13795.*delta/(1008*delta + 1008)
        self.aomegadotv545 = -21611.*delta/(1008*delta + 1008)
        self.aomegadotv535 = 33211./(1008*delta + 1008)
        self.aomegadotv525 = -5861./(144*delta + 144)
        self.aomegadotv515 = -4159.*pi/672
        self.aomegadotv4174 = 59.*nu**2/18
        self.aomegadotv4164 = 13661.*nu/2016
        self.aomegadotv4154 = 233.*delta/(nu*(96*delta + 96))
        self.aomegadotv4144 = 233.*delta/(nu*(96*delta + 96))
        self.aomegadotv4134 = -719.*delta/(nu*(96*delta + 96))
        self.aomegadotv4124 = -719.*delta/(nu*(96*delta + 96))
        self.aomegadotv4114 = 233./(nu*(32*delta + 32))
        self.aomegadotv4104 = -233./(nu*(96*delta + 96))
        self.aomegadotv494 = -719./(nu*(32*delta + 32))
        self.aomegadotv484 = 719./(nu*(96*delta + 96))
        self.aomegadotv474 = -247./(48*nu)
        self.aomegadotv464 = 721./(48*nu)
        self.aomegadotv454 = -233.*delta/(nu**2*(96*delta + 96))
        self.aomegadotv444 = 719.*delta/(nu**2*(96*delta + 96))
        self.aomegadotv434 = -233./(nu**2*(96*delta + 96))
        self.aomegadotv424 = 719./(nu**2*(96*delta + 96))
        self.aomegadotv414 = 34103/18144.
        self.aomegadotv363 = -25./(4*nu)
        self.aomegadotv353 = -19.*delta/(6*delta + 6)
        self.aomegadotv343 = -19.*delta/(6*delta + 6)
        self.aomegadotv333 = 56./(6*delta + 6)
        self.aomegadotv323 = -94./(6*delta + 6)
        self.aomegadotv313 = 4*pi
        self.aomegadotv222 = -11.*nu/4
        self.aomegadotv212 = -743/336.
        self.aomegadotv0 = 1.






"""
Coefficients entering the equation for Lhat to 4PN order
"""
cdef class lhatCoeffs:
    @cython.cdivision(True)
    @cython.embedsignature(True)
    def __cinit__(self, double nu, double delta):

        # coefficients entering lhat
        self.alhatvLog8 = -128.*nu/3
        self.alhatv8618 = -256.*nu*log(2)/3
        self.alhatv8608 = -55.*nu**4/31104
        self.alhatv8598 = -215.*nu**3/1728
        self.alhatv8588 = -2255.*pi**2*nu**2/576
        self.alhatv8578 = 356035.*nu**2/3456
        self.alhatv8568 = 5.*delta*nu/(144*delta + 144)
        self.alhatv8558 = -235.*delta*nu/(288*delta + 288)
        self.alhatv8548 = 5.*delta*nu/(144*delta + 144)
        self.alhatv8538 = -505.*delta*nu/(864*delta + 864)
        self.alhatv8528 = -505.*delta*nu/(864*delta + 864)
        self.alhatv8518 = -65.*nu/(16*delta + 16)
        self.alhatv8508 = 1799.*nu/(288*delta + 288)
        self.alhatv8498 = 595.*nu/(144*delta + 144)
        self.alhatv8488 = -113.*nu/(16*delta + 16)
        self.alhatv8478 = 16493.*nu/(864*delta + 864)
        self.alhatv8468 = -17503.*nu/(864*delta + 864)
        self.alhatv8458 = -223.*nu/288
        self.alhatv8448 = -5.*nu/72
        self.alhatv8438 = -223.*nu/288
        self.alhatv8428 = -235.*nu/288
        self.alhatv8418 = -6455.*pi**2*nu/1536
        self.alhatv8408 = -361.*nu/432
        self.alhatv8398 = -128.*euler_gamma*nu/3
        self.alhatv8388 = 98869.*nu/5760
        self.alhatv8378 = -125.*delta/(nu*(12*delta + 12))
        self.alhatv8368 = 509.*delta/(nu*(32*delta + 32))
        self.alhatv8358 = -125./(nu*(12*delta + 12))
        self.alhatv8348 = 509./(nu*(32*delta + 32))
        self.alhatv8338 = -287./(96*nu)
        self.alhatv8328 = 15./(8*nu)
        self.alhatv8318 = -5./(4*nu)
        self.alhatv8308 = 15./(8*nu)
        self.alhatv8298 = 15./(4*nu)
        self.alhatv8288 = 15.*delta**2/(nu**2*(32*delta + 32))
        self.alhatv8278 = -111.*delta**2/(nu**2*(64*delta + 64))
        self.alhatv8268 = 15.*delta/(nu**2*(16*delta + 16))
        self.alhatv8258 = -111.*delta/(nu**2*(32*delta + 32))
        self.alhatv8248 = 15./(nu**2*(32*delta + 32))
        self.alhatv8238 = -111./(nu**2*(64*delta + 64))
        self.alhatv8228 = 21.*delta/(64*nu**2)
        self.alhatv8218 = 21./(64*nu**2)
        self.alhatv8208 = 15./(8*delta - 16*nu + 8)
        self.alhatv8198 = 21./(16*delta - 32*nu + 16)
        self.alhatv8188 = -111./(16*delta - 32*nu + 16)
        self.alhatv8178 = 70.*delta/(9*delta + 9)
        self.alhatv8168 = 7.*delta/(3*delta + 3)
        self.alhatv8158 = -56.*delta/(9*delta + 9)
        self.alhatv8148 = 1925./(72*delta + 72)
        self.alhatv8138 = 7./(delta + 1)
        self.alhatv8128 = -455./(24*delta + 24)
        self.alhatv8118 = -14./(3*delta + 3)
        self.alhatv8108 = -2239./(72*delta + 72)
        self.alhatv898 = 199./(8*delta + 8)
        self.alhatv888 = -349/64.
        self.alhatv878 = 275/48.
        self.alhatv868 = -245/24.
        self.alhatv858 = -349/64.
        self.alhatv848 = 563/96.
        self.alhatv838 = 361/288.
        self.alhatv828 = 347/96.
        self.alhatv818 = 2835/128.
        self.alhatv7287 = delta*nu**2/(96.*delta + 96.)
        self.alhatv7277 = delta*nu**2/(96.*delta + 96.)
        self.alhatv7267 = 5.*delta*nu**2/(96*delta + 96)
        self.alhatv7257 = 5.*delta*nu**2/(96*delta + 96)
        self.alhatv7247 = 31.*nu**2/(96*delta + 96)
        self.alhatv7237 = -29.*nu**2/(96*delta + 96)
        self.alhatv7227 = 155.*nu**2/(96*delta + 96)
        self.alhatv7217 = -145.*nu**2/(96*delta + 96)
        self.alhatv7207 = 25.*delta*nu/(16*delta + 16)
        self.alhatv7197 = 55.*delta*nu/(32*delta + 32)
        self.alhatv7187 = 125.*delta*nu/(16*delta + 16)
        self.alhatv7177 = 275.*delta*nu/(32*delta + 32)
        self.alhatv7167 = -131.*nu/(16*delta + 16)
        self.alhatv7157 = 367.*nu/(32*delta + 32)
        self.alhatv7147 = -655.*nu/(16*delta + 16)
        self.alhatv7137 = 1835.*nu/(32*delta + 32)
        self.alhatv7127 = -27.*delta/(nu*(32*delta + 32))
        self.alhatv7117 = -135.*delta/(nu*(32*delta + 32))
        self.alhatv7107 = -27./(nu*(32*delta + 32))
        self.alhatv797 = -135./(nu*(32*delta + 32))
        self.alhatv787 = 75.*delta/(32*delta + 32)
        self.alhatv777 = -81.*delta/(32*delta + 32)
        self.alhatv767 = 375.*delta/(32*delta + 32)
        self.alhatv757 = -405.*delta/(32*delta + 32)
        self.alhatv747 = 129./(32*delta + 32)
        self.alhatv737 = -135./(32*delta + 32)
        self.alhatv727 = 645./(32*delta + 32)
        self.alhatv717 = -675./(32*delta + 32)
        self.alhatv6366 = 7*nu**3/1296.
        self.alhatv6356 = 31*nu**2/24.
        self.alhatv6346 = 41*pi**2*nu/24.
        self.alhatv6336 = -6889.*nu/144
        self.alhatv6326 = -8./(3*nu)
        self.alhatv6316 = -15./(8*nu)
        self.alhatv6306 = 5./(4*nu)
        self.alhatv6296 = 5./(4*nu)
        self.alhatv6286 = 79./(8*nu)
        self.alhatv6276 = -7./(6*nu)
        self.alhatv6266 = delta/(2.*nu**2)
        self.alhatv6256 = 11.*delta/(16*nu**2)
        self.alhatv6246 = -35.*delta/(16*nu**2)
        self.alhatv6236 = 1./(2*nu**2)
        self.alhatv6226 = 11./(16*nu**2)
        self.alhatv6216 = -35./(16*nu**2)
        self.alhatv6206 = 2./(delta - 2*nu + 1)
        self.alhatv6196 = 11./(4*delta - 8*nu + 4)
        self.alhatv6186 = -35./(4*delta - 8*nu + 4)
        self.alhatv6176 = 5.*delta/(24*delta + 24)
        self.alhatv6166 = -delta/(3.*(delta + 1))
        self.alhatv6156 = 10./(3*delta + 3)
        self.alhatv6146 = 29./(24*delta + 24)
        self.alhatv6136 = -11./(3*delta + 3)
        self.alhatv6126 = -1./(delta + 1)
        self.alhatv6116 = -11./(delta + 1)
        self.alhatv6106 = 11./(delta + 1)
        self.alhatv696 = -1/3.
        self.alhatv686 = -7/24.
        self.alhatv676 = 2/3.
        self.alhatv666 = -7/24.
        self.alhatv656 = 5/24.
        self.alhatv646 = 121/72.
        self.alhatv636 = 13/36.
        self.alhatv626 = 121/72.
        self.alhatv616 = 135/16.
        self.alhatv5205 = delta*nu/(48.*delta + 48.)
        self.alhatv5195 = delta*nu/(48.*delta + 48.)
        self.alhatv5185 = 11.*delta*nu/(144*delta + 144)
        self.alhatv5175 = 11.*delta*nu/(144*delta + 144)
        self.alhatv5165 = -59.*nu/(48*delta + 48)
        self.alhatv5155 = 61.*nu/(48*delta + 48)
        self.alhatv5145 = -649.*nu/(144*delta + 144)
        self.alhatv5135 = 671.*nu/(144*delta + 144)
        self.alhatv5125 = -9.*delta/(nu*(16*delta + 16))
        self.alhatv5115 = -33.*delta/(nu*(16*delta + 16))
        self.alhatv5105 = -9./(nu*(16*delta + 16))
        self.alhatv595 = -33./(nu*(16*delta + 16))
        self.alhatv585 = -5.*delta/(16*delta + 16)
        self.alhatv575 = -15.*delta/(16*delta + 16)
        self.alhatv565 = -55.*delta/(48*delta + 48)
        self.alhatv555 = -55.*delta/(16*delta + 16)
        self.alhatv545 = 13./(16*delta + 16)
        self.alhatv535 = -33./(16*delta + 16)
        self.alhatv525 = 143./(48*delta + 48)
        self.alhatv515 = -121./(16.*delta + 16)
        self.alhatv4194 = nu**2/24.
        self.alhatv4184 = -19.*nu/8
        self.alhatv4174 = 1./(2*nu)
        self.alhatv4164 = -1./(2*nu)
        self.alhatv4154 = 1./(2*nu)
        self.alhatv4144 = -1./nu
        self.alhatv4134 = 1./(2*nu)
        self.alhatv4124 = -1./nu
        self.alhatv4114 = 2./nu
        self.alhatv4104 = -delta/(4.*nu**2)
        self.alhatv494 = delta/(4.*nu**2)
        self.alhatv484 = delta/(2.*nu**2)
        self.alhatv474 = -1./(4*nu**2)
        self.alhatv464 = 1./(4*nu**2)
        self.alhatv454 = 1./(2*nu**2)
        self.alhatv444 = -1./(delta - 2*nu + 1)
        self.alhatv434 = 1./(delta - 2*nu + 1)
        self.alhatv424 = 2./(delta - 2*nu + 1)
        self.alhatv414 = 27/8.
        self.alhatv3113 = -7.*delta/(nu*(4*delta + 4))
        self.alhatv3103 = -7./(nu*(4*delta + 4))
        self.alhatv393 = -3/(4.*nu)
        self.alhatv383 = -delta/(4.*(delta + 1))
        self.alhatv373 = -delta/(4.*(delta + 1))
        self.alhatv363 = -7.*delta/(12*delta + 12)
        self.alhatv353 = -7.*delta/(12*delta + 12)
        self.alhatv343 = 5./(4*delta + 4)
        self.alhatv333 = -7./(4*delta + 4)
        self.alhatv323 = 35./(12*delta + 12)
        self.alhatv313 = -49./(12*delta + 12)
        self.alhatv222 = nu/6.
        self.alhatv212 = 3/2.
        self.alhatv01 = 1.


cdef class PNCoeffs:
    @cython.embedsignature(True)
    def __cinit__(self,nu,delta):
        # S1dot coeffs
        self.s1dot_coeffs = s1dotCoeffs(nu,delta)
        # S2dot coeffs
        self.s2dot_coeffs = s2dotCoeffs(nu,delta)
        # omegadot coeffs
        self.omegadot_coeffs = omegadotCoeffs(nu,delta)
        # lNdot coeffs
        self.lNdot_coeffs = lNdotCoeffs(nu,delta)
        # Lhat coeffs
        self.lhat_coeffs = lhatCoeffs(nu,delta)
