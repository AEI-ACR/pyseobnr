# cython: language_level=3
# distutils: language = c++

from abc import abstractmethod

cimport libc.math as cmath
from libcpp cimport bool
cimport libcpp.complex as ccomplex

from ._implementation cimport edot_zdot_xavg_flags

cdef:
    double M_EULER_GAMA = 0.577215664901532860606512090082


cdef class BaseCoupledExpressionsCalculation:
    """Base class for the coupled expressions calculation"""

    @abstractmethod
    def initialize(self):
        """Updates the internal state given the set of parameters

        .. note::

            What is considered as constant or variable is defined at the time of the
            code generation. The order of elements in the ``variable`` parameter
            is use-case dependent.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self):
        raise NotImplementedError

    cdef void _initialize(self):
        raise NotImplementedError

    cdef void _compute(self):
        raise NotImplementedError

    cpdef double get(self, str radial_or_azimuthal):
        raise NotImplementedError


cdef class edot_zdot_xavg_flags(BaseCoupledExpressionsCalculation):
    """
    Evolution equations for the parameters (e, z) and the PN equation for
    x. These equations correspond to Eq. (61a), (63) and (67) of
    [Gamboa2024]_ (see also appendix D of [Gamboa2024]_ and
    "docs/mathematica/evolution_equations/").
    """

    def __cinit__(self):
        self._initialized = False

    cdef void _initialize(self
            , double chiA=-1
            , double chiS=-1
            , double delta=-1
            , int flagPN1=-1
            , int flagPN2=-1
            , int flagPN3=-1
            , int flagPN32=-1
            , int flagPN52=-1
            , double nu=-1
        ):

        # internal computations intermediate variables declaration/initialisation
        cdef:
            double tmp_init_0 = 11.0*nu
            double tmp_init_1 = chiA*delta
            double tmp_init_2 = cmath.pow(nu, 2)
            double tmp_init_3 = 10.0*tmp_init_2
            double tmp_init_4 = 12.0*chiS
            double tmp_init_5 = tmp_init_1*(537.0*nu - 2848.0)
            double tmp_init_6 = -2771.0*nu + 390.0*tmp_init_2 + 2848.0
            double tmp_init_7 = 3.0*tmp_init_1*(81.0*nu - 32.0)
            double tmp_init_8 = -561.0*nu + 162.0*tmp_init_2 + 96.0
            double tmp_init_9 = 4.0*tmp_init_1
            double tmp_init_10 = 4.0*chiS
            double tmp_init_11 = 8.0*tmp_init_1
            double tmp_init_12 = 8.0*chiS
            double tmp_init_13 = tmp_init_1*(nu - 3.0)
            double tmp_init_14 = 8.0*nu
            double tmp_init_15 = -tmp_init_14
            double tmp_init_16 = tmp_init_15 + tmp_init_2 + 6.0
            double tmp_init_17 = -384.0*chiS*tmp_init_16 + 768.0*tmp_init_13
            double tmp_init_18 = -tmp_init_17
            double tmp_init_19 = 0.0034722222222222222*flagPN52
            double tmp_init_20 = cmath.pow(chiS, 2)
            double tmp_init_21 = chiS*tmp_init_1
            double tmp_init_22 = cmath.pow(chiA, 2)
            double tmp_init_23 = 4.0*nu
            double tmp_init_24 = tmp_init_23 - 1.0
            double tmp_init_25 = 32.0*tmp_init_2
            double tmp_init_26 = 576.0*tmp_init_22
            double tmp_init_27 = 4.0*tmp_init_2
            double tmp_init_28 = 576.0*tmp_init_20
            double tmp_init_29 = 13.0*nu
            double tmp_init_30 = tmp_init_29 - 72.0
            double tmp_init_31 = 1152.0*tmp_init_21
            double tmp_init_32 = 48.0*nu
            double tmp_init_33 = 288.0*tmp_init_20
            double tmp_init_34 = 1152.0*tmp_init_22
            double tmp_init_35 = -18816.0*nu + tmp_init_33*(11.0*tmp_init_2 - tmp_init_32 + 144.0) + tmp_init_34*(-145.0*nu + tmp_init_27 + 36.0) + 50688.0
            double tmp_init_36 = 9.8696044010893586
            double tmp_init_37 = 107168.0 - 1845.0*tmp_init_36
            double tmp_init_38 = 3.0*nu
            double tmp_init_39 = 29952.0*tmp_init_2 - 3456.0*tmp_init_21*(31.0*nu - 23.0) + tmp_init_26*(-301.0*nu + 70.0*tmp_init_2 + 69.0) + tmp_init_33*(-322.0*nu + 83.0*tmp_init_2 + 138.0) + 199296.0
            double tmp_init_40 = 3.0*tmp_init_2
            double tmp_init_41 = tmp_init_14 - tmp_init_40 + 6.0
            double tmp_init_42 = 1152.0*tmp_init_20
            double tmp_init_43 = 8.0*tmp_init_2
            double tmp_init_44 = -62784.0*nu + 4608.0*tmp_init_21*(nu + 3.0) + 2304.0*tmp_init_22*(-14.0*nu + tmp_init_43 + 3.0) + 131328.0
            double tmp_init_45 = -55296.0*nu + 8640.0*tmp_init_2 + 129600.0
            double tmp_init_46 = 41472.0*nu - 103680.0
            double tmp_init_47 = -123.0*tmp_init_36
            double tmp_init_48 = tmp_init_47 + 10880.0
            double tmp_init_49 = 18.0*nu
            double tmp_init_50 = 6.0*tmp_init_2
            double tmp_init_51 = tmp_init_22*(-46.0*nu + tmp_init_50 + 11.0)
            double tmp_init_52 = 16.0*nu
            double tmp_init_53 = -tmp_init_52
            double tmp_init_54 = tmp_init_20*(tmp_init_27 + tmp_init_53 + 11.0)
            double tmp_init_55 = 9.0*nu
            double tmp_init_56 = tmp_init_55 - 11.0
            double tmp_init_57 = 17280.0*tmp_init_2 - 6912.0*tmp_init_21*tmp_init_56 + 3456.0*tmp_init_51 + 3456.0*tmp_init_54 + 51840.0
            double tmp_init_58 = -tmp_init_48*tmp_init_49 + tmp_init_57
            double tmp_init_59 = nu + 6.0
            double tmp_init_60 = 2304.0*tmp_init_21
            double tmp_init_61 = 8448.0*nu - 1152.0*tmp_init_22*(23.0*nu + tmp_init_27 - 6.0) + tmp_init_42*tmp_init_59 + tmp_init_59*tmp_init_60 + 50688.0
            double tmp_init_62 = tmp_init_14 - tmp_init_2 + 26.0
            double tmp_init_63 = 53184.0*nu + 1872.0*tmp_init_2 + tmp_init_34*(50.0*nu + tmp_init_43 - 13.0) - tmp_init_60*(tmp_init_38 + 13.0) + 44352.0
            double tmp_init_64 = 0.00019290123456790123*flagPN3
            double tmp_init_65 = 12.0*nu
            double tmp_init_66 = 96.0*nu - 24.0
            double tmp_init_67 = 48.0*chiS
            double tmp_init_68 = tmp_init_1*tmp_init_67 + tmp_init_15 + 24.0*tmp_init_20 + 48.0
            double tmp_init_69 = 24.0*nu - 6.0
            double tmp_init_70 = tmp_init_1*tmp_init_4 + 6.0*tmp_init_20
            double tmp_init_71 = tmp_init_70 + 36.0
            double tmp_init_72 = 42.0*nu
            double tmp_init_73 = 12.0*tmp_init_20
            double tmp_init_74 = 24.0*tmp_init_21
            double tmp_init_75 = tmp_init_32 - 12.0
            double tmp_init_76 = 36.0*nu
            double tmp_init_77 = tmp_init_76 - 90.0
            double tmp_init_78 = -tmp_init_77
            double tmp_init_79 = 0.055555555555555556*flagPN2
            double tmp_init_80 = 2.0*tmp_init_1
            double tmp_init_81 = nu - 2.0
            double tmp_init_82 = -tmp_init_48*tmp_init_49 + tmp_init_57
            double tmp_init_83 = -tmp_init_22*tmp_init_69
            double tmp_init_84 = flagPN32*(-chiS*tmp_init_81 + tmp_init_80)
            double tmp_init_85 = 120.0*tmp_init_1
            double tmp_init_86 = 120.0*chiS*(55.0*nu + tmp_init_3 - 192.0) - tmp_init_85*(tmp_init_0 + 192.0)
            double tmp_init_87 = 27.0*nu
            double tmp_init_88 = 15.0*tmp_init_1
            double tmp_init_89 = 18.0*tmp_init_2
            double tmp_init_90 = 377.0*nu
            double tmp_init_91 = 480.0*tmp_init_1
            double tmp_init_92 = 60.0*tmp_init_1
            double tmp_init_93 = chiS*tmp_init_16
            double tmp_init_94 = 7680.0*tmp_init_13 - 3840.0*tmp_init_93
            double tmp_init_95 = 1920.0*tmp_init_1
            double tmp_init_96 = 128.0*chiS
            double tmp_init_97 = 384.0*tmp_init_2
            double tmp_init_98 = 144.0*nu
            double tmp_init_99 = 85.0*nu
            double tmp_init_100 = 192.0*tmp_init_22
            double tmp_init_101 = 192.0*tmp_init_20
            double tmp_init_102 = 64.0*tmp_init_22
            double tmp_init_103 = 64.0*tmp_init_20
            double tmp_init_104 = 3.1415926535897932*delta
            double tmp_init_105 = -513965390400.0*tmp_init_104
            double tmp_init_106 = chiS*delta
            double tmp_init_107 = 1455300.0*tmp_init_36
            double tmp_init_108 = 0.69314718055994531
            double tmp_init_109 = cmath.pow(nu, 3)
            double tmp_init_110 = 1.6094379124341004
            double tmp_init_111 = 1.0986122886681097
            double tmp_init_112 = 3.1415926535897932*chiS
            double tmp_init_113 = -2337616325721600.0*tmp_init_108 + 1570602880000.0*tmp_init_109 + 579304687500000.0*tmp_init_110 + 638683384600800.0*tmp_init_111 + 3880800.0*tmp_init_112*(42287.0*nu - 132438.0) + 282528472943.56691
            double tmp_init_114 = -8736588907200.0*tmp_init_104
            double tmp_init_115 = 3326400.0*tmp_init_106
            double tmp_init_116 = 1663200.0*tmp_init_22
            double tmp_init_117 = 1663200.0*tmp_init_20
            double tmp_init_118 = -2697716740561920.0*tmp_init_108 + 7164771768000.0*tmp_init_109 + 868957031250000.0*tmp_init_110 + 510129379766160.0*tmp_init_111 - 7761600.0*tmp_init_112*(95561.0*nu + 1125617.0) + 7048255400646.843
            double tmp_init_119 = tmp_init_1*tmp_init_12 + 4.0*tmp_init_20 + tmp_init_22*(4.0 - tmp_init_52)
            double tmp_init_120 = 7306397683200.0*tmp_init_104
            double tmp_init_121 = 4435200.0*tmp_init_106
            double tmp_init_122 = 2217600.0*tmp_init_22
            double tmp_init_123 = 2217600.0*tmp_init_20
            double tmp_init_124 = -254015272304640.0*tmp_init_108 - 5965749020000.0*tmp_init_109 + 91540726421760.0*tmp_init_111 + 186278400.0*tmp_init_112*(74954.0*nu - 39223.0) - 14631654901296.161
            double tmp_init_125 = 31322368000.0*tmp_init_109
            double tmp_init_126 = chiA*(9161916825600.0*tmp_init_104 + 8870400.0*tmp_init_106*(3985373.0*nu - 2069461.0)) + 24375719121600.0*nu + 6569517588480.0*tmp_init_108 + 798806624000.0*tmp_init_109 + 7472712360960.0*tmp_init_111 - 1490227200.0*tmp_init_112*(4937.0*nu - 6148.0) + 1983096561600.0*tmp_init_2 - 4435200.0*tmp_init_20*(-6612781.0*nu + 2014516.0*tmp_init_2 + 2069461.0) - 4435200.0*tmp_init_22*(-9635809.0*nu + 4395216.0*tmp_init_2 + 2069461.0) - 46569600.0*tmp_init_36*(17917.0*nu + 49216.0) + 4044479733679.6361

        # computations
        self._gr_k_0 = 12.0*tmp_init_1*(tmp_init_0 - 128.0) - tmp_init_4*(-105.0*nu + tmp_init_3 + 128.0)
        self._gr_k_1 = -chiS*tmp_init_6 + tmp_init_5
        self._gr_k_2 = -chiS*tmp_init_8 + tmp_init_7
        self._gr_k_3 = -tmp_init_10*(-985.0*nu + 210.0*tmp_init_2 + 464.0) + tmp_init_9*(339.0*nu - 464.0)
        self._gr_k_4 = tmp_init_11*(183.0*nu - 184.0) - tmp_init_12*(-545.0*nu + 78.0*tmp_init_2 + 184.0)
        self._gr_k_5 = -tmp_init_10*(-1729.0*nu + 378.0*tmp_init_2 + 416.0) + tmp_init_9*(675.0*nu - 416.0)
        self._gr_k_6 = tmp_init_18
        self._gr_k_7 = tmp_init_17
        self._gr_k_8 = tmp_init_19
        self._gr_k_9 = -1728.0*nu + 6336.0*tmp_init_20 + 12672.0*tmp_init_21 - 6336.0*tmp_init_22*tmp_init_24 + 19712.0
        self._gr_k_10 = -54720.0*nu + 116352.0*tmp_init_21 - tmp_init_26*(396.0*nu + tmp_init_25 - 101.0) + tmp_init_28*(tmp_init_15 + tmp_init_27 + 101.0) + 46464.0
        self._gr_k_11 = -tmp_init_30*tmp_init_31 + tmp_init_35
        self._gr_k_12 = -tmp_init_37*tmp_init_38 + tmp_init_39
        self._gr_k_13 = tmp_init_41*tmp_init_42 + tmp_init_44
        self._gr_k_14 = 19872.0*nu + 69120.0
        self._gr_k_15 = 24768.0*nu + 103680.0
        self._gr_k_16 = tmp_init_45
        self._gr_k_17 = -tmp_init_46
        self._gr_k_18 = tmp_init_58
        self._gr_k_19 = -tmp_init_61
        self._gr_k_20 = -tmp_init_28*tmp_init_62 + tmp_init_63
        self._gr_k_21 = tmp_init_58
        self._gr_k_22 = tmp_init_64
        self._gr_k_23 = tmp_init_65 + 90.0
        self._gr_k_24 = -tmp_init_22*tmp_init_66 + tmp_init_68
        self._gr_k_25 = tmp_init_22*tmp_init_69 - tmp_init_71
        self._gr_k_26 = tmp_init_22*tmp_init_75 + tmp_init_72 - tmp_init_73 - tmp_init_74 + 15.0
        self._gr_k_27 = tmp_init_14 + 96.0
        self._gr_k_28 = tmp_init_77
        self._gr_k_29 = tmp_init_78
        self._gr_k_30 = tmp_init_79
        self._gr_k_31 = 0.66666666666666667*flagPN1
        self._gr_k_32 = -0.66666666666666667*flagPN32*(-chiS*tmp_init_81 + tmp_init_80)
        self._gr_k_33 = -chiS*tmp_init_6 + tmp_init_5
        self._gr_k_34 = -chiS*tmp_init_8 + tmp_init_7
        self._gr_k_35 = tmp_init_18
        self._gr_k_36 = tmp_init_17
        self._gr_k_37 = tmp_init_19
        self._gr_k_38 = tmp_init_61
        self._gr_k_39 = tmp_init_28*tmp_init_62 - tmp_init_63
        self._gr_k_40 = -tmp_init_30*tmp_init_31 + tmp_init_35
        self._gr_k_41 = -tmp_init_37*tmp_init_38 + tmp_init_39
        self._gr_k_42 = tmp_init_41*tmp_init_42 + tmp_init_44
        self._gr_k_43 = tmp_init_46
        self._gr_k_44 = -tmp_init_45
        self._gr_k_45 = -tmp_init_82
        self._gr_k_46 = tmp_init_82
        self._gr_k_47 = tmp_init_64
        self._gr_k_48 = tmp_init_71 + tmp_init_83
        self._gr_k_49 = -tmp_init_22*tmp_init_75 - tmp_init_72 + tmp_init_73 + tmp_init_74 - 15.0
        self._gr_k_50 = -tmp_init_22*tmp_init_66 + tmp_init_68
        self._gr_k_51 = tmp_init_78
        self._gr_k_52 = tmp_init_79
        self._gr_k_53 = -0.66666666666666667*tmp_init_84
        self._gr_k_54 = tmp_init_86
        self._gr_k_55 = tmp_init_86
        self._gr_k_56 = 15.0*chiS*(-73.0*nu + tmp_init_89 + 32.0) + tmp_init_88*(32.0 - tmp_init_87)
        self._gr_k_57 = 30.0*chiS*(1421.0*nu + 310.0*tmp_init_2 - 5216.0) - 30.0*tmp_init_1*(tmp_init_90 + 5216.0)
        self._gr_k_58 = 15.0*chiS*(239.0*nu + 130.0*tmp_init_2 - 1376.0) - tmp_init_88*(179.0*nu + 1376.0)
        self._gr_k_59 = 120.0*chiS*(-113.0*nu + 26.0*tmp_init_2 + 32.0) + tmp_init_85*(32.0 - 43.0*nu)
        self._gr_k_60 = 120.0*chiS*(-139.0*nu + 158.0*tmp_init_2 - 1248.0) - tmp_init_85*(289.0*nu + 1248.0)
        self._gr_k_61 = 480.0*chiS*(21.0*nu + 38.0*tmp_init_2 - 416.0) - tmp_init_91*(67.0*nu + 416.0)
        self._gr_k_62 = 60.0*chiS*(-1275.0*nu + 254.0*tmp_init_2 + 256.0) + tmp_init_92*(256.0 - 481.0*nu)
        self._gr_k_63 = 60.0*chiS*(-3133.0*nu + 1074.0*tmp_init_2 - 5632.0) - tmp_init_92*(2247.0*nu + 5632.0)
        self._gr_k_64 = tmp_init_94
        self._gr_k_65 = tmp_init_94
        self._gr_k_66 = 240.0*chiS*(-1099.0*nu + 166.0*tmp_init_2 + 304.0) + 240.0*tmp_init_1*(304.0 - tmp_init_90)
        self._gr_k_67 = 1920.0*chiS*(-136.0*nu + 25.0*tmp_init_2 - 38.0) - tmp_init_95*(59.0*nu + 38.0)
        self._gr_k_68 = 30720.0*tmp_init_13 - 15360.0*tmp_init_93
        self._gr_k_69 = 480.0*chiS*(-1007.0*nu + 114.0*tmp_init_2 + 400.0) - tmp_init_91*(309.0*nu - 400.0)
        self._gr_k_70 = 15360.0*tmp_init_13 - 7680.0*tmp_init_93
        self._gr_k_71 = 128.0*nu
        self._gr_k_72 = 1920.0*chiS*(-89.0*nu + tmp_init_43 + 44.0) + tmp_init_95*(44.0 - 25.0*nu)
        self._gr_k_73 = 0.00026041666666666667*flagPN52
        self._gr_k_74 = 6.0*chiS*(-1067672.0*nu + 272552.0*tmp_init_2 - 3441339.0) - 5813548.6213944483*nu - 6.0*tmp_init_1*(769244.0*nu + 3441339.0) - 27100776.238596404
        self._gr_k_75 = -353402155.83203087*nu - tmp_init_11*(4790765.0*nu + 7388082.0) - tmp_init_12*(16825619.0*nu + 2435258.0*tmp_init_2 + 7388082.0) - 778173050.07290616
        self._gr_k_76 = 3252480.0*chiS*tmp_init_16 - 6504960.0*tmp_init_13
        self._gr_k_77 = 48.0*chiA*delta*(1745023.0*nu + 4476734.0) - 1003363520.4644293*nu - tmp_init_67*(3800571.0*nu + 3015082.0*tmp_init_2 - 4476734.0) - 949295718.15287038
        self._gr_k_78 = 4919040.0*chiS*tmp_init_16 - 9838080.0*tmp_init_13
        self._gr_k_79 = 16343040.0*tmp_init_13 - 8171520.0*tmp_init_93
        self._gr_k_80 = 38792.386086526767 - 4310.2651207251963*nu
        self._gr_k_81 = 128.0*chiA*delta*(952868.0*nu + 57681.0) - 348573829.5163581*nu - tmp_init_96*(-985610.0*nu + 670516.0*tmp_init_2 - 57681.0) - 159145141.88463065
        self._gr_k_82 = 4.0792084377610693e-7*flagPN52
        self._gr_k_83 = tmp_init_84
        self._gr_k_84 = 29904.0*tmp_init_1 - tmp_init_67*(92.0*nu - 623.0) + 76079.949291984023
        self._gr_k_85 = 64.0*chiS*(1993.0*nu - 62.0) - 3968.0*tmp_init_1 + 900103.99436531884
        self._gr_k_86 = -259712.0*tmp_init_1 + tmp_init_96*(1664.0*nu - 2029.0) + 594138.0026469017
        self._gr_k_87 = 3.4265350877192982e-5*flagPN32
        self._gr_k_88 = -3840.0*nu + tmp_init_97 - 2016.0
        self._gr_k_89 = tmp_init_98*(tmp_init_38 - 4.0)
        self._gr_k_90 = -3584.0*chiS*tmp_init_13 - 6880.0*nu + 160.0*tmp_init_2 + 256.0*tmp_init_20*(-tmp_init_29 + tmp_init_40 + 21.0) + 256.0*tmp_init_22*(tmp_init_27 - tmp_init_99 + 21.0) - 26688.0
        self._gr_k_91 = 864.0*nu*tmp_init_21 - 384.0*nu + 1440.0*tmp_init_2 - tmp_init_20*tmp_init_98*(nu - 5.0) - tmp_init_22*tmp_init_24*tmp_init_98 + 48.0
        self._gr_k_92 = 32.0*chiA*chiS*delta*(576.0 - 79.0*nu) - 20224.0*nu - 800.0*tmp_init_2 + 16.0*tmp_init_20*(-169.0*nu + 45.0*tmp_init_2 + 576.0) - 16.0*tmp_init_22*(2293.0*nu + 44.0*tmp_init_2 - 576.0) - 3792.0
        self._gr_k_93 = 384.0*chiA*chiS*delta*(tmp_init_52 - 1.0) + nu*(tmp_init_47 + 12224.0) - tmp_init_100*(-tmp_init_55 + tmp_init_89 + 1.0) - tmp_init_101*(tmp_init_50 - tmp_init_87 + 1.0) - tmp_init_97 + 1056.0
        self._gr_k_94 = 128.0*chiA*chiS*delta*(61.0*nu + 237.0) + 8.0*nu*(tmp_init_47 + 2260.0) - tmp_init_102*(911.0*nu + 100.0*tmp_init_2 - 237.0) - tmp_init_103*(24.0*tmp_init_2 - tmp_init_99 - 237.0) - 5632.0*tmp_init_2 + 21696.0
        self._gr_k_95 = 640.0*chiA*chiS*delta*(29.0*nu - 9.0) + 8.0*nu*(tmp_init_47 + 5212.0) - tmp_init_102*(-223.0*nu + 124.0*tmp_init_2 + 45.0) - tmp_init_103*(-247.0*nu + 63.0*tmp_init_2 + 45.0) - 5248.0*tmp_init_2 + 14208.0
        self._gr_k_96 = 4320.0*nu - 17280.0
        self._gr_k_97 = 5760.0*nu - 14400.0
        self._gr_k_98 = 6528.0*nu + 960.0*tmp_init_2 - 17280.0
        self._gr_k_99 = 2.0*nu*(-tmp_init_47 - 8000.0) + 1920.0*tmp_init_2 - 768.0*tmp_init_21*tmp_init_56 + 384.0*tmp_init_51 + 384.0*tmp_init_54 - 8640.0
        self._gr_k_100 = 2016.0*nu - 3456.0
        self._gr_k_101 = 1152.0*chiA*chiS*delta*(19.0*nu - 13.0) + nu*(62176.0 - 1722.0*tmp_init_36) - tmp_init_100*(-171.0*nu + tmp_init_25 + 39.0) - tmp_init_101*(-99.0*nu + 28.0*tmp_init_2 + 39.0) - 4608.0*tmp_init_2 + 3456.0
        self._gr_k_102 = 0.0026041666666666667*flagPN3
        self._gr_k_103 = 27599862.0*nu + 5073936.0*tmp_init_2 - 2268.0*tmp_init_20*(20.0*nu + 107.0) - 485352.0*tmp_init_21 + 2268.0*tmp_init_22*(448.0*nu - 107.0) + 22189718.0
        self._gr_k_104 = 49648356.0*nu + 11850804.0*tmp_init_2 - 1008.0*tmp_init_20*(540.0*nu + 881.0) - 1776096.0*tmp_init_21 + 1008.0*tmp_init_22*(4064.0*nu - 881.0) - 11389620.0
        self._gr_k_105 = 1018824.0*nu + 180768.0*tmp_init_2 + 3169323.0
        self._gr_k_106 = 3049200.0 - 1219680.0*nu
        self._gr_k_107 = 7660800.0 - 3064320.0*nu
        self._gr_k_108 = 5423040.0*chiA*chiS*delta + 12363120.0*nu + 3612672.0*tmp_init_2 - 10080.0*tmp_init_20*(tmp_init_76 - 269.0) - 10080.0*tmp_init_22*(1040.0*nu - 269.0) - 12243152.0
        self._gr_k_109 = 1.6316833751044277e-6*flagPN2
        self._gr_k_110 = chiA*(tmp_init_105 + 831600.0*tmp_init_106*(4003818.0*nu - 6397597.0)) + 47873641384800.0*nu + tmp_init_107*(9061.0*nu - 110016.0) + tmp_init_113 + 5953045190400.0*tmp_init_2 - 415800.0*tmp_init_20*(-6830264.0*nu + 1593536.0*tmp_init_2 + 6397597.0) - 415800.0*tmp_init_22*(-26767760.0*nu + 3295488.0*tmp_init_2 + 6397597.0) + 18009590842332.0
        self._gr_k_111 = chiA*(tmp_init_105 + 2494800.0*tmp_init_106*(521486.0*nu - 1138719.0)) + 36614796372000.0*nu + tmp_init_113 + 12469436179200.0*tmp_init_2 - 1247400.0*tmp_init_20*(-831208.0*nu + 169792.0*tmp_init_2 + 1138719.0) - 3742200.0*tmp_init_22*(-1588880.0*nu + 185472.0*tmp_init_2 + 379573.0) + 13097700.0*tmp_init_36*(6519.0*nu - 12224.0) + 2264761392444.0
        self._gr_k_112 = 19768.0*nu + 94887.0
        self._gr_k_113 = 257124.0*nu + 464376.0
        self._gr_k_114 = 196448.0*nu + 164376.0
        self._gr_k_115 = chiA*(tmp_init_114 + tmp_init_115*(2182831.0*nu - 3131923.0)) + 78553258707000.0*nu + tmp_init_107*(790193.0*nu - 2744576.0) - tmp_init_116*(-13837483.0*nu + 1693552.0*tmp_init_2 + 3131923.0) - tmp_init_117*(-3055871.0*nu + 1466108.0*tmp_init_2 + 3131923.0) + tmp_init_118 + 62213110759800.0*tmp_init_2 - 206275000274856.0
        self._gr_k_116 = chiA*(tmp_init_114 + tmp_init_115*(1260511.0*nu - 2004643.0)) + 64875068643000.0*nu + tmp_init_107*(865223.0*nu - 2744576.0) - tmp_init_116*(-9123403.0*nu + 1078672.0*tmp_init_2 + 2004643.0) - tmp_init_117*(-1416191.0*nu + 1056188.0*tmp_init_2 + 2004643.0) + tmp_init_118 + 60944647995000.0*tmp_init_2 - 158355480094632.0
        self._gr_k_117 = -6.0*nu - 3.0
        self._gr_k_118 = tmp_init_119 + tmp_init_53 - 80.0
        self._gr_k_119 = chiA*(tmp_init_120 + tmp_init_121*(4433569.0*nu + 7701538.0)) - 38365525968000.0*nu - tmp_init_122*(28354633.0*nu + 6528312.0*tmp_init_2 - 7701538.0) - tmp_init_123*(-6415619.0*nu + 3646412.0*tmp_init_2 - 7701538.0) - tmp_init_124 + 44062562728800.0*tmp_init_2 + 46569600.0*tmp_init_36*(28413.0*nu - 178048.0) - 175188432361920.0
        self._gr_k_120 = chiA*(tmp_init_120 + tmp_init_121*(3284449.0*nu + 9106018.0)) - 62018474179200.0*nu - tmp_init_122*(34227913.0*nu + 5762232.0*tmp_init_2 - 9106018.0) - tmp_init_123*(-4372739.0*nu + 3135692.0*tmp_init_2 - 9106018.0) - tmp_init_124 + 50052530959200.0*tmp_init_2 + 186278400.0*tmp_init_36*(8077.0*nu - 44512.0) - 194179275208896.0
        self._gr_k_121 = -32.0*nu + tmp_init_119 + 4.0
        self._gr_k_122 = tmp_init_65 - 30.0
        self._gr_k_123 = tmp_init_23 - 24.0
        self._gr_k_124 = tmp_init_23 - 60.0
        self._gr_k_125 = -40.0*nu + tmp_init_70 + tmp_init_83 + 48.0
        self._gr_k_126 = -709055424000.0*nu + tmp_init_125 - 514084032000.0*tmp_init_2 + 8213198775075.0
        self._gr_k_127 = 575031441600.0*nu + tmp_init_125 + 227955974400.0*tmp_init_2 + 1844398491075.0
        self._gr_k_128 = tmp_init_126 - 40817973192384.0
        self._gr_k_129 = tmp_init_126 - 41077048481472.0
        self._gr_k_130 = -5.886303661992885e-12*flagPN3
        self._gr_k_131 = -1.9580200501253133e-5*flagPN1
        self._gr_k_132 = -0.25*flagPN2
        self._gr_k_133 = -20.266666666666667*nu
        self._gr_k_134 = -3.0*flagPN1


    cdef void _compute(self
            , double e=-1
            , double omega=-1
            , double z=-1
        ):
        """
        values being computed:
        - edotExpResumParser
        - xAvgOmegaInstParser
        - zdotParser
        """

        # internal computations intermediate variables declaration/initialisation
        cdef:
            double tmp_0 = cmath.pow(e, 2)
            double tmp_1 = cmath.fabs(tmp_0 - 1)
            double tmp_2 = cmath.pow(tmp_1, 1.5)
            double tmp_3 = cmath.cos(z)
            double tmp_4 = e*tmp_3
            double tmp_5 = tmp_4 + 1.0
            double tmp_6 = cmath.fabs(tmp_5)
            double tmp_7 = cmath.pow(e, 3)
            double tmp_8 = cmath.cos(3*z)
            double tmp_9 = cmath.pow(e, 4)
            double tmp_10 = cmath.cos(2*z)
            double tmp_11 = self._gr_k_0*tmp_9 + self._gr_k_5*tmp_4 + tmp_0*(self._gr_k_3*tmp_10 + self._gr_k_4)
            double tmp_12 = cmath.pow(tmp_6, -3.3333333333333333)
            double tmp_13 = cmath.pow(omega, 1.6666666666666667)*tmp_12
            double tmp_14 = cmath.pow(e, 6)
            double tmp_15 = cmath.pow(e, 5)
            double tmp_16 = tmp_15*tmp_3
            double tmp_17 = self._gr_k_14*tmp_14 + self._gr_k_15*tmp_16 + tmp_7*(self._gr_k_10*tmp_3 + self._gr_k_9*tmp_8)
            double tmp_18 = cmath.pow(omega, 2)/cmath.pow(tmp_6, 4)
            double tmp_19 = tmp_3*tmp_7
            double tmp_20 = self._gr_k_23*tmp_9 + self._gr_k_27*tmp_19 + self._gr_k_28*tmp_2
            double tmp_21 = cmath.pow(omega, 1.3333333333333333)*cmath.pow(tmp_6, -2.6666666666666667)
            double tmp_22 = cmath.pow(tmp_6, -2)
            double tmp_23 = e*omega*tmp_22*(e + tmp_3)
            double tmp_24 = cmath.pow(omega, 0.66666666666666667)
            double tmp_25 = cmath.pow(tmp_6, -1.3333333333333333)
            double tmp_26 = self._gr_k_31*e*tmp_24*tmp_25*(3.0*e + 2.0*tmp_3) + 1.0
            double tmp_27 = self._gr_k_37*tmp_13*(self._gr_k_35*tmp_2 + self._gr_k_36 + tmp_11 + tmp_7*(self._gr_k_33*tmp_3 + self._gr_k_34*tmp_8)) + self._gr_k_47*tmp_18*(self._gr_k_42*tmp_4 + self._gr_k_46 + tmp_0*(self._gr_k_40*tmp_10 + self._gr_k_41) + tmp_17 + tmp_2*(self._gr_k_43*tmp_4 + self._gr_k_44*tmp_0 + self._gr_k_45) + tmp_9*(self._gr_k_38*tmp_10 + self._gr_k_39)) + self._gr_k_52*tmp_21*(self._gr_k_50*tmp_4 + self._gr_k_51 + tmp_0*(self._gr_k_48*tmp_10 + self._gr_k_49) + tmp_20) + self._gr_k_53*tmp_23 + tmp_26
            double tmp_28 = tmp_25*tmp_27
            double tmp_29 = tmp_24*tmp_28
            double tmp_30 = tmp_21*cmath.pow(tmp_27, 2)
            double tmp_31 = tmp_1*tmp_24
            double tmp_32 = cmath.fabs(tmp_31*(self._gr_k_22*tmp_18*(self._gr_k_13*tmp_4 + self._gr_k_21 + tmp_0*(self._gr_k_11*tmp_10 + self._gr_k_12) + tmp_17 - tmp_2*(self._gr_k_16*tmp_0 + self._gr_k_17*tmp_4 + self._gr_k_18) - tmp_9*(self._gr_k_19*tmp_10 + self._gr_k_20)) + self._gr_k_30*tmp_21*(self._gr_k_24*tmp_4 + self._gr_k_29 - tmp_0*(self._gr_k_25*tmp_10 + self._gr_k_26) + tmp_20) + self._gr_k_32*tmp_23 + self._gr_k_8*tmp_13*(self._gr_k_6*tmp_2 + self._gr_k_7 + tmp_11 + tmp_7*(self._gr_k_1*tmp_3 + self._gr_k_2*tmp_8)) + tmp_26))
            double tmp_33 = tmp_22*cmath.pow(tmp_32, 1.5)/tmp_2
            double tmp_34 = cmath.pow(e, 8)
            double tmp_35 = cmath.sqrt(tmp_1)
            double tmp_36 = cmath.pow(tmp_1, -2.5)*tmp_12*cmath.pow(tmp_32, 2.5)
            double tmp_37 = tmp_35 + 1.0
            double tmp_38 = tmp_18*cmath.pow(tmp_27, 3)
            double tmp_39 = cmath.pow(tmp_5, 2)
            double tmp_40 = tmp_4 + 2.0
            double tmp_41 = cmath.cos(4*z)

        # edot/math.m: --- edotFullExpResumSimplified ---#
        self.edotExpResumParser = self._gr_k_133*e*cmath.pow(omega, 2.6666666666666667)*tmp_2*cmath.pow(tmp_27, 4)*cmath.pow(tmp_6, -5.3333333333333333)*(self._gr_k_109*tmp_30*(self._gr_k_103*tmp_9 + self._gr_k_104*tmp_0 + self._gr_k_105*tmp_14 + self._gr_k_108 + tmp_2*(self._gr_k_106*tmp_0 + self._gr_k_107)) + self._gr_k_130*tmp_38*(self._gr_k_110*tmp_14 + self._gr_k_115*tmp_9 + self._gr_k_120*tmp_0 + self._gr_k_126*tmp_34 + self._gr_k_129 + tmp_35*(self._gr_k_111*tmp_14 + self._gr_k_116*tmp_9 + self._gr_k_119*tmp_0 + self._gr_k_127*tmp_34 + self._gr_k_128) + 284739840.0*tmp_37*(-0.5*cmath.log(cmath.pow(tmp_37, 2)) + cmath.log(2*tmp_1*cmath.sqrt(tmp_32)/cmath.pow(tmp_6, 2.0/3.0)))*(89024.0*tmp_0 + 1719.0*tmp_14 + 42884.0*tmp_9 + 24608.0))/tmp_37 + self._gr_k_131*tmp_29*(self._gr_k_112*tmp_9 + self._gr_k_113*tmp_0 + self._gr_k_114) + self._gr_k_82*tmp_36*(self._gr_k_74*tmp_14 + self._gr_k_75*tmp_9 + self._gr_k_77*tmp_0 + self._gr_k_80*tmp_34 + self._gr_k_81 + tmp_35*(self._gr_k_76*tmp_9 + self._gr_k_78*tmp_0 + self._gr_k_79)) + self._gr_k_87*tmp_33*(self._gr_k_84*tmp_9 + self._gr_k_85*tmp_0 + self._gr_k_86 - 307.87608005179974*tmp_14) + 0.39802631578947368*tmp_0 + 1.0)
        # xavg_omegainst/math.m: xAvg\[CapitalOmega]Simplified
        self.xAvgOmegaInstParser = tmp_28*tmp_31
        # zdot/math.m: --- \[Chi]dotFullSimplified ---#
        self.zdotParser = tmp_33*tmp_39*(self._gr_k_102*tmp_38*(self._gr_k_100*tmp_14 + self._gr_k_101 + self._gr_k_95*tmp_4 + self._gr_k_96*tmp_16 + tmp_0*(self._gr_k_93*tmp_10 + self._gr_k_94) + tmp_2*(self._gr_k_97*tmp_4 + self._gr_k_98*tmp_0 + self._gr_k_99) + tmp_7*(self._gr_k_91*tmp_8 + self._gr_k_92*tmp_3) + tmp_9*(self._gr_k_88*tmp_10 + self._gr_k_89*tmp_41 + self._gr_k_90)) + self._gr_k_132*tmp_30*(self._gr_k_121*tmp_4 + self._gr_k_122*tmp_2 + self._gr_k_123*tmp_9 + self._gr_k_124*tmp_19 + self._gr_k_125 + tmp_0*(self._gr_k_117*tmp_10 + self._gr_k_118)) + self._gr_k_134*tmp_29*(tmp_0 + tmp_5) + self._gr_k_73*tmp_36*(self._gr_k_69*tmp_4 + self._gr_k_71*e*cmath.pow(tmp_1, 3)*tmp_40*(-1762.0*tmp_0 + 5635.0*tmp_14 - 3265.0*tmp_9 - 608.0)*cmath.sin(z) + self._gr_k_72 + tmp_0*(self._gr_k_66*tmp_10 + self._gr_k_67) + tmp_14*(self._gr_k_54*tmp_10 + self._gr_k_55) + tmp_15*(self._gr_k_56*cmath.cos(5*z) + self._gr_k_57*tmp_3 + self._gr_k_58*tmp_8) + tmp_2*(self._gr_k_68*tmp_4 + self._gr_k_70 + tmp_0*(self._gr_k_64*tmp_10 + self._gr_k_65)) + tmp_7*(self._gr_k_62*tmp_8 + self._gr_k_63*tmp_3) + tmp_9*(self._gr_k_59*tmp_41 + self._gr_k_60*tmp_10 + self._gr_k_61))/tmp_39 + self._gr_k_83*tmp_33*(tmp_0 + tmp_40) + 1.0)


    cpdef double get(self, str expression_name):
        if expression_name == "edot":
            return self.edotExpResumParser
        elif expression_name == "xavg_omegainst":
            return self.xAvgOmegaInstParser
        elif expression_name == "zdot":
            return self.zdotParser
        raise RuntimeError(f"Unsupported expression '{expression_name}'")

    def initialize(self, *, chiA, chiS, delta, flagPN1, flagPN2, flagPN3, flagPN32, flagPN52, nu):
        ret = self._initialize(chiA, chiS, delta, flagPN1, flagPN2, flagPN3, flagPN32, flagPN52, nu)
        self._initialized = True
        return ret

    cpdef void compute(self
        , double e
        , double omega
        , double z
    ):
        if not self._initialized:
            raise RuntimeError("Instance has not been initialized yet")
        self._compute(e, omega, z)
