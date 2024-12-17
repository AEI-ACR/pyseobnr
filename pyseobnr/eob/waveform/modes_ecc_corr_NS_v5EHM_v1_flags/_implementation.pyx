# cython: language_level=3
# distutils: language = c++

from abc import abstractmethod

cimport libc.math as cmath
from libcpp cimport bool
cimport libcpp.complex as ccomplex

from ._implementation cimport hlm_ecc_corr_NS_v5EHM_v1_flags

cdef:
    double M_EULER_GAMA = 0.577215664901532860606512090082


cdef class BaseModesCalculation:
    """Base class for the modes calculation"""

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

    cpdef ccomplex.complex[double] get(self, int l, int m):
        raise NotImplementedError



cdef class hlm_ecc_corr_NS_v5EHM_v1_flags(BaseModesCalculation):
    """
    Eccentricity corrections to the waveform modes. These equations
    correspond to Eq. (113) of [Gamboa2024]_. See also
    "docs/mathematica/modes/".
    """

    def __cinit__(self):
        self._initialized = False

    cdef void _initialize(self
            , int flagMemory=-1
            , int flagPA=-1
            , int flagPN1=-1
            , int flagPN12=-1
            , int flagPN2=-1
            , int flagPN3=-1
            , int flagPN32=-1
            , int flagPN52=-1
            , int flagTail=-1
            , double nu=-1
        ):

        # internal computations intermediate variables declaration/initialisation
        cdef:
            double tmp_init_0 = 496.0*nu
            double tmp_init_1 = 0.69314718055994531
            double tmp_init_2 = 624.0*nu
            ccomplex.complex[double] tmp_init_3 = ccomplex.complex[double](0, 2.0)
            double tmp_init_4 = 4.0*tmp_init_1
            double tmp_init_5 = tmp_init_4 + 1.0
            ccomplex.complex[double] tmp_init_6 = ccomplex.complex[double](0, 3.0)
            double tmp_init_7 = 1.0986122886681097
            double tmp_init_8 = 405.0*tmp_init_7
            double tmp_init_9 = 4.0*nu
            double tmp_init_10 = 42.0*nu
            double tmp_init_11 = 336.0*nu
            double tmp_init_12 = 1068.0*nu
            double tmp_init_13 = 54.0*nu
            double tmp_init_14 = tmp_init_13 + 3.0
            double tmp_init_15 = 2.0*nu
            double tmp_init_16 = 13.0*nu
            ccomplex.complex[double] tmp_init_17 = ccomplex.complex[double](0, 10.0)
            ccomplex.complex[double] tmp_init_18 = ccomplex.complex[double](0, nu)
            double tmp_init_19 = 9.4247779607693797
            double tmp_init_20 = nu*tmp_init_1
            double tmp_init_21 = 10935.0*tmp_init_7
            double tmp_init_22 = 1.6094379124341004
            double tmp_init_23 = 729.0*tmp_init_7
            double tmp_init_24 = 25.0*nu
            double tmp_init_25 = 25.0*tmp_init_20 + 126.0
            double tmp_init_26 = 2.0*tmp_init_1
            ccomplex.complex[double] tmp_init_27 = ccomplex.complex[double](0, 18.0)
            double tmp_init_28 = 26.0*nu
            double tmp_init_29 = 12.566370614359173
            double tmp_init_30 = 180.0*nu
            ccomplex.complex[double] tmp_init_31 = ccomplex.complex[double](0, 4.0)
            double tmp_init_32 = 18.849555921538759
            double tmp_init_33 = nu - 4.0
            double tmp_init_34 = 6.2831853071795865
            ccomplex.complex[double] tmp_init_35 = ccomplex.complex[double](0, tmp_init_5)
            ccomplex.complex[double] tmp_init_36 = tmp_init_34 - tmp_init_35
            double tmp_init_37 = 6.0*nu
            double tmp_init_38 = 3.0*nu
            double tmp_init_39 = 15.707963267948966
            double tmp_init_40 = 7290.0*tmp_init_7
            double tmp_init_41 = -tmp_init_40
            double tmp_init_42 = 30.0*nu
            double tmp_init_43 = tmp_init_30 - 95.0
            double tmp_init_44 = 31.0*nu
            ccomplex.complex[double] tmp_init_45 = ccomplex.complex[double](0, 4320.0)
            ccomplex.complex[double] tmp_init_46 = <double>(flagTail)*tmp_init_45
            double tmp_init_47 = tmp_init_38 - 1.0
            double tmp_init_48 = flagTail*tmp_init_47
            double tmp_init_49 = 22.0*nu
            double tmp_init_50 = 4523.8934211693023
            double tmp_init_51 = tmp_init_15 - 1.0
            double tmp_init_52 = flagTail*tmp_init_51
            double tmp_init_53 = 120.0*tmp_init_1
            double tmp_init_54 = 36.0*nu
            ccomplex.complex[double] tmp_init_55 = ccomplex.complex[double](0, tmp_init_54)
            ccomplex.complex[double] tmp_init_56 = ccomplex.complex[double](0, tmp_init_52)
            double tmp_init_57 = cmath.pow(nu, 2)
            double tmp_init_58 = 168.0*nu
            double tmp_init_59 = 12780.0*tmp_init_57
            double tmp_init_60 = 225792.0*nu - 564480.0
            double tmp_init_61 = 1.9459101490553133
            double tmp_init_62 = 1272306707.0*tmp_init_61
            double tmp_init_63 = 37.699111843077519
            double tmp_init_64 = nu*tmp_init_7
            double tmp_init_65 = 410625.0*tmp_init_22
            double tmp_init_66 = 5265.0*tmp_init_7
            double tmp_init_67 = 5.0*nu
            double tmp_init_68 = 72.256631032565244
            double tmp_init_69 = 21.991148575128553
            double tmp_init_70 = 1130.9733552923256
            double tmp_init_71 = 32805.0*tmp_init_7
            ccomplex.complex[double] tmp_init_72 = ccomplex.complex[double](0, 5.0)
            double tmp_init_73 = 39.0*nu
            ccomplex.complex[double] tmp_init_74 = ccomplex.complex[double](0, 30.0)
            double tmp_init_75 = 53379.0*tmp_init_7
            double tmp_init_76 = 31.415926535897932
            double tmp_init_77 = 5654.8667764616278
            ccomplex.complex[double] tmp_init_78 = ccomplex.complex[double](0, 6.0)
            ccomplex.complex[double] tmp_init_79 = ccomplex.complex[double](0, 36.0)
            double tmp_init_80 = 205.0*nu
            double tmp_init_81 = 9.0*nu
            double tmp_init_82 = 1008.0*nu
            ccomplex.complex[double] tmp_init_83 = ccomplex.complex[double](0, 24.0)
            ccomplex.complex[double] tmp_init_84 = <double>(flagPA)*tmp_init_18
            ccomplex.complex[double] tmp_init_85 = <double>(flagMemory)*tmp_init_18
            ccomplex.complex[double] tmp_init_86 = ccomplex.complex[double](0, tmp_init_82)
            ccomplex.complex[double] tmp_init_87 = ccomplex.complex[double](0, flagTail)
            double tmp_init_88 = 31680.0*nu - 10560.0
            double tmp_init_89 = 38016.0*nu
            double tmp_init_90 = cmath.pow(tmp_init_47, 2)
            ccomplex.complex[double] tmp_init_91 = ccomplex.complex[double](0, 950400.0*tmp_init_90)
            ccomplex.complex[double] tmp_init_92 = tmp_init_87*tmp_init_90
            double tmp_init_93 = 3.1415926535897932*tmp_init_47
            double tmp_init_94 = 160.0*tmp_init_93
            double tmp_init_95 = nu*(960.0*tmp_init_1 - 1193.0) - 320.0*tmp_init_1 + 336.0
            double tmp_init_96 = flagTail*tmp_init_90
            double tmp_init_97 = cmath.pow(tmp_init_90, -1)
            double tmp_init_98 = flagPN52*tmp_init_97
            double tmp_init_99 = 0.015625*flagPN52
            double tmp_init_100 = 0.0625*flagPN32
            double tmp_init_101 = cmath.pow(nu, 3)
            double tmp_init_102 = cmath.pow(nu, 4)
            ccomplex.complex[double] tmp_init_103 = tmp_init_85*tmp_init_93
            double tmp_init_104 = flagMemory*nu
            double tmp_init_105 = tmp_init_90*(tmp_init_15 - 5.0)
            double tmp_init_106 = tmp_init_104*tmp_init_47
            double tmp_init_107 = flagPN3*tmp_init_97
            double tmp_init_108 = 20.0*nu
            double tmp_init_109 = 56.548667764616278
            double tmp_init_110 = 50.0*nu
            double tmp_init_111 = 614.0*nu
            double tmp_init_112 = 8.0*nu
            double tmp_init_113 = 160.0*nu
            double tmp_init_114 = 3392.9200658769767
            double tmp_init_115 = 6561.0*tmp_init_7
            double tmp_init_116 = 47.123889803846899
            double tmp_init_117 = 12.0*nu
            double tmp_init_118 = 235.61944901923449
            double tmp_init_119 = tmp_init_9 - 9.0
            double tmp_init_120 = 94.247779607693797
            ccomplex.complex[double] tmp_init_121 = ccomplex.complex[double](0, 9.0)
            double tmp_init_122 = 7.0*nu
            double tmp_init_123 = 27.0*nu
            double tmp_init_124 = 128.0*nu
            double tmp_init_125 = 112.0*nu
            double tmp_init_126 = 64.0*nu
            double tmp_init_127 = tmp_init_117 - 251.0
            double tmp_init_128 = 56.0*nu
            double tmp_init_129 = 81.0 - tmp_init_54
            double tmp_init_130 = tmp_init_122 + 16.0
            double tmp_init_131 = 3.1415926535897932*tmp_init_130
            double tmp_init_132 = 320.0*tmp_init_131
            double tmp_init_133 = 10.0*tmp_init_1 + 7.0
            double tmp_init_134 = 4480.0*tmp_init_1
            double tmp_init_135 = tmp_init_10 - 267.0
            double tmp_init_136 = 114.0*nu - 147.0
            double tmp_init_137 = 96.0*nu
            double tmp_init_138 = tmp_init_137 - 2784.0
            double tmp_init_139 = 48.0*nu
            double tmp_init_140 = 190.0*nu - 359.0
            double tmp_init_141 = nu + 4.0
            ccomplex.complex[double] tmp_init_142 = ccomplex.complex[double](0, tmp_init_1)
            ccomplex.complex[double] tmp_init_143 = ccomplex.complex[double](0, tmp_init_7)
            double tmp_init_144 = 328.0*nu
            double tmp_init_145 = 1330.0*nu - 2291.0
            double tmp_init_146 = 4855.0*nu - 10334.0
            double tmp_init_147 = 123.0*nu
            double tmp_init_148 = 1832.0*nu - 5023.0
            double tmp_init_149 = 149.0*nu - 412.0
            double tmp_init_150 = 66015.0*tmp_init_7
            double tmp_init_151 = 188.49555921538759
            double tmp_init_152 = 219.91148575128553
            double tmp_init_153 = 171062500.0*tmp_init_22
            double tmp_init_154 = -58671875.0*tmp_init_22
            double tmp_init_155 = -1120.0*tmp_init_1
            double tmp_init_156 = -30375.0*tmp_init_7
            ccomplex.complex[double] tmp_init_157 = ccomplex.complex[double](tmp_init_39, -tmp_init_133)
            ccomplex.complex[double] tmp_init_158 = ccomplex.complex[double](0, tmp_init_141)
            double tmp_init_159 = 38.0*nu
            double tmp_init_160 = 311.0*nu
            double tmp_init_161 = 141.3716694115407
            double tmp_init_162 = 861.0*nu
            ccomplex.complex[double] tmp_init_163 = 39600.0*tmp_init_18
            ccomplex.complex[double] tmp_init_164 = 19800.0*tmp_init_18
            ccomplex.complex[double] tmp_init_165 = ccomplex.complex[double](0, 32.0)
            double tmp_init_166 = 560.0*tmp_init_1
            ccomplex.complex[double] tmp_init_167 = 158400.0*tmp_init_18
            ccomplex.complex[double] tmp_init_168 = -316800.0*tmp_init_18
            double tmp_init_169 = 10.0*nu
            ccomplex.complex[double] tmp_init_170 = ccomplex.complex[double](0, -tmp_init_133 + 10.0*tmp_init_7)
            ccomplex.complex[double] tmp_init_171 = tmp_init_170 + tmp_init_39
            ccomplex.complex[double] tmp_init_172 = ccomplex.complex[double](0, 8.0)
            double tmp_init_173 = 14.0*nu
            double tmp_init_174 = tmp_init_173 - 1.0
            double tmp_init_175 = 2556.0*nu
            double tmp_init_176 = 70.0*nu - 5.0
            double tmp_init_177 = 4298.0*nu
            double tmp_init_178 = 58.0*nu + 1567.0
            double tmp_init_179 = 2398.0*nu - 3941.0
            double tmp_init_180 = 490.0*nu
            double tmp_init_181 = tmp_init_180 - 305.0
            double tmp_init_182 = 127.0*nu
            double tmp_init_183 = tmp_init_182 + 412.0
            double tmp_init_184 = 277.0*nu
            double tmp_init_185 = tmp_init_184 + 2641.0
            double tmp_init_186 = 1585572380.0*tmp_init_61
            double tmp_init_187 = 1354728235.0*tmp_init_61
            ccomplex.complex[double] tmp_init_188 = ccomplex.complex[double](0, flagMemory)
            double tmp_init_189 = 31250.0*tmp_init_22
            double tmp_init_190 = 288.0*nu
            double tmp_init_191 = 5103.0*tmp_init_7
            ccomplex.complex[double] tmp_init_192 = 11880.0*tmp_init_18
            double tmp_init_193 = 122145.12237157116
            double tmp_init_194 = 1272.3450247038663
            ccomplex.complex[double] tmp_init_195 = ccomplex.complex[double](0, tmp_init_15)
            ccomplex.complex[double] tmp_init_196 = ccomplex.complex[double](0, tmp_init_169)
            ccomplex.complex[double] tmp_init_197 = ccomplex.complex[double](0, tmp_init_9)
            double tmp_init_198 = cmath.pow(tmp_init_51, -1)
            double tmp_init_199 = flagPN3*tmp_init_198
            double tmp_init_200 = 3.1415926535897932*tmp_init_51
            double tmp_init_201 = 140.0*tmp_init_1 + 181.0
            ccomplex.complex[double] tmp_init_202 = ccomplex.complex[double](0, tmp_init_201)
            ccomplex.complex[double] tmp_init_203 = ccomplex.complex[double](0, tmp_init_37)
            double tmp_init_204 = 2240.0*tmp_init_200
            ccomplex.complex[double] tmp_init_205 = tmp_init_152 - tmp_init_202
            ccomplex.complex[double] tmp_init_206 = 8.0*tmp_init_18
            ccomplex.complex[double] tmp_init_207 = -tmp_init_206
            double tmp_init_208 = 2496.0*nu - 1248.0
            double tmp_init_209 = cmath.pow(tmp_init_51, 2)
            ccomplex.complex[double] tmp_init_210 = ccomplex.complex[double](0, tmp_init_209)
            double tmp_init_211 = flagTail*tmp_init_209
            ccomplex.complex[double] tmp_init_212 = tmp_init_209*tmp_init_87
            double tmp_init_213 = flagPN3/tmp_init_209
            double tmp_init_214 = 1632960.0*tmp_init_200
            ccomplex.complex[double] tmp_init_215 = ccomplex.complex[double](0, tmp_init_201 - 140.0*tmp_init_7)
            ccomplex.complex[double] tmp_init_216 = tmp_init_152 - tmp_init_215
            double tmp_init_217 = 0.0001220703125*flagPN3
            double tmp_init_218 = 2872.0*tmp_init_57
            double tmp_init_219 = 4.3402777777777778e-5*flagPN2
            double tmp_init_220 = 0.25*flagPN12
            double tmp_init_221 = flagPN52*tmp_init_198
            double tmp_init_222 = cmath.pow(5.0*tmp_init_57 - tmp_init_67 + 1.0, -1)
            double tmp_init_223 = flagPN3*tmp_init_222
            double tmp_init_224 = 85050.0*tmp_init_101
            double tmp_init_225 = 196.0*tmp_init_101
            double tmp_init_226 = 102060.0*tmp_init_101
            double tmp_init_227 = 836640.0*tmp_init_57
            double tmp_init_228 = cmath.pow(tmp_init_47, -1)
            double tmp_init_229 = flagPN2*tmp_init_228
            double tmp_init_230 = 9.8696044010893586
            double tmp_init_231 = 15280650.0*tmp_init_230
            ccomplex.complex[double] tmp_init_232 = ccomplex.complex[double](0, 3.1415926535897932)
            ccomplex.complex[double] tmp_init_233 = 95672586240.0*tmp_init_232 - 110447430958.74169
            double tmp_init_234 = 779.0*nu
            double tmp_init_235 = 451.0*nu
            double tmp_init_236 = 41.0*nu
            double tmp_init_237 = nu*tmp_init_230
            double tmp_init_238 = tmp_init_230*(tmp_init_236 - 320.0)
            double tmp_init_239 = 55223715479.370844
            double tmp_init_240 = 816.0*tmp_init_57
            ccomplex.complex[double] tmp_init_241 = 3.1415926535897932*tmp_init_85
            double tmp_init_242 = 22353408000.0*tmp_init_57
            double tmp_init_243 = 22711600.0*tmp_init_101
            double tmp_init_244 = 2546775.0*tmp_init_230
            ccomplex.complex[double] tmp_init_245 = <double>(flagPN2)*tmp_init_198*tmp_init_85
            ccomplex.complex[double] tmp_init_246 = <double>(flagPN32)*tmp_init_228*tmp_init_85
            double tmp_init_247 = 39223800.0*tmp_init_102
            double tmp_init_248 = 437500.0*tmp_init_101
            ccomplex.complex[double] tmp_init_249 = <double>(flagPN3)*tmp_init_85/(3.0*tmp_init_57 - tmp_init_9 + 1.0)
            double tmp_init_250 = 32340.0*tmp_init_101
            ccomplex.complex[double] tmp_init_251 = <double>(flagPN52)*tmp_init_222*tmp_init_85

        # computations
        self._gr_k_0 = -3.1415926535897932*tmp_init_0 + tmp_init_3*(124.0*nu + tmp_init_1*tmp_init_2 - 1460.0*tmp_init_1 + 1331.0) + 17052.564923685398
        self._gr_k_1 = 163.36281798666925*nu + tmp_init_6*(4.0*nu*tmp_init_5 - 36.0*tmp_init_1 - 121.0) - 2456.7254551072183
        self._gr_k_2 = ccomplex.complex[double](20979.555740672639 - 2638.9378290154263*nu, -21292.0*tmp_init_1 + 22356.0*tmp_init_7 + tmp_init_9*(356.0*tmp_init_1 - tmp_init_8 + 105.0) + 357.0)
        self._gr_k_3 = tmp_init_10 - 343.0
        self._gr_k_4 = tmp_init_11 - 413.0
        self._gr_k_5 = tmp_init_12 - 2071.0
        self._gr_k_6 = tmp_init_14
        self._gr_k_7 = 43.982297150257105*tmp_init_15 + tmp_init_17*(tmp_init_1*tmp_init_16 + 488.0*tmp_init_1 - 126.0) + 6553.3622753883087
        self._gr_k_8 = 2968.8050576423546*tmp_init_16 + 81.0*tmp_init_18*(2118.0*tmp_init_1 - 835.0*tmp_init_7) + tmp_init_3*(83848.0*tmp_init_1 - 92961.0*tmp_init_7 - 2898.0) - 71251.321383416511
        self._gr_k_9 = ccomplex.complex[double](tmp_init_19*(2841.0*nu - 6680.0), nu*tmp_init_21 - 127696.0*tmp_init_1 - 8310.0*tmp_init_20 + 69498.0*tmp_init_7 - 18396.0)
        self._gr_k_10 = ccomplex.complex[double](13599.954597390215*nu - 10580.884057290424, -213298.0*nu*tmp_init_1 + 42096.0*tmp_init_1 - 77750.0*tmp_init_22 + tmp_init_24*(3625.0*tmp_init_22 + tmp_init_23) + 70470.0*tmp_init_7 - 756.0)
        self._gr_k_11 = 273.31856086231201*nu + tmp_init_3*(-454.0*tmp_init_1 + tmp_init_25) + 18.849555921538759
        self._gr_k_12 = -775.97338543667893*nu - tmp_init_27*(15.0*nu*(-tmp_init_1 + tmp_init_7) - tmp_init_26 + 2.0*tmp_init_7 - 42.0) + 1350.8848410436111
        self._gr_k_13 = tmp_init_29*(tmp_init_28 - 1567.0) - tmp_init_31*(816.0*nu*tmp_init_1 - 3628.0*tmp_init_1 - tmp_init_30 + 1245.0)
        self._gr_k_14 = ccomplex.complex[double](-tmp_init_32*(2729.0*nu - 19785.0), 6.0*nu*(5240.0*tmp_init_1 - tmp_init_21 + 1374.0) - 880080.0*tmp_init_1 + 1007478.0*tmp_init_7 - 46836.0)
        self._gr_k_15 = -78414.152633601239*nu - tmp_init_31*(-2011604.0*tmp_init_1 + tmp_init_15*(83104.0*tmp_init_1 - 39609.0*tmp_init_7 - 1560.0) + 1087425.0*tmp_init_7 + 15315.0) + 473576.24297273979
        self._gr_k_16 = ccomplex.complex[double](3088.1855784787668*nu - 60117.517019094283, nu*(8296.0*tmp_init_1 - 7533.0*tmp_init_7 + 474.0) - 112816.0*tmp_init_1 + 108378.0*tmp_init_7 - 10716.0)
        self._gr_k_17 = ccomplex.complex[double](111589.37105550946 - 28161.236546778907*nu, nu*(733960.0*tmp_init_1 - 265625.0*tmp_init_22 - 109350.0*tmp_init_7 + 4482.0) - 7759232.0*tmp_init_1 + 2271750.0*tmp_init_22 + 1733562.0*tmp_init_7 - 14736.0)
        self._gr_k_18 = 7.0*tmp_init_33*tmp_init_36
        self._gr_k_19 = ccomplex.complex[double](1043.0087609918114 - 3.1415926535897932*tmp_init_37, -112.0*tmp_init_1 + 28.0*tmp_init_20 + tmp_init_38 + 170.0)
        self._gr_k_20 = -tmp_init_18*(7290.0*tmp_init_1 + tmp_init_41 + 3479.0) + tmp_init_39*(497.0*nu + 500.0) + 20.0*ccomplex.complex[double](0, 1)*(1458.0*tmp_init_1 - 1458.0*tmp_init_7 + 149.0)
        self._gr_k_21 = ccomplex.complex[double](47815.040187636653 - 21504.201713822135*nu, nu*(890.0*tmp_init_1 + tmp_init_41 + 9583.0) - 27880.0*tmp_init_1 + 29160.0*tmp_init_7 - 4028.0)
        self._gr_k_22 = tmp_init_42 + 459.0
        self._gr_k_23 = 1118.0 - 708.0*nu
        self._gr_k_24 = -tmp_init_43
        self._gr_k_25 = tmp_init_44 + 104.0
        self._gr_k_26 = 92.0 - 101.0*nu
        self._gr_k_27 = flagTail
        self._gr_k_28 = flagTail
        self._gr_k_29 = -tmp_init_46
        self._gr_k_30 = tmp_init_48
        self._gr_k_31 = ccomplex.complex[double](0, 540.0*tmp_init_48)
        self._gr_k_32 = tmp_init_45*tmp_init_47
        self._gr_k_33 = ccomplex.complex[double](tmp_init_47*tmp_init_50, 2160.0 - 432.0*tmp_init_49)
        self._gr_k_34 = ccomplex.complex[double](0, -tmp_init_13)
        self._gr_k_35 = tmp_init_52
        self._gr_k_36 = ccomplex.complex[double](-tmp_init_50*tmp_init_51, 48.0*nu*(tmp_init_53 + 1661.0) - 2880.0*tmp_init_1 - 3072.0)
        self._gr_k_37 = -tmp_init_45*tmp_init_51
        self._gr_k_38 = -tmp_init_55
        self._gr_k_39 = -1080.0*tmp_init_56
        self._gr_k_40 = 11214.0*nu + 1323.0*tmp_init_57 + 4662.0
        self._gr_k_41 = 149304.0*nu + 3240.0*tmp_init_57 - 1007694.0
        self._gr_k_42 = -220104.0*nu - 11232.0*tmp_init_57 + 336258.0
        self._gr_k_43 = -322636.0*nu - 12426.0*tmp_init_57 + 1931116.0
        self._gr_k_44 = -43890.0*nu - 3465.0*tmp_init_57 + 1050.0
        self._gr_k_45 = tmp_init_58 - 128268.0
        self._gr_k_46 = 2982.0*nu + 441.0*tmp_init_57 - 33348.0
        self._gr_k_47 = -102922.0*nu - 3975.0*tmp_init_57 + 257182.0
        self._gr_k_48 = -35308.0*nu - 2310.0*tmp_init_57 + 14308.0
        self._gr_k_49 = 23520.0*nu + 17640.0
        self._gr_k_50 = -154256.0*nu - 192.0*tmp_init_57 + 101018.0
        self._gr_k_51 = -94280.0*nu + 60.0*tmp_init_57 - 271627.0
        self._gr_k_52 = -295080.0*nu - tmp_init_59 + 171609.0
        self._gr_k_53 = tmp_init_60
        self._gr_k_54 = 24692.0*nu + 102.0*tmp_init_57 + 3949.0
        self._gr_k_55 = -tmp_init_60
        self._gr_k_56 = 4.4288548752834467e-6*flagPN52
        self._gr_k_57 = -4178196.0*nu + 82632.0*tmp_init_57 + 10126314.0
        self._gr_k_58 = -9854690.0*nu + 1698820.0*tmp_init_57 + 7921185.0
        self._gr_k_59 = -22143960.0*nu + 3026160.0*tmp_init_57 + 33150060.0
        self._gr_k_60 = -64350.0*nu + 12540.0*tmp_init_57 - 118305.0
        self._gr_k_61 = -2087060.0*nu + 571720.0*tmp_init_57 + 224490.0
        self._gr_k_62 = -37313058.0*nu + 4740996.0*tmp_init_57 + 76843377.0
        self._gr_k_63 = -92190.0*nu + 83580.0*tmp_init_57 + 25935.0
        self._gr_k_64 = -620730.0*nu + 37620.0*tmp_init_57 - 711315.0
        self._gr_k_65 = -93855460.0*nu + 10670120.0*tmp_init_57 + 146683170.0
        self._gr_k_66 = -67457228.0*nu + 9273016.0*tmp_init_57 + 67134582.0
        self._gr_k_67 = -20889602.0*nu + 4152484.0*tmp_init_57 + 7954113.0
        self._gr_k_68 = -18792838.0*nu - 162004.0*tmp_init_57 + 34973427.0
        self._gr_k_69 = -1780110.0*nu + 752220.0*tmp_init_57 + 138375.0
        self._gr_k_70 = -6232080.0*nu + 115110.0*tmp_init_57 + 1281480.0
        self._gr_k_71 = -9559926.0*nu + 1105782.0*tmp_init_57 + 3667989.0
        self._gr_k_72 = -254894.0*nu + 6178.0*tmp_init_57 - 2009199.0
        self._gr_k_73 = -2312248.0*nu + 474026.0*tmp_init_57 + 247512.0
        self._gr_k_74 = -4710060.0*nu + 158460.0*tmp_init_57 - 2152020.0
        self._gr_k_75 = -169364.0*nu + 12238.0*tmp_init_57 - 3545859.0
        self._gr_k_76 = -2961196.0*nu + 432902.0*tmp_init_57 - 218571.0
        self._gr_k_77 = 77020.0*nu + 2020.0*tmp_init_57 - 725565.0
        self._gr_k_78 = -18915072.0*nu + 2855424.0*tmp_init_57 + 41171328.0
        self._gr_k_79 = -90972344.0*nu + 14104528.0*tmp_init_57 + 44680716.0
        self._gr_k_80 = -173419620.0*nu + 18642360.0*tmp_init_57 + 167883210.0
        self._gr_k_81 = -2167638.0*nu + 9636.0*tmp_init_57 - 7930593.0
        self._gr_k_82 = -59216440.0*nu + 296720.0*tmp_init_57 + 57152460.0
        self._gr_k_83 = -12512790.0*nu + 3494340.0*tmp_init_57 + 1427295.0
        self._gr_k_84 = -288514.0*nu + 16928.0*tmp_init_57 - 299754.0
        self._gr_k_85 = 20528640.0*nu - 51321600.0
        self._gr_k_86 = 1425600.0 - 570240.0*nu
        self._gr_k_87 = 1.4613729891507669e-7*flagPN52
        self._gr_k_88 = 55.0*nu - 107.0
        self._gr_k_89 = ccomplex.complex[double](0, -3732904704.0*tmp_init_1 + 2511921875.0*tmp_init_22 + tmp_init_38*(1438742848.0*tmp_init_1 - 795390625.0*tmp_init_22 - 305534453.0*tmp_init_61 + 798357465.0*tmp_init_7) + tmp_init_62 - 3577874247.0*tmp_init_7)
        self._gr_k_90 = tmp_init_3*(282844.0*tmp_init_1 + 157704.0*tmp_init_20 - 74115.0*tmp_init_64 - 241218.0*tmp_init_7 + 1512.0) + tmp_init_63*(642.0*nu - 1453.0)
        self._gr_k_91 = ccomplex.complex[double](17002.299441227961*nu - 26734.95348204914, -4404.0*tmp_init_1 - 1250512.0*tmp_init_20 + tmp_init_24*(18125.0*tmp_init_22 + 6804.0*tmp_init_7) - tmp_init_65 + 536382.0*tmp_init_7)
        self._gr_k_92 = ccomplex.complex[double](11539.069816635311*nu - 24136.856357530381, nu*tmp_init_66 - 133068.0*tmp_init_1 - 2960.0*tmp_init_20 + 76113.0*tmp_init_7 - 6048.0)
        self._gr_k_93 = ccomplex.complex[double](tmp_init_68*(tmp_init_67 - 18.0), -tmp_init_26*(90.0*nu + 481.0))
        self._gr_k_94 = ccomplex.complex[double](tmp_init_69*(tmp_init_30 - 71.0), nu*(728324.0*tmp_init_1 - 453125.0*tmp_init_22 + 211977.0*tmp_init_7) - 344438.0*tmp_init_1 + tmp_init_65 - 388935.0*tmp_init_7)
        self._gr_k_95 = ccomplex.complex[double](tmp_init_39*(8919.0*nu - 12647.0), -159855840.0*tmp_init_1 - 28469520.0*tmp_init_20 + 15788925.0*tmp_init_64 + 106282935.0*tmp_init_7 - 362880.0)
        self._gr_k_96 = -tmp_init_17*(53431240.0*tmp_init_1 - 67091875.0*tmp_init_22 + 65037006.0*tmp_init_7) + 15.0*tmp_init_18*(81283136.0*tmp_init_1 - 47221875.0*tmp_init_22 + 18199242.0*tmp_init_7) + tmp_init_70*(516.0*nu - 655.0)
        self._gr_k_97 = 324070.99018105512*nu - tmp_init_72*(-nu*tmp_init_71 + 2374288.0*tmp_init_1 + 61824.0*tmp_init_20 - 1650942.0*tmp_init_7) - 1244416.266013453
        self._gr_k_98 = 32421.236185046666*tmp_init_73 - tmp_init_74*(24273872.0*nu*tmp_init_1 - 7703125.0*nu*tmp_init_22 + 3688048.0*tmp_init_1 + 6958750.0*tmp_init_22 - 4439286.0*tmp_init_64 - 11724156.0*tmp_init_7 - 6048.0) - 2690962.6033588733
        self._gr_k_99 = 165.0*nu - 321.0
        self._gr_k_100 = tmp_init_19*(1409.0*nu - 53686.0) + tmp_init_6*(-nu*tmp_init_75 - 176816.0*tmp_init_1 + 67104.0*tmp_init_20 + 236466.0*tmp_init_7)
        self._gr_k_101 = tmp_init_17*(20518064.0*tmp_init_1 + 6749232.0*tmp_init_20 - 3510135.0*tmp_init_64 - 14946930.0*tmp_init_7 + 90720.0) + tmp_init_76*(37839.0*nu - 82326.0)
        self._gr_k_102 = ccomplex.complex[double](tmp_init_77*(18.0*nu + 1.0), 456621808.0*tmp_init_1 - 310237500.0*tmp_init_22 - tmp_init_38*(186875744.0*tmp_init_1 - 104218750.0*tmp_init_22 - 43647779.0*tmp_init_61 + 111973185.0*tmp_init_7) - 164463698.0*tmp_init_61 + 457658910.0*tmp_init_7)
        self._gr_k_103 = tmp_init_19*(45.0*nu - 127.0) + tmp_init_78*(-685.0*tmp_init_1 + tmp_init_25)
        self._gr_k_104 = tmp_init_34*(17967.0*nu + 4147.0) - tmp_init_79*(16.0*nu*(1211.0*tmp_init_1 - 891.0*tmp_init_7) - 42112.0*tmp_init_1 + 60507.0*tmp_init_7)
        self._gr_k_105 = tmp_init_17*(93608576.0*nu*tmp_init_1 - 29.0*nu*(921875.0*tmp_init_22 + 713691.0*tmp_init_7) + 28973696.0*tmp_init_1 + 21558125.0*tmp_init_22 - 48009267.0*tmp_init_7 - 12096.0) + tmp_init_76*(tmp_init_80 + 809.0)
        self._gr_k_106 = -tmp_init_72*(-343449824.0*tmp_init_1 + 396971875.0*tmp_init_22 + tmp_init_38*(241287568.0*tmp_init_1 - 131890625.0*tmp_init_22 + 41369778.0*tmp_init_7) - 366532938.0*tmp_init_7)
        self._gr_k_107 = -1520028.1895128856*nu
        self._gr_k_108 = ccomplex.complex[double](tmp_init_69*(43.0*nu - 3675.0), -2482144.0*tmp_init_1 + 1463103.0*tmp_init_7 + tmp_init_81*(84784.0*tmp_init_1 - tmp_init_75))
        self._gr_k_109 = ccomplex.complex[double](0, -nu*(2314597424.0*tmp_init_1 - 675156250.0*tmp_init_22 - 916603359.0*tmp_init_61 + 1152275625.0*tmp_init_7) + 2767796320.0*tmp_init_1 - 742643750.0*tmp_init_22 - tmp_init_62 + 1595227689.0*tmp_init_7)
        self._gr_k_110 = -6748.1410199108759*nu + tmp_init_78*(-nu*tmp_init_8 + 1684.0*tmp_init_1 + 380.0*tmp_init_20 - 999.0*tmp_init_7 + 1008.0) + 15092.211107845367
        self._gr_k_111 = -tmp_init_3*(4479.0*nu*tmp_init_1 + 1237.0*tmp_init_1 - 1215.0*tmp_init_64 - 2997.0*tmp_init_7 - 630.0) - tmp_init_68*(69.0*nu - 79.0)
        self._gr_k_112 = tmp_init_10 - 511.0
        self._gr_k_113 = 1995.0 - tmp_init_82
        self._gr_k_114 = 3163.0 - tmp_init_12
        self._gr_k_115 = -270.0*nu - 15.0
        self._gr_k_116 = 1416.0*nu - 3496.0
        self._gr_k_117 = 720.0*nu - 527.0
        self._gr_k_118 = tmp_init_44 + 6.0
        self._gr_k_119 = 430.0 - 303.0*nu
        self._gr_k_120 = tmp_init_11
        self._gr_k_121 = 84672000.0*nu*(tmp_init_83 + 3.1415926535897932)
        self._gr_k_122 = 588.0*tmp_init_84
        self._gr_k_123 = 372.0*nu - 656.0
        self._gr_k_124 = -tmp_init_58
        self._gr_k_125 = ccomplex.complex[double](0, tmp_init_58)
        self._gr_k_126 = 59908.0 - 81192.0*nu
        self._gr_k_127 = 71885.0 - 77454.0*nu
        self._gr_k_128 = 85524.0*nu - 51713.0
        self._gr_k_129 = 250.0*tmp_init_85
        self._gr_k_130 = 175.0*flagTail
        self._gr_k_131 = tmp_init_86
        self._gr_k_132 = tmp_init_86
        self._gr_k_133 = -589010923.43624316*nu - 1016064000.0*tmp_init_18 + 1038685929.5004718
        self._gr_k_134 = -189000.0*tmp_init_87
        self._gr_k_135 = -12600.0*tmp_init_18
        self._gr_k_136 = 1.1810279667422525e-8*flagPN52
        self._gr_k_137 = 198000.0*nu - 66000.0
        self._gr_k_138 = -355525.0*nu + 63660.0*tmp_init_57 + 106335.0
        self._gr_k_139 = 12672.0*nu - 4224.0
        self._gr_k_140 = 608256.0*nu - 202752.0
        self._gr_k_141 = tmp_init_88
        self._gr_k_142 = 3041280.0*nu - 1013760.0
        self._gr_k_143 = 63360.0 - 190080.0*nu
        self._gr_k_144 = 37224.0 - 111672.0*nu
        self._gr_k_145 = 12672.0 - tmp_init_89
        self._gr_k_146 = -962797.0*nu + 851646.0*tmp_init_57 + 225795.0
        self._gr_k_147 = 1782.0 - 5346.0*nu
        self._gr_k_148 = tmp_init_85
        self._gr_k_149 = tmp_init_91
        self._gr_k_150 = 59400.0*tmp_init_92
        self._gr_k_151 = ccomplex.complex[double](1980.0*tmp_init_47*tmp_init_94, 1980.0*tmp_init_47*tmp_init_95)
        self._gr_k_152 = -5940.0*tmp_init_18*tmp_init_47
        self._gr_k_153 = -110.0*tmp_init_96
        self._gr_k_154 = 2.4660669191919192e-8*tmp_init_98
        self._gr_k_155 = 5.0e-6*flagPN52
        self._gr_k_156 = 1.3281030862990761e-7*flagPN52
        self._gr_k_157 = 7.1444901691815272e-6*flagPN52
        self._gr_k_158 = tmp_init_99
        self._gr_k_159 = 2.0e-7*flagPN52
        self._gr_k_160 = 7.0*flagTail
        self._gr_k_161 = 15120.0*tmp_init_87
        self._gr_k_162 = -320.0*tmp_init_85
        self._gr_k_163 = 1.2400793650793651e-5*flagPN32
        self._gr_k_164 = tmp_init_100
        self._gr_k_165 = 0.0023148148148148148*flagPN32
        self._gr_k_166 = 0.0001*flagPN32
        self._gr_k_167 = tmp_init_100
        self._gr_k_168 = 0.00025720164609053498*flagPN32
        self._gr_k_169 = -4556829662208.0*nu - 3261633177600.0*tmp_init_101 + 435409551360.0*tmp_init_102 + 8503338074112.0*tmp_init_57 + 690166738944.0
        self._gr_k_170 = -4006247190505.0*nu - 2829114303120.0*tmp_init_101 + 327177601380.0*tmp_init_102 + 7483352903985.0*tmp_init_57 + 605207252706.0
        self._gr_k_171 = -300741101305.0*nu - 586526228400.0*tmp_init_101 + 93896831700.0*tmp_init_102 + 741898246845.0*tmp_init_57 + 38450918070.0
        self._gr_k_172 = -20539086715.0*nu - 117473991600.0*tmp_init_101 + 27528410700.0*tmp_init_102 + 85175911835.0*tmp_init_57 + 1402716630.0
        self._gr_k_173 = -2667652182531.0*nu - 2220525163920.0*tmp_init_101 + 254701169100.0*tmp_init_102 + 5161566133899.0*tmp_init_57 + 395215036650.0
        self._gr_k_174 = 3526978455.0*nu - 3866662800.0*tmp_init_101 + 386486100.0*tmp_init_102 - 3837428595.0*tmp_init_57 - 610840230.0
        self._gr_k_175 = -317010248555.0*nu - 152271559440.0*tmp_init_101 - 2978915940.0*tmp_init_102 + 563034767295.0*tmp_init_57 + 48808569810.0
        self._gr_k_176 = -1288380768819.0*nu - 1465439073840.0*tmp_init_101 + 187582687740.0*tmp_init_102 + 2685068073191.0*tmp_init_57 + 183312072558.0
        self._gr_k_177 = -978907545.0*nu - 6956334000.0*tmp_init_101 + 3540744900.0*tmp_init_102 + 3998946105.0*tmp_init_57 + 95904270.0
        self._gr_k_178 = 88088.0*tmp_init_103
        self._gr_k_179 = 785109054625.0*nu - 248525084640.0*tmp_init_101 + 3283347060.0*tmp_init_102 - 1061425767535.0*tmp_init_57 - 134682620562.0
        self._gr_k_180 = -453814432404.0*nu - 1394310078000.0*tmp_init_101 + 255545335080.0*tmp_init_102 + 1355652379746.0*tmp_init_57 + 49213485054.0
        self._gr_k_181 = 206483746150.0*nu + 32671540440.0*tmp_init_101 + 650020140.0*tmp_init_102 - 327387010335.0*tmp_init_57 - 33682068288.0
        self._gr_k_182 = 74321497150.0*nu - 105655624200.0*tmp_init_101 + 5598337500.0*tmp_init_102 - 54654824875.0*tmp_init_57 - 14898593700.0
        self._gr_k_183 = 103225676010.0*nu - 547949997840.0*tmp_init_101 + 13864341960.0*tmp_init_102 + 134589422870.0*tmp_init_57 - 29375470470.0
        self._gr_k_184 = 285179286883.0*nu - 28849779000.0*tmp_init_101 + 1220107140.0*tmp_init_102 - 415862827597.0*tmp_init_57 - 47824991163.0
        self._gr_k_185 = -55026687725.0*nu - 416246959800.0*tmp_init_101 + 50537358900.0*tmp_init_102 + 294911620535.0*tmp_init_57 + 316151937.0
        self._gr_k_186 = -17528330160.0*nu - 120433143600.0*tmp_init_101 + 38948193900.0*tmp_init_102 + 78845129055.0*tmp_init_57 + 1061851725.0
        self._gr_k_187 = -2490018398340.0*nu - 5342785712880.0*tmp_init_101 + 672343499520.0*tmp_init_102 + 6434741559400.0*tmp_init_57 + 304686258624.0
        self._gr_k_188 = -6042976276870.0*nu - 8663994808560.0*tmp_init_101 + 762804937440.0*tmp_init_102 + 13559160871620.0*tmp_init_57 + 819450338460.0
        self._gr_k_189 = -1919540108100.0*nu - 2829220119600.0*tmp_init_101 - 7199236800.0*tmp_init_102 + 4417760857160.0*tmp_init_57 + 253601186400.0
        self._gr_k_190 = 303738079431.0*nu - 147735318360.0*tmp_init_101 + 923065920.0*tmp_init_102 - 386075351914.0*tmp_init_57 - 52916218734.0
        self._gr_k_191 = -232846819449.0*nu - 928595400600.0*tmp_init_101 + 178993291680.0*tmp_init_102 + 800316349686.0*tmp_init_57 + 20853242034.0
        self._gr_k_192 = -2452540299585.0*nu - 6254411530080.0*tmp_init_101 + 602809347420.0*tmp_init_102 + 6889890567875.0*tmp_init_57 + 275894583402.0
        self._gr_k_193 = -886806371457.0*nu - 3626942551680.0*tmp_init_101 + 38309646060.0*tmp_init_102 + 3264539920663.0*tmp_init_57 + 66140311638.0
        self._gr_k_194 = -457499548991.0*nu - 1820950420800.0*tmp_init_101 + 286547218020.0*tmp_init_102 + 1593429871229.0*tmp_init_57 + 39260618946.0
        self._gr_k_195 = 29362132800.0*nu - 32616183600.0*tmp_init_101 + 1159458300.0*tmp_init_102 - 30136591485.0*tmp_init_57 - 5245174935.0
        self._gr_k_196 = -9480585567320.0*nu - 9670008455520.0*tmp_init_101 + 1062661204080.0*tmp_init_102 + 19234397967820.0*tmp_init_57 + 1369334059380.0
        self._gr_k_197 = -1707537882900.0*nu - 992019918960.0*tmp_init_101 - 36410353560.0*tmp_init_102 + 3131892601810.0*tmp_init_57 + 258442072974.0
        self._gr_k_198 = -11756895343408.0*nu - 9361910787600.0*tmp_init_101 + 935097799860.0*tmp_init_102 + 22549607085257.0*tmp_init_57 + 1750088781963.0
        self._gr_k_199 = -3443180679520.0*nu - 5289046830480.0*tmp_init_101 + 705716155620.0*tmp_init_102 + 7839464883285.0*tmp_init_57 + 464381426367.0
        self._gr_k_200 = ccomplex.complex[double](3303300.0*tmp_init_104*tmp_init_95, -3303300.0*tmp_init_104*tmp_init_94)
        self._gr_k_201 = -634233600.0*tmp_init_105
        self._gr_k_202 = 324727603200.0*tmp_init_105
        self._gr_k_203 = -15756136784872.0*nu - 21761893784400.0*tmp_init_101 + 2456567938440.0*tmp_init_102 + 34784846802398.0*tmp_init_57 + 2164334059536.0
        self._gr_k_204 = 261777756240.0*nu - 174898443720.0*tmp_init_101 + 378558180.0*tmp_init_102 - 312811804305.0*tmp_init_57 - 46045419420.0
        self._gr_k_205 = -5312570982880.0*nu - 4587080555400.0*tmp_init_101 - 81989063100.0*tmp_init_102 + 10530115988455.0*tmp_init_57 + 771668477496.0
        self._gr_k_206 = -24093972955480.0*nu - 24119647172880.0*tmp_init_101 + 2160745954920.0*tmp_init_102 + 48765507364070.0*tmp_init_57 + 3481845260136.0
        self._gr_k_207 = -3435716301016.0*nu - 8064890975400.0*tmp_init_101 + 1219005455220.0*tmp_init_102 + 9158399933179.0*tmp_init_57 + 411620024916.0
        self._gr_k_208 = -197296406040.0*nu - 967025375400.0*tmp_init_101 + 233056019700.0*tmp_init_102 + 748386987915.0*tmp_init_57 + 15542894040.0
        self._gr_k_209 = 1585584000.0*tmp_init_106
        self._gr_k_210 = 6.1590082896901079e-12*tmp_init_107
        self._gr_k_211 = tmp_init_108 - 101.0
        self._gr_k_212 = tmp_init_39*(293.0*nu - 12330.0) + tmp_init_72*(nu*(26864.0*tmp_init_1 - 22599.0*tmp_init_7 - 60.0) - 330812.0*tmp_init_1 + 297918.0*tmp_init_7 - 3603.0)
        self._gr_k_213 = ccomplex.complex[double](-tmp_init_109*(68.0*nu - 601.0), 726.0*nu - 15804.0*tmp_init_1 + 3384.0*tmp_init_20 + 2829.0)
        self._gr_k_214 = ccomplex.complex[double](-tmp_init_109*(338.0*nu - 2409.0), -197652.0*tmp_init_1 + 3.0*tmp_init_15*(1820.0*tmp_init_1 - 2430.0*tmp_init_7 + 507.0) + 214812.0*tmp_init_7 - 6561.0)
        self._gr_k_215 = ccomplex.complex[double](854.51320177642376*nu - 14193.715608918686, 1668.0*tmp_init_1 - tmp_init_110*(4.0*tmp_init_1 - 3.0) - 3267.0)
        self._gr_k_216 = ccomplex.complex[double](-tmp_init_32*(tmp_init_111 - 3219.0), 429132.0*tmp_init_1 - tmp_init_15*(18316.0*tmp_init_1 - tmp_init_40 - 921.0) - 214812.0*tmp_init_7 - 4617.0)
        self._gr_k_217 = -tmp_init_109*(1096.0*nu + 3925.0) - tmp_init_78*(-43500824.0*tmp_init_1 + tmp_init_112*(419152.0*tmp_init_1 - 257823.0*tmp_init_7 + 60.0) + 26054703.0*tmp_init_7 + 12570.0)
        self._gr_k_218 = -tmp_init_114*(tmp_init_113 - 801.0) - tmp_init_78*(-3326276888.0*tmp_init_1 + tmp_init_112*(39086272.0*tmp_init_1 - 21171875.0*tmp_init_22 + 6450192.0*tmp_init_7 - 1800.0) + 1616228875.0*tmp_init_22 - 272394738.0*tmp_init_7 + 72090.0)
        self._gr_k_219 = -589111.45440115803*nu - 10560.0*tmp_init_18*(11564.0*tmp_init_1 - tmp_init_115 - 9.0) - 180.0*ccomplex.complex[double](0, 1)*(-9222808.0*tmp_init_1 + 5481999.0*tmp_init_7 + 6522.0) + 3815338.6140786603
        self._gr_k_220 = ccomplex.complex[double](-tmp_init_116*(6994.0*nu - 22075.0), -43990200.0*tmp_init_1 + 45.0*tmp_init_117*(8264.0*tmp_init_1 - tmp_init_115 + 18.0) + 38181375.0*tmp_init_7 - 324270.0)
        self._gr_k_221 = ccomplex.complex[double](-tmp_init_118*(3965.0*nu - 25163.0), 15.0*nu*(77442464.0*tmp_init_1 - 22578125.0*tmp_init_22 - 16609293.0*tmp_init_7 + 9960.0) - 13089195480.0*tmp_init_1 + 3352575000.0*tmp_init_22 + 3426121395.0*tmp_init_7 - 1132470.0)
        self._gr_k_222 = ccomplex.complex[double](21171.192892541617*nu - 400873.5057833648, nu*(15777952.0*tmp_init_1 - 3359375.0*tmp_init_22 - 5084775.0*tmp_init_7 + 3240.0) - 215550696.0*tmp_init_1 + 51304875.0*tmp_init_22 + 61979580.0*tmp_init_7 - 67770.0)
        self._gr_k_223 = ccomplex.complex[double](-25446.900494077325*tmp_init_119, nu*(837927200.0*tmp_init_1 - 458203125.0*tmp_init_22 - 167179229.0*tmp_init_61 + 438368841.0*tmp_init_7 + 16200.0) - 8539619976.0*tmp_init_1 + 4512675000.0*tmp_init_22 + 1317291843.0*tmp_init_61 - 3555462033.0*tmp_init_7 - 36450.0)
        self._gr_k_224 = ccomplex.complex[double](-tmp_init_120*(1684.0*nu - 9151.0), 5.0*nu*(4130320.0*tmp_init_1 - 1328125.0*tmp_init_22 - 753786.0*tmp_init_7 + 5052.0) - 224183220.0*tmp_init_1 + 61168750.0*tmp_init_22 + 54419850.0*tmp_init_7 - 137265.0)
        self._gr_k_225 = -tmp_init_74*(nu*(195968.0*tmp_init_1 - 103761.0*tmp_init_7 - 984.0) - 2503380.0*tmp_init_1 + 1430541.0*tmp_init_7 + 7527.0) - tmp_init_76*(5923.0*nu - 39197.0)
        self._gr_k_226 = ccomplex.complex[double](-tmp_init_76*(1901.0*nu - 12301.0), 30.0*nu*(704.0*tmp_init_1 - tmp_init_66 + 312.0) - 1713240.0*tmp_init_1 + 2349810.0*tmp_init_7 - 106110.0)
        self._gr_k_227 = 5500.928736435728*nu - tmp_init_121*(-292652.0*tmp_init_1 + 21328.0*tmp_init_20 - tmp_init_67*(2511.0*tmp_init_7 + 20.0) + 165510.0*tmp_init_7 + 2185.0) - 126021.84770610097
        self._gr_k_228 = ccomplex.complex[double](120731.40567745575 - 1979.2033717615697*tmp_init_108, -nu*(11729584.0*tmp_init_1 - 6640625.0*tmp_init_22 + 2405214.0*tmp_init_7 - 6300.0) + 123086036.0*tmp_init_1 - 61168750.0*tmp_init_22 + 12223386.0*tmp_init_7 - 19215.0)
        self._gr_k_229 = 2424.0 - 480.0*nu
        self._gr_k_230 = -161280.0*nu*tmp_init_36
        self._gr_k_231 = 420.0*nu - 3591.0
        self._gr_k_232 = 1421.0 - 140.0*nu
        self._gr_k_233 = tmp_init_54 - 753.0
        self._gr_k_234 = tmp_init_30 - 405.0
        self._gr_k_235 = tmp_init_122 + 56.0
        self._gr_k_236 = tmp_init_123 - 332.0
        self._gr_k_237 = tmp_init_117 - 363.0
        self._gr_k_238 = tmp_init_124 - 1058.0
        self._gr_k_239 = 300.0*nu - 1347.0
        self._gr_k_240 = -tmp_init_125
        self._gr_k_241 = 16.0*ccomplex.complex[double](0, 1)*nu*(32.0*tmp_init_1 - 167.0) - tmp_init_29*(tmp_init_126 - 417.0) - 834.0*tmp_init_35
        self._gr_k_242 = tmp_init_127*tmp_init_36
        self._gr_k_243 = -9.0*tmp_init_119*tmp_init_36
        self._gr_k_244 = ccomplex.complex[double](tmp_init_128*tmp_init_34, tmp_init_128*(49.0 - tmp_init_4))
        self._gr_k_245 = tmp_init_117 - 55.0
        self._gr_k_246 = 834.0 - tmp_init_124
        self._gr_k_247 = tmp_init_127
        self._gr_k_248 = tmp_init_129
        self._gr_k_249 = tmp_init_128
        self._gr_k_250 = 27000.0*tmp_init_87
        self._gr_k_251 = 25.0*flagTail
        self._gr_k_252 = -240.0*tmp_init_18
        self._gr_k_253 = -224.0*tmp_init_84
        self._gr_k_254 = 6.2003968253968254e-8*flagPN3
        self._gr_k_255 = -tmp_init_132 + 1024.0*ccomplex.complex[double](0, 1)*tmp_init_133 - tmp_init_18*(81271.0 - tmp_init_134)
        self._gr_k_256 = 126.0*nu - 801.0
        self._gr_k_257 = tmp_init_10 + 1329.0
        self._gr_k_258 = 14756.0 - 3856.0*nu
        self._gr_k_259 = 6678.0 - 2304.0*nu
        self._gr_k_260 = tmp_init_0 - 4730.0
        self._gr_k_261 = 19537.0 - 3470.0*nu
        self._gr_k_262 = 735.0 - 570.0*nu
        self._gr_k_263 = 388.0*nu - 1376.0
        self._gr_k_264 = 244.0*nu - 2480.0
        self._gr_k_265 = tmp_init_135
        self._gr_k_266 = 572.0*nu - 6226.0
        self._gr_k_267 = tmp_init_136
        self._gr_k_268 = 22596110.0 - 1090972.0*nu
        self._gr_k_269 = tmp_init_138
        self._gr_k_270 = tmp_init_38 - 12.0
        self._gr_k_271 = 73.0*nu - 164.0
        self._gr_k_272 = tmp_init_139
        self._gr_k_273 = tmp_init_73 - 429.0
        self._gr_k_274 = 410.0*nu - 1246.0
        self._gr_k_275 = 415.0*nu - 965.0
        self._gr_k_276 = 426.0*nu - 1827.0
        self._gr_k_277 = tmp_init_10 - 369.0
        self._gr_k_278 = 230.0*nu - 1309.0
        self._gr_k_279 = tmp_init_140
        self._gr_k_280 = tmp_init_138
        self._gr_k_281 = tmp_init_139
        self._gr_k_282 = 94743000.0*tmp_init_18
        self._gr_k_283 = 90.0*tmp_init_141*(tmp_init_118 - 8192.0*tmp_init_142 + 4374.0*tmp_init_143)
        self._gr_k_284 = tmp_init_54 - 549.0
        self._gr_k_285 = ccomplex.complex[double](160723.88015765382*nu - 1428670.6751464944, -75200.0*nu*tmp_init_1 + 20832.0*nu + 1356640.0*tmp_init_1 - 719760.0)
        self._gr_k_286 = 1547422.8774521886*nu + tmp_init_83*(-61286680.0*tmp_init_1 + tmp_init_15*(10819760.0*tmp_init_1 - 6571935.0*tmp_init_7 - 13496.0) + 35899605.0*tmp_init_7 + 139292.0) - 6101098.5969775221
        self._gr_k_287 = 448.0*nu + 1024.0
        self._gr_k_288 = 210.0*nu - 1599.0
        self._gr_k_289 = 12796.0 - 1424.0*nu
        self._gr_k_290 = tmp_init_144 - 4214.0
        self._gr_k_291 = 11462.0 - 2776.0*nu
        self._gr_k_292 = -tmp_init_145
        self._gr_k_293 = 248.0*nu - 7180.0
        self._gr_k_294 = 33312.0 - 5334.0*nu
        self._gr_k_295 = 38460.0 - 10512.0*nu
        self._gr_k_296 = 501.0*nu - 4590.0
        self._gr_k_297 = -tmp_init_146
        self._gr_k_298 = tmp_init_147 - 1383.0
        self._gr_k_299 = 256.0*nu + 1024.0
        self._gr_k_300 = 10395.0 - 2799.0*nu
        self._gr_k_301 = 1037.0 - 517.0*nu
        self._gr_k_302 = 5343.0 - 2151.0*nu
        self._gr_k_303 = 2926.0 - 1112.0*nu
        self._gr_k_304 = -tmp_init_148
        self._gr_k_305 = tmp_init_149
        self._gr_k_306 = 1012503.8963254545*nu - tmp_init_78*(nu*(883640.0*tmp_init_1 - 820125.0*tmp_init_7 + 74676.0) - 2387200.0*tmp_init_1 + 2340090.0*tmp_init_7 - 362688.0) - 6545508.2937543342
        self._gr_k_307 = tmp_init_116*(2009.0*nu - 18320.0) + tmp_init_6*(nu*(71240.0*tmp_init_1 - tmp_init_150 + 14028.0) - 933680.0*tmp_init_1 + 814860.0*tmp_init_7 - 124920.0)
        self._gr_k_308 = 1981465.3184721544*nu + tmp_init_83*(nu*(1714160.0*tmp_init_1 - 942435.0*tmp_init_7 - 36792.0) - 4740660.0*tmp_init_1 + 2539755.0*tmp_init_7 + 126150.0) - 7249539.2074238069
        self._gr_k_309 = ccomplex.complex[double](tmp_init_146*tmp_init_151, 109337360.0*tmp_init_1 - 32787500.0*tmp_init_22 - tmp_init_67*(7674152.0*tmp_init_1 - 2346875.0*tmp_init_22 - 1640250.0*tmp_init_7 + 81564.0) - 23750820.0*tmp_init_7 + 835656.0)
        self._gr_k_310 = tmp_init_120*(8375.0*nu - 47926.0) + tmp_init_78*(-34292120.0*tmp_init_1 + tmp_init_67*(2455136.0*tmp_init_1 - 1441719.0*tmp_init_7 - 11760.0) + 19408410.0*tmp_init_7 + 350412.0)
        self._gr_k_311 = -tmp_init_121*(nu*(114192.0*tmp_init_1 - tmp_init_150 - 1960.0) - 1444728.0*tmp_init_1 + 860220.0*tmp_init_7 + 14924.0) + tmp_init_19*(4193.0*nu - 32092.0)
        self._gr_k_312 = tmp_init_152*(1285.0*nu - 11966.0) - tmp_init_3*(nu*(1409440.0*tmp_init_1 - 1184625.0*tmp_init_7 + 46032.0) - 3997160.0*tmp_init_1 + 3331530.0*tmp_init_7 - 198444.0)
        self._gr_k_313 = ccomplex.complex[double](tmp_init_151*(4106.0*nu - 13753.0), -nu*(215962160.0*tmp_init_1 + tmp_init_154 - 53093070.0*tmp_init_7 + 344904.0) + 616205560.0*tmp_init_1 - tmp_init_153 - 146368620.0*tmp_init_7 + 1155252.0)
        self._gr_k_314 = ccomplex.complex[double](tmp_init_39*(4825.0*nu - 58724.0), nu*(675440.0*tmp_init_1 - 594135.0*tmp_init_7 + 45192.0) - 9353560.0*tmp_init_1 + 7741980.0*tmp_init_7 - 488292.0)
        self._gr_k_315 = ccomplex.complex[double](tmp_init_145*tmp_init_151, nu*(114660944.0*tmp_init_1 + tmp_init_154 + 14103234.0*tmp_init_7 - 111720.0) - 338113816.0*tmp_init_1 + tmp_init_153 - 38231676.0*tmp_init_7 + 192444.0)
        self._gr_k_316 = 480.0*tmp_init_131 + tmp_init_78*(nu*(tmp_init_155 + 6937.0) - 2560.0*tmp_init_1 - 1792.0)
        self._gr_k_317 = tmp_init_120*(4263.0*nu - 44015.0) + tmp_init_27*(-5057480.0*tmp_init_1 + 3860055.0*tmp_init_7 + tmp_init_9*(53900.0*tmp_init_1 - 40095.0*tmp_init_7 + 98.0) + 19604.0)
        self._gr_k_318 = tmp_init_116*(tmp_init_16 - 209.0) + tmp_init_121*(4.0*nu*tmp_init_133 - 610.0*tmp_init_1 - 207.0)
        self._gr_k_319 = tmp_init_3*(520.0*nu*tmp_init_1 - 3892.0*nu - 1850.0*tmp_init_1 + 7361.0) + tmp_init_76*(556.0*nu - 1463.0)
        self._gr_k_320 = ccomplex.complex[double](tmp_init_148*tmp_init_39, -nu*(30480.0*tmp_init_1 + tmp_init_156 + 12824.0) + 83190.0*tmp_init_1 - 82620.0*tmp_init_7 + 26341.0)
        self._gr_k_321 = 3.0*tmp_init_157*tmp_init_33
        self._gr_k_322 = ccomplex.complex[double](tmp_init_149*tmp_init_39, 30.0*nu*tmp_init_1 - 1043.0*nu - tmp_init_53 + 2164.0)
        self._gr_k_323 = -72.0*tmp_init_158*(6761984.0*tmp_init_1 - 2984375.0*tmp_init_22 + 106434.0*tmp_init_7)
        self._gr_k_324 = ccomplex.complex[double](-9877.1673028863099*tmp_init_141, -24.0*tmp_init_141*(-165888.0*tmp_init_1 + 104247.0*tmp_init_7))
        self._gr_k_325 = ccomplex.complex[double](-114856.62741524284*nu - 242279.62544484485, -48.0*nu*(1013360.0*tmp_init_1 - 623295.0*tmp_init_7 - 1176.0) - 194703360.0*tmp_init_1 + 119672640.0*tmp_init_7 + 129024.0)
        self._gr_k_326 = -15.0*tmp_init_141*(-22282240.0*tmp_init_142 + 6708987.0*tmp_init_143 + 5046875.0*ccomplex.complex[double](0, 1)*tmp_init_22 + 172.78759594743863)
        self._gr_k_327 = tmp_init_141*(-2945024.0*tmp_init_142 + 25.0*ccomplex.complex[double](0, 1)*(19375.0*tmp_init_22 + 45927.0*tmp_init_7) + 116.23892818282235)
        self._gr_k_328 = tmp_init_158*(201052160.0*tmp_init_1 - 102421875.0*tmp_init_22 - 23882747.0*tmp_init_61 + 65498463.0*tmp_init_7)
        self._gr_k_329 = tmp_init_29*(17326.0*nu - 169283.0) - tmp_init_31*(-112866280.0*tmp_init_1 + 8807360.0*tmp_init_20 - tmp_init_37*(882981.0*tmp_init_7 + 3472.0) + 68740083.0*tmp_init_7 + 198660.0)
        self._gr_k_330 = ccomplex.complex[double](6785.8401317539534*tmp_init_124 - 2517546.6888807167, 12.0*nu*(506151936.0*tmp_init_1 - 248468750.0*tmp_init_22 + 45405036.0*tmp_init_7 - 32256.0) - 17901185376.0*tmp_init_1 + 8849362500.0*tmp_init_22 - 1691180856.0*tmp_init_7 + 1121904.0)
        self._gr_k_331 = ccomplex.complex[double](47617.119850460496*nu - 298407.3197938801, nu*(28911136.0*tmp_init_1 - 6453125.0*tmp_init_22 - 8912025.0*tmp_init_7 + 21168.0) - 360934128.0*tmp_init_1 + 73312500.0*tmp_init_22 + 121181670.0*tmp_init_7 - 134568.0)
        self._gr_k_332 = ccomplex.complex[double](tmp_init_116*(27885.0*nu - 158344.0), 11667772400.0*tmp_init_1 - 3034531250.0*tmp_init_22 - tmp_init_67*(808909472.0*tmp_init_1 - 199484375.0*tmp_init_22 - 222576579.0*tmp_init_7 + 116592.0) - 2999052540.0*tmp_init_7 + 3411816.0)
        self._gr_k_333 = -tmp_init_18*(2548475488.0*tmp_init_1 - 1349453125.0*tmp_init_22 - 434007161.0*tmp_init_61 + 1137045357.0*tmp_init_7 + 95760.0) + tmp_init_3*(3856898952.0*tmp_init_1 - 2046484375.0*tmp_init_22 - 626814664.0*tmp_init_61 + 1674354078.0*tmp_init_7 + 61740.0) + tmp_init_77*(tmp_init_159 - 49.0)
        self._gr_k_334 = tmp_init_161*(tmp_init_160 - 1155.0) - tmp_init_6*(nu*(31090.0*tmp_init_1 + tmp_init_156 + 6531.0) - 85450.0*tmp_init_1 + 85050.0*tmp_init_7 - 18315.0)
        self._gr_k_335 = ccomplex.complex[double](1916.3715186897739*nu - 21724.11319957342, -370.0*nu*tmp_init_1 + 7530.0*tmp_init_1 + tmp_init_162 - 8781.0)
        self._gr_k_336 = ccomplex.complex[double](tmp_init_39*(592.0*nu - 2543.0), 2430.0*nu*tmp_init_1 - 3619.0*nu - 8190.0*tmp_init_1 + 5639.0)
        self._gr_k_337 = ccomplex.complex[double](tmp_init_116*(717.0*nu - 1781.0), nu*(187210.0*tmp_init_1 - 91125.0*tmp_init_7 - 15057.0) - 519370.0*tmp_init_1 + 255150.0*tmp_init_7 + 32901.0)
        self._gr_k_338 = -tmp_init_137 - 384.0
        self._gr_k_339 = -18432.0*tmp_init_141*tmp_init_157
        self._gr_k_340 = 8078905.0 - 1355282.0*nu
        self._gr_k_341 = tmp_init_163
        self._gr_k_342 = 297000.0*tmp_init_87
        self._gr_k_343 = ccomplex.complex[double](0, -19008000.0*tmp_init_130)
        self._gr_k_344 = tmp_init_164
        self._gr_k_345 = 80.0*tmp_init_131 + tmp_init_165*(nu*(618.0 - 35.0*tmp_init_1) - 80.0*tmp_init_1 - 56.0)
        self._gr_k_346 = ccomplex.complex[double](-tmp_init_132, 8.0*nu*(tmp_init_166 - 26985.0) + 10240.0*tmp_init_1 + 7168.0)
        self._gr_k_347 = 16.0*tmp_init_130*tmp_init_157
        self._gr_k_348 = tmp_init_167
        self._gr_k_349 = 114928.0*nu + 459712.0
        self._gr_k_350 = 196024.0*nu - 442925.0
        self._gr_k_351 = 1088366400.0*tmp_init_18
        self._gr_k_352 = -981288000.0*tmp_init_18
        self._gr_k_353 = -800.0*tmp_init_85
        self._gr_k_354 = -528.0*tmp_init_84
        self._gr_k_355 = -tmp_init_163
        self._gr_k_356 = tmp_init_168
        self._gr_k_357 = tmp_init_168
        self._gr_k_358 = -275.0*flagTail
        self._gr_k_359 = -tmp_init_164
        self._gr_k_360 = -tmp_init_167
        self._gr_k_361 = -1584.0*tmp_init_84
        self._gr_k_362 = -141193800.0*tmp_init_18
        self._gr_k_363 = 2.6304713804713805e-8*flagPN3
        self._gr_k_364 = 729.0*tmp_init_171*(tmp_init_169 - 159.0)
        self._gr_k_365 = 9756341.6494292465*nu - tmp_init_172*(nu*(776385.0*tmp_init_1 - 776385.0*tmp_init_7 - 872228.0) - 1807920.0*tmp_init_1 + 1807920.0*tmp_init_7 - 1265544.0) - 22718992.761112236
        self._gr_k_366 = 4901073.0351592928*nu - tmp_init_3*(-3010770.0*tmp_init_1 + tmp_init_15*(780030.0*tmp_init_1 - 780030.0*tmp_init_7 - 586537.0) + 3010770.0*tmp_init_7 - 2107539.0) - 9458612.9136485418
        self._gr_k_367 = 1755836.1340913354*nu - tmp_init_83*(nu*(46575.0*tmp_init_1 - 46575.0*tmp_init_7 - 7846.0) - 56700.0*tmp_init_1 + 56700.0*tmp_init_7 - 39690.0) - 2137539.6415024953
        self._gr_k_368 = 1215.0*tmp_init_171*tmp_init_174
        self._gr_k_369 = 1632.0*nu - 3552.0
        self._gr_k_370 = 1284.0*nu - 2478.0
        self._gr_k_371 = 460.0*nu - 560.0
        self._gr_k_372 = tmp_init_175 - 5952.0
        self._gr_k_373 = tmp_init_42 - 477.0
        self._gr_k_374 = tmp_init_176
        self._gr_k_375 = 9667.0 - tmp_init_177
        self._gr_k_376 = tmp_init_178
        self._gr_k_377 = -tmp_init_179
        self._gr_k_378 = -tmp_init_181
        self._gr_k_379 = tmp_init_183
        self._gr_k_380 = 572.0 - 403.0*nu
        self._gr_k_381 = 8934.0 - 4386.0*nu
        self._gr_k_382 = tmp_init_185
        self._gr_k_383 = 1625.0 - 1555.0*nu
        self._gr_k_384 = -1296.0*nu
        self._gr_k_385 = -16160855.264890471*nu - tmp_init_172*(nu*(83864360.0*tmp_init_1 - 66078125.0*tmp_init_22 + 47971845.0*tmp_init_7 - 900228.0) + 183097220.0*tmp_init_1 - 19521875.0*tmp_init_22 - 94729905.0*tmp_init_7 + 1587642.0) + 30333459.371177033
        self._gr_k_386 = -10302413.284476224*nu - tmp_init_172*(nu*(293440.0*tmp_init_1 + 492075.0*tmp_init_7 - 573888.0) + 52127020.0*tmp_init_1 - 35352855.0*tmp_init_7 + 1105062.0) + 30627512.443553038
        self._gr_k_387 = ccomplex.complex[double](-tmp_init_70*(19687.0*nu - 50768.0), 6.0*nu*(48707400.0*tmp_init_1 - 25234375.0*tmp_init_22 + 153090.0*tmp_init_7 + 1653708.0) + 1719406080.0*tmp_init_1 - 105562500.0*tmp_init_22 - 827735760.0*tmp_init_7 - 24135552.0)
        self._gr_k_388 = tmp_init_172*(902570.0*tmp_init_1 - 809190.0*tmp_init_7 + 101829.0) - tmp_init_18*(118280.0*tmp_init_1 - 98415.0*tmp_init_7 + 30828.0) + tmp_init_39*(1769.0*nu - 99244.0)
        self._gr_k_389 = ccomplex.complex[double](3541831.5576571329 - 3959349.2213192164*nu, 5.0*nu*(82926904.0*tmp_init_1 - 40968750.0*tmp_init_22 - 38706521.0*tmp_init_61 + 74911311.0*tmp_init_7 + 352884.0) + 78390320.0*tmp_init_1 - 59437500.0*tmp_init_22 + 226510340.0*tmp_init_61 - 356306040.0*tmp_init_7 - 1481160.0)
        self._gr_k_390 = tmp_init_114*(2732.0*nu - 6337.0) + tmp_init_3*(nu*(1155950240.0*tmp_init_1 - 809453125.0*tmp_init_22 + 477181530.0*tmp_init_7 - 2065392.0) + 3227622200.0*tmp_init_1 - 545593750.0*tmp_init_22 - 1285299900.0*tmp_init_7 + 5082372.0)
        self._gr_k_391 = tmp_init_114*(tmp_init_175 - 7163.0) - tmp_init_3*(nu*(203545760.0*tmp_init_1 - 87578125.0*tmp_init_22 - 13581270.0*tmp_init_7 + 1932336.0) + 1658063960.0*tmp_init_1 - 166281250.0*tmp_init_22 - 760157460.0*tmp_init_7 - 6290028.0)
        self._gr_k_392 = 27498.36049687146*nu - tmp_init_6*(113816.0*tmp_init_1 + 11696.0*tmp_init_20 - 104976.0*tmp_init_7 - tmp_init_81*(tmp_init_23 + 280.0) + 100548.0) + 556464.0235450529
        self._gr_k_393 = ccomplex.complex[double](tmp_init_151*(23842.0*nu - 37217.0), -nu*(3567935920.0*tmp_init_1 - tmp_init_187 - 1886093750.0*tmp_init_22 + 2893623345.0*tmp_init_7 + 2002728.0) - 2296629640.0*tmp_init_1 - tmp_init_186 + 1012437500.0*tmp_init_22 + 2747484360.0*tmp_init_7 + 3126228.0)
        self._gr_k_394 = ccomplex.complex[double](tmp_init_39*(196723.0*nu - 523352.0), nu*(1198960.0*tmp_init_1 + 1228365.0*tmp_init_7 - 1358616.0) + 309481960.0*tmp_init_1 - 201991320.0*tmp_init_7 + 2950620.0)
        self._gr_k_395 = ccomplex.complex[double](1696.4600329384883*tmp_init_180 - 425811.46826756058, nu*(2847492944.0*tmp_init_1 - tmp_init_187 - 442343750.0*tmp_init_22 + 1253000097.0*tmp_init_7 - 370440.0) - 2282969176.0*tmp_init_1 + tmp_init_186 - 253812500.0*tmp_init_22 - 997586928.0*tmp_init_7 + 189756.0)
        self._gr_k_396 = 3944894.0 - 2216588.0*nu
        self._gr_k_397 = 122278.0*nu - 172471.0
        self._gr_k_398 = 139040.0 - 69520.0*nu
        self._gr_k_399 = tmp_init_139*tmp_init_188
        self._gr_k_400 = 4752.0*tmp_init_84
        self._gr_k_401 = 11309.733552923256*tmp_init_144 + tmp_init_31*(nu*(49842376640.0*tmp_init_1 - 9370312500.0*tmp_init_22 - 22929084206.0*tmp_init_61 + 22903326882.0*tmp_init_7 - 413280.0) - 33114448168.0*tmp_init_1 - 6384281250.0*tmp_init_22 + 26708942491.0*tmp_init_61 - 17072778321.0*tmp_init_7 + 497700.0) - 4467344.753404686
        self._gr_k_402 = -tmp_init_121*(nu*(4190.0*tmp_init_1 + 14175.0*tmp_init_7 - 17563.0) + 640890.0*tmp_init_1 - 456840.0*tmp_init_7 + 26467.0) - tmp_init_161*(2509.0*nu - 7441.0)
        self._gr_k_403 = ccomplex.complex[double](-tmp_init_118*(2233.0*nu - 4913.0), 5.0*nu*(454590.0*tmp_init_1 - 296875.0*tmp_init_22 + 38637.0*tmp_init_7 + 46893.0) + 8767530.0*tmp_init_1 + 5.0*tmp_init_189 - 4505220.0*tmp_init_7 - 398685.0)
        self._gr_k_404 = tmp_init_161*(59.0*nu - 35.0) - tmp_init_18*(16910.0*tmp_init_1 - 15795.0*tmp_init_7 + 3717.0) + tmp_init_72*(80150.0*tmp_init_1 - 67068.0*tmp_init_7 + 4869.0)
        self._gr_k_405 = ccomplex.complex[double](239813.47521177687 - 193632.06320400691*nu, -nu*(1646170.0*tmp_init_1 - 1484375.0*tmp_init_22 + 1364445.0*tmp_init_7 - 86289.0) - 2217550.0*tmp_init_1 - 156250.0*tmp_init_22 + 1953720.0*tmp_init_7 - 87969.0)
        self._gr_k_406 = -tmp_init_3*(845930.0*tmp_init_1 - 648810.0*tmp_init_7 + tmp_init_9*(650.0*tmp_init_1 + 12150.0*tmp_init_7 - 15841.0) + 62855.0) - tmp_init_76*(9052.0*nu - 25025.0)
        self._gr_k_407 = ccomplex.complex[double](tmp_init_116*(512.0*nu + 827.0), -nu*(33920.0*tmp_init_1 - tmp_init_71 + 10752.0) + 378070.0*tmp_init_1 - 328050.0*tmp_init_7 + 24213.0)
        self._gr_k_408 = ccomplex.complex[double](-tmp_init_118*(932.0*nu - 1483.0), 5.0*nu*(68920.0*tmp_init_1 + tmp_init_21 - 59375.0*tmp_init_22 + 19572.0) + 566030.0*tmp_init_1 + 200000.0*tmp_init_22 - 371790.0*tmp_init_7 - 94695.0)
        self._gr_k_409 = 373248.0*nu*tmp_init_171
        self._gr_k_410 = tmp_init_54 - 72.0
        self._gr_k_411 = tmp_init_190 - 576.0
        self._gr_k_412 = tmp_init_34*(7619.0*nu - 404371.0) + tmp_init_78*(636520.0*tmp_init_1 - 433755.0*tmp_init_7 + tmp_init_9*(20356.0*tmp_init_1 - 16281.0*tmp_init_7 + 630.0) - 64260.0)
        self._gr_k_413 = ccomplex.complex[double](14205025.342471609*nu - 37299501.257540897, 115051636800.0*tmp_init_1 + 120.0*tmp_init_15*(142881744.0*tmp_init_1 - 90390625.0*tmp_init_22 + 43277814.0*tmp_init_7 - 26376.0) - 24367875000.0*tmp_init_22 - 37537842960.0*tmp_init_7 + 21287520.0)
        self._gr_k_414 = tmp_init_63*(70274.0*nu - 157873.0) + tmp_init_79*(22440280.0*tmp_init_1 + tmp_init_37*(3712.0*tmp_init_1 + tmp_init_191 - 5824.0) - 14216229.0*tmp_init_7 + 93156.0)
        self._gr_k_415 = ccomplex.complex[double](tmp_init_39*(645883.0*nu - 1776074.0), -nu*(2487389920.0*tmp_init_1 - 927734375.0*tmp_init_22 - 300460995.0*tmp_init_7 + 4502736.0) - 28214938480.0*tmp_init_1 + 3610625000.0*tmp_init_22 + 12216122730.0*tmp_init_7 + 21299544.0)
        self._gr_k_416 = ccomplex.complex[double](tmp_init_70*(8942.0*nu - 18253.0), -nu*(107350125920.0*tmp_init_1 - 58522484375.0*tmp_init_22 - 33868205875.0*tmp_init_61 + 77810584095.0*tmp_init_7 + 4506768.0) - 101524887920.0*tmp_init_1 + 39879875000.0*tmp_init_22 - 38527526450.0*tmp_init_61 + 73533522870.0*tmp_init_7 + 9899352.0)
        self._gr_k_417 = 39584.067435231395*tmp_init_174 - tmp_init_18*(123851303456.0*tmp_init_1 - 6360546875.0*tmp_init_22 - 41996575285.0*tmp_init_61 + 5560700463.0*tmp_init_7 + 246960.0) + tmp_init_3*(73198537016.0*tmp_init_1 + 2481250000.0*tmp_init_22 - 25132263415.0*tmp_init_61 - 5302961055.0*tmp_init_7 + 8820.0)
        self._gr_k_418 = -495.0*flagTail
        self._gr_k_419 = -tmp_init_192
        self._gr_k_420 = 35874.0 - 13158.0*nu
        self._gr_k_421 = tmp_init_184 + 481.0
        self._gr_k_422 = 10069.0 - 7775.0*nu
        self._gr_k_423 = 3888.0*nu
        self._gr_k_424 = -tmp_init_165*(-89910.0*tmp_init_1 + tmp_init_169*(4131.0*tmp_init_1 - 4131.0*tmp_init_7 - 5198.0) + 89910.0*tmp_init_7 - 62937.0) + tmp_init_193*(17.0*nu - 37.0)
        self._gr_k_425 = tmp_init_183*tmp_init_194 - tmp_init_78*(nu*(17145.0*tmp_init_1 - 17145.0*tmp_init_7 - 190241.0) + 55620.0*tmp_init_1 - 55620.0*tmp_init_7 + 38934.0)
        self._gr_k_426 = ccomplex.complex[double](727781.3541306115 - 16540.485321150261*tmp_init_44, 2.0*nu*(163215.0*tmp_init_1 - 163215.0*tmp_init_7 + 316493.0) - 463320.0*tmp_init_1 + 463320.0*tmp_init_7 - 324324.0)
        self._gr_k_427 = -405.0*tmp_init_171*(98.0*nu - 61.0)
        self._gr_k_428 = ccomplex.complex[double](12299759.353812275 - 8906.4151729270638*tmp_init_111, -7830270.0*tmp_init_1 + 3.0*tmp_init_15*(580230.0*tmp_init_1 - 580230.0*tmp_init_7 - 402809.0) + 7830270.0*tmp_init_7 - 5481189.0)
        self._gr_k_429 = 319221.0*tmp_init_170 - tmp_init_179*tmp_init_194 + tmp_init_195*(971190.0*tmp_init_1 - 971190.0*tmp_init_7 - 129137.0)
        self._gr_k_430 = 126927.0*tmp_init_170 + tmp_init_178*tmp_init_194 - tmp_init_196*(4698.0*tmp_init_1 - 4698.0*tmp_init_7 - 61429.0)
        self._gr_k_431 = -5580505.2783511574*nu + 723654.0*tmp_init_170 + tmp_init_197*(888165.0*tmp_init_1 - 888165.0*tmp_init_7 + 176782.0) + 11367130.450704341
        self._gr_k_432 = 213921.0*tmp_init_170 + tmp_init_185*tmp_init_194 - tmp_init_195*(112185.0*tmp_init_1 - 112185.0*tmp_init_7 - 1337168.0)
        self._gr_k_433 = ccomplex.complex[double](2067560.6651437827 - 6361.7251235193313*tmp_init_160, 3.0*nu*(419850.0*tmp_init_1 - 419850.0*tmp_init_7 + 374792.0) - 1316250.0*tmp_init_1 + 1316250.0*tmp_init_7 - 921375.0)
        self._gr_k_434 = -144.0*nu*(tmp_init_3*(-3645.0*tmp_init_1 + 3645.0*tmp_init_7 + 37897.0) + 11451.105222334796)
        self._gr_k_435 = tmp_init_182 + 250.0
        self._gr_k_436 = 1522.0 - 806.0*nu
        self._gr_k_437 = tmp_init_42 - 765.0
        self._gr_k_438 = 10458.0 - 3852.0*nu
        self._gr_k_439 = 8184.0 - tmp_init_175
        self._gr_k_440 = 3880.0 - 2300.0*nu
        self._gr_k_441 = 35.0 - tmp_init_180
        self._gr_k_442 = 4796.0*nu - 10582.0
        self._gr_k_443 = tmp_init_177 - 13285.0
        self._gr_k_444 = 1470.0*nu - 1077.0
        self._gr_k_445 = -178200.0*tmp_init_87
        self._gr_k_446 = 1.6237477657230744e-9*flagPN3
        self._gr_k_447 = 4.3402777777777778e-5*tmp_init_199
        self._gr_k_448 = 27.0*tmp_init_52
        self._gr_k_449 = 108.0*tmp_init_18
        self._gr_k_450 = -9720.0*tmp_init_56
        self._gr_k_451 = ccomplex.complex[double](0, -116640.0*tmp_init_51)
        self._gr_k_452 = ccomplex.complex[double](-tmp_init_193*tmp_init_51, 16.0*nu*(9720.0*tmp_init_1 - 9720.0*tmp_init_7 + 16301.0) - 77760.0*tmp_init_1 + 77760.0*tmp_init_7 - 82944.0)
        self._gr_k_453 = 1.7861225422953818e-7*tmp_init_199
        self._gr_k_454 = 19890.0*nu - 9945.0
        self._gr_k_455 = 218556.0*nu - 109278.0
        self._gr_k_456 = 147629156.0*nu - 128967272.0*tmp_init_57 - 41696632.0
        self._gr_k_457 = 20709.0 - 41418.0*nu
        self._gr_k_458 = 141804.0 - 283608.0*nu
        self._gr_k_459 = 1170.0 - 2340.0*nu
        self._gr_k_460 = tmp_init_172*(-40.0*nu*(7.0*tmp_init_1 - 97.0) + tmp_init_201) + 560.0*tmp_init_200
        self._gr_k_461 = 3360.0*tmp_init_200 + 48.0*tmp_init_202 - tmp_init_203*(2240.0*tmp_init_1 + 54603.0)
        self._gr_k_462 = ccomplex.complex[double](-tmp_init_204, 8960.0*nu*tmp_init_1 - 214489.0*nu - tmp_init_134 - 5792.0)
        self._gr_k_463 = ccomplex.complex[double](-tmp_init_204, 8.0*nu*(-tmp_init_155 - 21439.0) - 8.0*tmp_init_166 - 5792.0)
        self._gr_k_464 = 8.0*tmp_init_205*tmp_init_51
        self._gr_k_465 = tmp_init_197
        self._gr_k_466 = tmp_init_195
        self._gr_k_467 = 14595.0*tmp_init_18
        self._gr_k_468 = 118420.0*tmp_init_18
        self._gr_k_469 = -tmp_init_18
        self._gr_k_470 = tmp_init_207
        self._gr_k_471 = tmp_init_207
        self._gr_k_472 = ccomplex.complex[double](0, -tmp_init_67)
        self._gr_k_473 = ccomplex.complex[double](0, -tmp_init_108)
        self._gr_k_474 = -tmp_init_195
        self._gr_k_475 = -152820.0*tmp_init_18
        self._gr_k_476 = -13965.0*tmp_init_18
        self._gr_k_477 = 1872.0*nu - 936.0
        self._gr_k_478 = 178308.0*nu - 89154.0
        self._gr_k_479 = 194454.0 - 388908.0*nu
        self._gr_k_480 = 28748516.0*nu - 29089844.0*tmp_init_57 - 7094293.0
        self._gr_k_481 = 21177.0 - 42354.0*nu
        self._gr_k_482 = 1557504.0*nu - 778752.0
        self._gr_k_483 = -1555057.0*nu + 1800442.0*tmp_init_57 + 325010.0
        self._gr_k_484 = 1248.0*nu - 624.0
        self._gr_k_485 = 58734.0*nu - 29367.0
        self._gr_k_486 = tmp_init_208
        self._gr_k_487 = 1163008.0*nu - 13216.0*tmp_init_57 - 591416.0
        self._gr_k_488 = 156.0 - 312.0*nu
        self._gr_k_489 = 6289920.0*tmp_init_210
        self._gr_k_490 = tmp_init_113*tmp_init_188
        self._gr_k_491 = 13.0*tmp_init_211
        self._gr_k_492 = -98280.0*tmp_init_212
        self._gr_k_493 = -393120.0*tmp_init_210
        self._gr_k_494 = -1872.0*tmp_init_205*tmp_init_209
        self._gr_k_495 = 4.7695360195360195e-7*tmp_init_213
        self._gr_k_496 = 1188720.0*nu - 594360.0
        self._gr_k_497 = 340470.0*nu - 170235.0
        self._gr_k_498 = 3112200.0*nu - 1556100.0
        self._gr_k_499 = 2354040.0*nu - 1177020.0
        self._gr_k_500 = 9629636.0*nu - 4510184.0*tmp_init_57 - 3663472.0
        self._gr_k_501 = 42120.0*nu - 21060.0
        self._gr_k_502 = 8125.0*tmp_init_211
        self._gr_k_503 = 12480.0*nu - 6240.0
        self._gr_k_504 = 61425000.0*tmp_init_212
        self._gr_k_505 = -1228500000.0*tmp_init_210
        self._gr_k_506 = 9984.0*nu - 4992.0
        self._gr_k_507 = 9360.0*nu - 4680.0
        self._gr_k_508 = 6469632.0*nu - 3234816.0
        self._gr_k_509 = -963200.0*nu + 358400.0*tmp_init_57 + 368200.0
        self._gr_k_510 = 273000.0*nu - 136500.0
        self._gr_k_511 = tmp_init_208
        self._gr_k_512 = 10000.0*tmp_init_85
        self._gr_k_513 = -585000.0*tmp_init_18*tmp_init_51
        self._gr_k_514 = -1872.0*tmp_init_51*(-28.0*tmp_init_18*(31250.0*tmp_init_1 - tmp_init_189 + 52917.0) + 218750.0*tmp_init_200 + 3125.0*ccomplex.complex[double](0, 1)*(tmp_init_201 - 140.0*tmp_init_22))
        self._gr_k_515 = 1.221001221001221e-12*tmp_init_213
        self._gr_k_516 = ccomplex.complex[double](-tmp_init_214, nu*(6531840.0*tmp_init_1 - 6531840.0*tmp_init_7 + 4655359.0) - 3265920.0*tmp_init_1 + 3265920.0*tmp_init_7 - 4222368.0)
        self._gr_k_517 = tmp_init_18
        self._gr_k_518 = tmp_init_18
        self._gr_k_519 = 9835.0*tmp_init_18
        self._gr_k_520 = -320.0*tmp_init_18*(5103.0*tmp_init_1 - tmp_init_191 + 3427.0) + 408240.0*tmp_init_200 + 5832.0*tmp_init_215
        self._gr_k_521 = 2449440.0*tmp_init_200 - tmp_init_203*(1632960.0*tmp_init_1 - 1632960.0*tmp_init_7 + 770467.0) + 34992.0*tmp_init_215
        self._gr_k_522 = 8.0*ccomplex.complex[double](0, 1)*nu*(816480.0*tmp_init_1 - 816480.0*tmp_init_7 + 288649.0) - tmp_init_214 - 23328.0*tmp_init_215
        self._gr_k_523 = 5832.0*tmp_init_216*tmp_init_51
        self._gr_k_524 = tmp_init_206
        self._gr_k_525 = 7035.0*tmp_init_18
        self._gr_k_526 = 55140.0*tmp_init_18
        self._gr_k_527 = 40060.0*tmp_init_18
        self._gr_k_528 = 40.0*tmp_init_18
        self._gr_k_529 = tmp_init_197
        self._gr_k_530 = tmp_init_55
        self._gr_k_531 = tmp_init_195
        self._gr_k_532 = tmp_init_196
        self._gr_k_533 = 1198080.0*nu - 599040.0
        self._gr_k_534 = 1173744.0*nu - 586872.0
        self._gr_k_535 = 951912.0*nu - 475956.0
        self._gr_k_536 = 356616.0*nu - 178308.0
        self._gr_k_537 = 266760.0*nu - 133380.0
        self._gr_k_538 = 213730660.0*nu - 173831336.0*tmp_init_57 - 63367008.0
        self._gr_k_539 = 52650.0*nu - 26325.0
        self._gr_k_540 = 7488.0*nu - 3744.0
        self._gr_k_541 = 243243.0 - 486486.0*nu
        self._gr_k_542 = 149526.0 - 299052.0*nu
        self._gr_k_543 = -16835824.0*nu + 14723660.0*tmp_init_57 + 4739601.0
        self._gr_k_544 = 70902.0 - 141804.0*nu
        self._gr_k_545 = 56160.0 - 112320.0*nu
        self._gr_k_546 = 44928.0 - 89856.0*nu
        self._gr_k_547 = 2340.0 - 4680.0*nu
        self._gr_k_548 = 909440.0*nu - 172480.0*tmp_init_57 - 405720.0
        self._gr_k_549 = 5616.0*nu - 2808.0
        self._gr_k_550 = 13756055040.0*tmp_init_210
        self._gr_k_551 = 117.0*tmp_init_211
        self._gr_k_552 = -3538080.0*tmp_init_210
        self._gr_k_553 = -294840.0*tmp_init_212
        self._gr_k_554 = -16848.0*tmp_init_209*tmp_init_216
        self._gr_k_555 = -16.0*tmp_init_85
        self._gr_k_556 = 6.5425734150013985e-10*tmp_init_213
        self._gr_k_557 = 1.6744898834019204e-7*flagPN3
        self._gr_k_558 = 3.7252902984619141e-9*flagPN3
        self._gr_k_559 = 1.9073486328125e-6*flagPN3
        self._gr_k_560 = tmp_init_217
        self._gr_k_561 = 1.8605443148910227e-8*flagPN3
        self._gr_k_562 = 252.0*nu + 5544.0*tmp_init_57 - 22113.0
        self._gr_k_563 = -802740.0*nu + 201864.0*tmp_init_57 + 699.0
        self._gr_k_564 = -1923684.0*nu + 82152.0*tmp_init_57 + 4068519.0
        self._gr_k_565 = -8681608.0*nu + 989776.0*tmp_init_57 + 16444598.0
        self._gr_k_566 = -3410792.0*nu + 513296.0*tmp_init_57 + 3326638.0
        self._gr_k_567 = -35700.0*nu + 34440.0*tmp_init_57 + 13755.0
        self._gr_k_568 = -921176.0*nu + 47312.0*tmp_init_57 - 533894.0
        self._gr_k_569 = 91148.0*nu + tmp_init_218 - 701749.0
        self._gr_k_570 = -507828.0*nu + 85752.0*tmp_init_57 - 145221.0
        self._gr_k_571 = -46620.0*nu + 5544.0*tmp_init_57 - 8253.0
        self._gr_k_572 = -1982544.0*nu + 352512.0*tmp_init_57 + 607344.0
        self._gr_k_573 = -1866576.0*nu + 28032.0*tmp_init_57 + 2803824.0
        self._gr_k_574 = -4812008.0*nu + 578288.0*tmp_init_57 + 5164042.0
        self._gr_k_575 = -196700.0*nu + 80360.0*tmp_init_57 - 9485.0
        self._gr_k_576 = -980.0*nu + 224.0*tmp_init_57 + 380569.0
        self._gr_k_577 = 1354752.0*nu - 3386880.0
        self._gr_k_578 = -2056960.0*nu + 283648.0*tmp_init_57 + 4507136.0
        self._gr_k_579 = 99212.0*nu + tmp_init_218 - 719137.0
        self._gr_k_580 = -316548.0*nu + 22104.0*tmp_init_57 - 286245.0
        self._gr_k_581 = 1724804.0*nu - 63104.0*tmp_init_57 - 457201.0
        self._gr_k_582 = 2678212.0*nu - 355072.0*tmp_init_57 - 671813.0
        self._gr_k_583 = 522220.0*nu - 133408.0*tmp_init_57 + 59653.0
        self._gr_k_584 = 846720.0 - 338688.0*nu
        self._gr_k_585 = 1.4762849584278156e-6*flagPN2
        self._gr_k_586 = tmp_init_219
        self._gr_k_587 = tmp_init_46
        self._gr_k_588 = tmp_init_124*tmp_init_188
        self._gr_k_589 = tmp_init_219
        self._gr_k_590 = 38880.0*tmp_init_87
        self._gr_k_591 = -tmp_init_126*tmp_init_188
        self._gr_k_592 = -27.0*flagTail
        self._gr_k_593 = 5.3583676268861454e-7*flagPN2
        self._gr_k_594 = 0.0001220703125*flagPN2
        self._gr_k_595 = 4.0187757201646091e-6*flagPN2
        self._gr_k_596 = 0.0009765625*flagPN2
        self._gr_k_597 = 1.52587890625e-5*flagPN2
        self._gr_k_598 = tmp_init_220
        self._gr_k_599 = 0.027777777777777778*flagPN12
        self._gr_k_600 = tmp_init_220
        self._gr_k_601 = 0.03125*flagPN1
        self._gr_k_602 = 0.001953125*flagPN1
        self._gr_k_603 = 0.0625*flagPN1
        self._gr_k_604 = -163888.0*nu + 12696.0*tmp_init_57 + 76744.0
        self._gr_k_605 = -43215.0*nu + 6300.0*tmp_init_57 + 19590.0
        self._gr_k_606 = 306864.0*nu - 46008.0*tmp_init_57 - 135720.0
        self._gr_k_607 = 493862.0*nu - 42096.0*tmp_init_57 - 226244.0
        self._gr_k_608 = 44745.0*nu - 17100.0*tmp_init_57 - 17250.0
        self._gr_k_609 = 3168.0 - 6336.0*nu
        self._gr_k_610 = -20213.0*nu + 2100.0*tmp_init_57 + 9434.0
        self._gr_k_611 = 130087.0*nu - 16476.0*tmp_init_57 - 58798.0
        self._gr_k_612 = 42502.0*nu - 11400.0*tmp_init_57 - 17836.0
        self._gr_k_613 = -24059.0*nu + 876.0*tmp_init_57 + 12062.0
        self._gr_k_614 = 44282.0*nu - 5376.0*tmp_init_57 - 22604.0
        self._gr_k_615 = 161193.0*nu - 33948.0*tmp_init_57 - 71442.0
        self._gr_k_616 = tmp_init_37 - 3.0
        self._gr_k_617 = 133.0*nu - 24.0*tmp_init_57 - 67.0
        self._gr_k_618 = 0.00011837121212121212*tmp_init_221
        self._gr_k_619 = 9504.0*nu - 4752.0
        self._gr_k_620 = -232726.0*nu + 38568.0*tmp_init_57 + 97116.0
        self._gr_k_621 = -261229.0*nu + 31524.0*tmp_init_57 + 114282.0
        self._gr_k_622 = -40821.0*nu + 13860.0*tmp_init_57 + 14058.0
        self._gr_k_623 = -157264.0*nu + 18728.0*tmp_init_57 + 69368.0
        self._gr_k_624 = -2986632.0*nu + 430848.0*tmp_init_57 + 1275120.0
        self._gr_k_625 = -212586.0*nu + 27984.0*tmp_init_57 + 91740.0
        self._gr_k_626 = 12065.0*nu + 260.0*tmp_init_57 - 5890.0
        self._gr_k_627 = -94000.0*nu + 18360.0*tmp_init_57 + 37800.0
        self._gr_k_628 = -8855.0*nu + 4620.0*tmp_init_57 + 2310.0
        self._gr_k_629 = -1129348.0*nu + 298320.0*tmp_init_57 + 427944.0
        self._gr_k_630 = -41236.0*nu - 15504.0*tmp_init_57 + 19560.0
        self._gr_k_631 = 1051.0*nu + 144.0*tmp_init_57 - 513.0
        self._gr_k_632 = 9394.0*nu - 2112.0*tmp_init_57 - 3729.0
        self._gr_k_633 = 4.3841189674523008e-6*tmp_init_221
        self._gr_k_634 = -15513600.0*nu + 4147200.0*tmp_init_57 + 6274560.0
        self._gr_k_635 = 9510020.0*nu + 581100.0*tmp_init_57 - 4532255.0
        self._gr_k_636 = 19543000.0*nu + 858900.0*tmp_init_57 - 9257455.0
        self._gr_k_637 = 264420.0*nu + 16380.0*tmp_init_57 - 125931.0
        self._gr_k_638 = 20679540.0*nu + 447780.0*tmp_init_57 - 9737721.0
        self._gr_k_639 = 28004580.0*nu - 12205620.0*tmp_init_57 - 9457911.0
        self._gr_k_640 = 137598900.0*nu - 35023860.0*tmp_init_57 - 55012863.0
        self._gr_k_641 = 20175380.0*nu + 626220.0*tmp_init_57 - 9522839.0
        self._gr_k_642 = 93510780.0*nu - 31442940.0*tmp_init_57 - 34995717.0
        self._gr_k_643 = 17115580.0*nu - 8463420.0*tmp_init_57 - 5532901.0
        self._gr_k_644 = 208878280.0*nu - 49944960.0*tmp_init_57 - 84641908.0
        self._gr_k_645 = 76900760.0*nu - 28976640.0*tmp_init_57 - 27657692.0
        self._gr_k_646 = 200278920.0*nu - 57817800.0*tmp_init_57 - 77899350.0
        self._gr_k_647 = 9488000.0*nu - 5575500.0*tmp_init_57 - 2751275.0
        self._gr_k_648 = 175510.0*nu + 2560.0*tmp_init_57 - 82507.0
        self._gr_k_649 = 428930.0*nu - 174960.0*tmp_init_57 - 151933.0
        self._gr_k_650 = 2425340.0*nu - 887100.0*tmp_init_57 - 871925.0
        self._gr_k_651 = 184521880.0*nu - 49047480.0*tmp_init_57 - 73143274.0
        self._gr_k_652 = 103997320.0*nu - 33361320.0*tmp_init_57 - 39209566.0
        self._gr_k_653 = 171363980.0*nu - 40991580.0*tmp_init_57 - 69696629.0
        self._gr_k_654 = 2455020.0*nu - 1871100.0*tmp_init_57 - 558765.0
        self._gr_k_655 = 5902448.0*nu - 1726320.0*tmp_init_57 - 2277140.0
        self._gr_k_656 = 10062120.0*nu - 2507400.0*tmp_init_57 - 4088790.0
        self._gr_k_657 = 8567148.0*nu - 2209500.0*tmp_init_57 - 3422469.0
        self._gr_k_658 = 492520.0*nu - 262920.0*tmp_init_57 - 148246.0
        self._gr_k_659 = 28980.0*nu - 34020.0*tmp_init_57 - 2331.0
        self._gr_k_660 = 89237480.0*nu - 26505000.0*tmp_init_57 - 34523390.0
        self._gr_k_661 = 29876020.0*nu - 13141980.0*tmp_init_57 - 10240969.0
        self._gr_k_662 = 6.4102564102564103e-8*tmp_init_221
        self._gr_k_663 = -1176936.0*nu + 459864.0*tmp_init_57 + 496878.0
        self._gr_k_664 = -787336.0*nu + 102144.0*tmp_init_57 + 374596.0
        self._gr_k_665 = -3912536.0*nu + 900864.0*tmp_init_57 + 1707980.0
        self._gr_k_666 = -491176.0*nu + 94212.0*tmp_init_57 + 222283.0
        self._gr_k_667 = -978512.0*nu + 357924.0*tmp_init_57 + 393335.0
        self._gr_k_668 = -10352640.0*nu + 276480.0*tmp_init_57 + 4984320.0
        self._gr_k_669 = tmp_init_28 - 13.0
        self._gr_k_670 = 5358.0*nu - 944.0*tmp_init_57 - 2451.0
        self._gr_k_671 = -390084.0*nu + 89100.0*tmp_init_57 + 172167.0
        self._gr_k_672 = -2484108.0*nu + 281124.0*tmp_init_57 + 1162269.0
        self._gr_k_673 = -9140744.0*nu + 1153368.0*tmp_init_57 + 4200398.0
        self._gr_k_674 = -2909336.0*nu + 519432.0*tmp_init_57 + 1353050.0
        self._gr_k_675 = -3445156.0*nu + 830700.0*tmp_init_57 + 1483807.0
        self._gr_k_676 = -488044.0*nu + 212100.0*tmp_init_57 + 186517.0
        self._gr_k_677 = -3186828.0*nu + 231588.0*tmp_init_57 + 1514541.0
        self._gr_k_678 = -539368.0*nu + 157656.0*tmp_init_57 + 224230.0
        self._gr_k_679 = -60420.0*nu + 17820.0*tmp_init_57 + 25635.0
        self._gr_k_680 = -7950192.0*nu + 460368.0*tmp_init_57 + 3781236.0
        self._gr_k_681 = -650024.0*nu + 87768.0*tmp_init_57 + 299942.0
        self._gr_k_682 = -2388316.0*nu + 333924.0*tmp_init_57 + 1085437.0
        self._gr_k_683 = -53140.0*nu + 30300.0*tmp_init_57 + 18355.0
        self._gr_k_684 = -372348.0*nu + 235620.0*tmp_init_57 + 139341.0
        self._gr_k_685 = 357516.0*nu - 50868.0*tmp_init_57 - 166521.0
        self._gr_k_686 = -127508.0*nu + 28044.0*tmp_init_57 + 58319.0
        self._gr_k_687 = -688636.0*nu + 245700.0*tmp_init_57 + 281437.0
        self._gr_k_688 = -60892.0*nu + 3276.0*tmp_init_57 + 29731.0
        self._gr_k_689 = -498872.0*nu + 81576.0*tmp_init_57 + 229706.0
        self._gr_k_690 = 40164.0*nu - 81876.0*tmp_init_57 - 4269.0
        self._gr_k_691 = 0.00020032051282051282*tmp_init_221
        self._gr_k_692 = 1376.0 - tmp_init_144
        self._gr_k_693 = 2480.0 - 184.0*nu
        self._gr_k_694 = 6226.0 - 164.0*nu
        self._gr_k_695 = -tmp_init_135
        self._gr_k_696 = -tmp_init_136
        self._gr_k_697 = 192.0*nu + 2784.0
        self._gr_k_698 = 41.0 - tmp_init_49
        self._gr_k_699 = 524.0*nu - 1246.0
        self._gr_k_700 = tmp_init_13 - 429.0
        self._gr_k_701 = 430.0*nu - 965.0
        self._gr_k_702 = 366.0*nu - 1827.0
        self._gr_k_703 = 314.0*nu - 1309.0
        self._gr_k_704 = tmp_init_140
        self._gr_k_705 = 0.010416666666666667*flagPN32
        self._gr_k_706 = -84474.0*nu + 7668.0*tmp_init_57 + 7605.0
        self._gr_k_707 = -335100.0*nu + 4056.0*tmp_init_57 + 978702.0
        self._gr_k_708 = -151304.0*nu + 363728.0*tmp_init_57 + 14169636.0
        self._gr_k_709 = -662662.0*nu - 10868.0*tmp_init_57 + 7768827.0
        self._gr_k_710 = -1518022.0*nu - 74228.0*tmp_init_57 + 3927627.0
        self._gr_k_711 = -803964.0*nu - 8616.0*tmp_init_57 + 636558.0
        self._gr_k_712 = -111930.0*nu + 20340.0*tmp_init_57 + 20805.0
        self._gr_k_713 = 726.0*nu - 198.0
        self._gr_k_714 = -305404.0*nu + 12166.0*tmp_init_57 + 417582.0
        self._gr_k_715 = -24748.0*nu + 3018.0*tmp_init_57 + 33346.0
        self._gr_k_716 = 18314.0*nu + 21118.0*tmp_init_57 - 12867.0
        self._gr_k_717 = -174142.0*nu + 26618.0*tmp_init_57 + 137921.0
        self._gr_k_718 = 9300.0*nu + 35452.0*tmp_init_57 - 57028.0
        self._gr_k_719 = 12188.0*nu + 3366.0*tmp_init_57 - 19283.0
        self._gr_k_720 = -140828.0*nu + 34286.0*tmp_init_57 + 102021.0
        self._gr_k_721 = 376.0*nu + 5900.0*tmp_init_57 - 8883.0
        self._gr_k_722 = -162966.0*nu + tmp_init_59 + 106131.0
        self._gr_k_723 = -395994.0*nu + 6612.0*tmp_init_57 + 1690341.0
        self._gr_k_724 = -1711452.0*nu - 344488.0*tmp_init_57 + 6611854.0
        self._gr_k_725 = -2018260.0*nu - 78456.0*tmp_init_57 + 6438922.0
        self._gr_k_726 = -1622382.0*nu - 4868.0*tmp_init_57 + 1981967.0
        self._gr_k_727 = -330866.0*nu + 47460.0*tmp_init_57 + 149921.0
        self._gr_k_728 = -673464.0*nu + 5456.0*tmp_init_57 + 2725228.0
        self._gr_k_729 = -3775140.0*nu - 394824.0*tmp_init_57 + 8468346.0
        self._gr_k_730 = -4508216.0*nu + 45456.0*tmp_init_57 + 6612908.0
        self._gr_k_731 = -431350.0*nu + 35076.0*tmp_init_57 + 488839.0
        self._gr_k_732 = -1422902.0*nu + 191076.0*tmp_init_57 + 949511.0
        self._gr_k_733 = 1086720.0*nu + 416256.0*tmp_init_57 + 4060032.0
        self._gr_k_734 = 380160.0 - 152064.0*nu
        self._gr_k_735 = 95040.0 - tmp_init_89
        self._gr_k_736 = -1.9728535353535354e-5*flagPN52
        self._gr_k_737 = 1344.0*nu - 3552.0
        self._gr_k_738 = 424.0*nu - 560.0
        self._gr_k_739 = 1116.0*nu - 2478.0
        self._gr_k_740 = 2136.0*nu - 5952.0
        self._gr_k_741 = tmp_init_176
        self._gr_k_742 = tmp_init_15 - 103.0
        self._gr_k_743 = tmp_init_125 - 143.0
        self._gr_k_744 = 4188.0*nu - 8934.0
        self._gr_k_745 = 3758.0*nu - 9667.0
        self._gr_k_746 = 2218.0*nu - 3941.0
        self._gr_k_747 = tmp_init_173 - 1567.0
        self._gr_k_748 = tmp_init_181
        self._gr_k_749 = tmp_init_159 - 2641.0
        self._gr_k_750 = 1582.0*nu - 1625.0
        self._gr_k_751 = 0.00038580246913580247*flagPN32
        self._gr_k_752 = -3337425.0*nu + 3337425.0*tmp_init_57 + 667485.0
        self._gr_k_753 = 97172166.0*nu + 22702680.0*tmp_init_101 - 126570990.0*tmp_init_57 - 17563182.0
        self._gr_k_754 = 201514264.0*nu + 32303040.0*tmp_init_101 - 245115500.0*tmp_init_57 - 37488548.0
        self._gr_k_755 = 134248282.0*nu + 17860920.0*tmp_init_101 - 158636450.0*tmp_init_57 - 25269554.0
        self._gr_k_756 = 11637465.0*nu + 5821200.0*tmp_init_101 - 18719925.0*tmp_init_57 - 1886745.0
        self._gr_k_757 = -704403.0*nu + 278124.0*tmp_init_101 + 674072.0*tmp_init_57 + 135534.0
        self._gr_k_758 = 105323526.0*nu + 15107400.0*tmp_init_101 - 125972280.0*tmp_init_57 - 19726452.0
        self._gr_k_759 = 54001185.0*nu + 7535220.0*tmp_init_101 - 63800940.0*tmp_init_57 - 10175550.0
        self._gr_k_760 = 25555215.0*nu + 7276500.0*tmp_init_101 - 35000700.0*tmp_init_57 - 4509330.0
        self._gr_k_761 = -2924565.0*nu + 161700.0*tmp_init_101 + 2925300.0*tmp_init_57 + 580230.0
        self._gr_k_762 = 81785690.0*nu + 14737800.0*tmp_init_101 - 101747800.0*tmp_init_57 - 15067180.0
        self._gr_k_763 = 1713285.0*nu + 1455300.0*tmp_init_101 - 3483900.0*tmp_init_57 - 232470.0
        self._gr_k_764 = -66108315.0*nu - 9333324.0*tmp_init_101 + 78966888.0*tmp_init_57 + 12386166.0
        self._gr_k_765 = -62832077.0*nu - 12644940.0*tmp_init_101 + 79249240.0*tmp_init_57 + 11520514.0
        self._gr_k_766 = -12656805.0*nu - 4689300.0*tmp_init_101 + 18362120.0*tmp_init_57 + 2176314.0
        self._gr_k_767 = -84219464.0*nu - 14562240.0*tmp_init_101 + 103527900.0*tmp_init_57 + 15604948.0
        self._gr_k_768 = -5424902.0*nu + 826980.0*tmp_init_101 + 5383525.0*tmp_init_57 + 1064899.0
        self._gr_k_769 = -34728946.0*nu - 10575180.0*tmp_init_101 + 47595415.0*tmp_init_57 + 6145097.0
        self._gr_k_770 = -9566837.0*nu + 1122660.0*tmp_init_101 + 9571940.0*tmp_init_57 + 1880854.0
        self._gr_k_771 = -54270475.0*nu - 14437500.0*tmp_init_101 + 71836100.0*tmp_init_57 + 9760970.0
        self._gr_k_772 = 1.6767900068681319e-8*tmp_init_223
        self._gr_k_773 = 1189335.0*nu + tmp_init_224 - 1275750.0*tmp_init_57 - 232890.0
        self._gr_k_774 = 1023750.0*nu - 1023750.0*tmp_init_57 - 204750.0
        self._gr_k_775 = -20808886.0*nu - 4353720.0*tmp_init_101 + 25190270.0*tmp_init_57 + 3910622.0
        self._gr_k_776 = -27531028.0*nu - 3504480.0*tmp_init_101 + 31578050.0*tmp_init_57 + 5259446.0
        self._gr_k_777 = -3702195.0*nu - 1609650.0*tmp_init_101 + 5130300.0*tmp_init_57 + 664020.0
        self._gr_k_778 = 123501.0*nu + 2562.0*tmp_init_101 - 123424.0*tmp_init_57 - 24780.0
        self._gr_k_779 = -93275.0*nu + 93275.0*tmp_init_57 + 18655.0
        self._gr_k_780 = -221494.0*nu - 76440.0*tmp_init_101 + 143325.0*tmp_init_57 + 53183.0
        self._gr_k_781 = -1925623.0*nu - 556710.0*tmp_init_101 + 2341010.0*tmp_init_57 + 365426.0
        self._gr_k_782 = -6897093.0*nu - 1279026.0*tmp_init_101 + 8089032.0*tmp_init_57 + 1313796.0
        self._gr_k_783 = -4655.0*nu - 4746.0*tmp_init_101 - 79688.0*tmp_init_57 + 8296.0
        self._gr_k_784 = -2638377.0*nu - 885990.0*tmp_init_101 + 3400880.0*tmp_init_57 + 487632.0
        self._gr_k_785 = 276451.0*nu + 6090.0*tmp_init_101 - 278810.0*tmp_init_57 - 55262.0
        self._gr_k_786 = 902573.0*nu + 357210.0*tmp_init_101 - 795410.0*tmp_init_57 - 199906.0
        self._gr_k_787 = 2764125.0*nu + 144690.0*tmp_init_101 - 2923830.0*tmp_init_57 - 543270.0
        self._gr_k_788 = 861735.0*nu + tmp_init_224 - 948150.0*tmp_init_57 - 167370.0
        self._gr_k_789 = -21827694.0*nu - 3105900.0*tmp_init_101 + 25469220.0*tmp_init_57 + 4142148.0
        self._gr_k_790 = -21716898.0*nu - 2212140.0*tmp_init_101 + 24620820.0*tmp_init_57 + 4157676.0
        self._gr_k_791 = -8090565.0*nu - 2069550.0*tmp_init_101 + 10186050.0*tmp_init_57 + 1497630.0
        self._gr_k_792 = -820015.0*nu - 536550.0*tmp_init_101 + 1296050.0*tmp_init_57 + 138530.0
        self._gr_k_793 = 5.365728021978022e-7*tmp_init_223
        self._gr_k_794 = -0.0078125*flagPN2
        self._gr_k_795 = 34191360.0*nu + 2580480.0*tmp_init_101 - 37416960.0*tmp_init_57 - 6635520.0
        self._gr_k_796 = 131600.0*nu + 15092.0*tmp_init_101 - 139489.0*tmp_init_57 - 26075.0
        self._gr_k_797 = -1997688.0*nu + 261912.0*tmp_init_101 + 2028726.0*tmp_init_57 + 389394.0
        self._gr_k_798 = 4770668.0*nu + 1693664.0*tmp_init_101 - 6156388.0*tmp_init_57 - 883748.0
        self._gr_k_799 = -774452.0*nu - 162624.0*tmp_init_101 + 897428.0*tmp_init_57 + 148996.0
        self._gr_k_800 = 1587376.0*nu + 747348.0*tmp_init_101 - 2206981.0*tmp_init_57 - 285719.0
        self._gr_k_801 = 1849575.0*nu + 268324.0*tmp_init_101 - 2057188.0*tmp_init_57 - 359786.0
        self._gr_k_802 = 954625.0*nu + 444500.0*tmp_init_101 - 1348550.0*tmp_init_57 - 169860.0
        self._gr_k_803 = 5658723.0*nu + 491484.0*tmp_init_101 - 6128178.0*tmp_init_57 - 1105548.0
        self._gr_k_804 = 1369305.0*nu + 165228.0*tmp_init_101 - 1516746.0*tmp_init_57 - 265944.0
        self._gr_k_805 = 112455.0*nu + 26460.0*tmp_init_101 - 133770.0*tmp_init_57 - 21420.0
        self._gr_k_806 = 37744749.0*nu + 3318924.0*tmp_init_101 - 41587938.0*tmp_init_57 - 7314360.0
        self._gr_k_807 = 14558229.0*nu + 2130548.0*tmp_init_101 - 16759806.0*tmp_init_57 - 2783812.0
        self._gr_k_808 = 4889003.0*nu + 1261540.0*tmp_init_101 - 6077470.0*tmp_init_57 - 911976.0
        self._gr_k_809 = 75215.0*nu + 67620.0*tmp_init_101 - 132790.0*tmp_init_57 - 12040.0
        self._gr_k_810 = 5404140.0*nu + 522144.0*tmp_init_101 - 5867778.0*tmp_init_57 - 1056006.0
        self._gr_k_811 = 720090.0*nu + 132300.0*tmp_init_101 - 826665.0*tmp_init_57 - 138663.0
        self._gr_k_812 = 8665132.0*nu + 3233216.0*tmp_init_101 - 11526242.0*tmp_init_57 - 1580166.0
        self._gr_k_813 = 56341992.0*nu + 7570080.0*tmp_init_101 - 64061340.0*tmp_init_57 - 10823028.0
        self._gr_k_814 = 31154214.0*nu + 6912612.0*tmp_init_101 - 37633239.0*tmp_init_57 - 5873001.0
        self._gr_k_815 = 2116870.0*nu + 446348.0*tmp_init_101 - 2025191.0*tmp_init_57 - 443985.0
        self._gr_k_816 = 938490.0*nu + 608580.0*tmp_init_101 - 1456665.0*tmp_init_57 - 160671.0
        self._gr_k_817 = 4093075.0*nu + 92540.0*tmp_init_101 - 4103680.0*tmp_init_57 - 820350.0
        self._gr_k_818 = 64440054.0*nu + 14136472.0*tmp_init_101 - 77117824.0*tmp_init_57 - 12205244.0
        self._gr_k_819 = -6775818.0*nu + 1399944.0*tmp_init_101 + 6906872.0*tmp_init_57 + 1303932.0
        self._gr_k_820 = 33235027.0*nu + 10962644.0*tmp_init_101 - 42795508.0*tmp_init_57 - 6140754.0
        self._gr_k_821 = 5364135.0*nu + 2810220.0*tmp_init_101 - 7742560.0*tmp_init_57 - 949254.0
        self._gr_k_822 = 1094548.0*nu + 236096.0*tmp_init_101 - 1272082.0*tmp_init_57 - 210438.0
        self._gr_k_823 = 23814.0*nu + tmp_init_225 - 23947.0*tmp_init_57 - 4757.0
        self._gr_k_824 = 485478.0*nu + 37884.0*tmp_init_101 - 542563.0*tmp_init_57 - 93285.0
        self._gr_k_825 = 140889.0*nu + 2772.0*tmp_init_101 - 142674.0*tmp_init_57 - 28104.0
        self._gr_k_826 = 11179.0*nu + tmp_init_225 - 11312.0*tmp_init_57 - 2230.0
        self._gr_k_827 = 415835.0*nu + tmp_init_226 - 492380.0*tmp_init_57 - 79522.0
        self._gr_k_828 = 4464523.0*nu - 130228.0*tmp_init_101 - 4725014.0*tmp_init_57 - 866856.0
        self._gr_k_829 = 4199139.0*nu + 826644.0*tmp_init_101 - 4822818.0*tmp_init_57 - 809988.0
        self._gr_k_830 = -2197503.0*nu - 1239924.0*tmp_init_101 + 3148698.0*tmp_init_57 + 393396.0
        self._gr_k_831 = 1.7438616071428571e-5*tmp_init_223
        self._gr_k_832 = -9453440.0*nu + 4800000.0*tmp_init_57 + 2474480.0
        self._gr_k_833 = -1213440.0*nu + 368640.0*tmp_init_57 + 352512.0
        self._gr_k_834 = 123420.0*nu - 39985.0
        self._gr_k_835 = 577730.0*nu + 3840.0*tmp_init_57 - 187475.0
        self._gr_k_836 = 878110.0*nu + 13440.0*tmp_init_57 - 285589.0
        self._gr_k_837 = 6077810.0*nu - 1829280.0*tmp_init_57 - 1753407.0
        self._gr_k_838 = 3464210.0*nu - 1401600.0*tmp_init_57 - 951111.0
        self._gr_k_839 = 505630.0*nu - 348000.0*tmp_init_57 - 119485.0
        self._gr_k_840 = 1866600.0*nu - 587520.0*tmp_init_57 - 534954.0
        self._gr_k_841 = 7290040.0*nu - 2081760.0*tmp_init_57 - 2121000.0
        self._gr_k_842 = 1761960.0*nu - tmp_init_227 - 465912.0
        self._gr_k_843 = 5464340.0*nu - 1860960.0*tmp_init_57 - 1546530.0
        self._gr_k_844 = 617690.0*nu + 15360.0*tmp_init_57 - 201627.0
        self._gr_k_845 = 158130.0*nu - 151200.0*tmp_init_57 - 31395.0
        self._gr_k_846 = 2339300.0*nu - 824160.0*tmp_init_57 - 658930.0
        self._gr_k_847 = 645530.0*nu - 366240.0*tmp_init_57 - 163547.0
        self._gr_k_848 = 191100.0*nu - 115200.0*tmp_init_57 - 46965.0
        self._gr_k_849 = 2921580.0*nu - tmp_init_227 - 852221.0
        self._gr_k_850 = 868360.0*nu - 340800.0*tmp_init_57 - 239034.0
        self._gr_k_851 = 9100.0*nu - 16800.0*tmp_init_57 - 665.0
        self._gr_k_852 = 1484912.0 - 4583040.0*nu
        self._gr_k_853 = 2.2194602272727273e-6*tmp_init_229
        self._gr_k_854 = 294.0*nu - 413.0
        self._gr_k_855 = 858.0*nu - 2071.0
        self._gr_k_856 = tmp_init_14
        self._gr_k_857 = tmp_init_117 - 459.0
        self._gr_k_858 = nu - 26.0
        self._gr_k_859 = tmp_init_123 - 23.0
        self._gr_k_860 = tmp_init_2 - 1118.0
        self._gr_k_861 = tmp_init_43
        self._gr_k_862 = tmp_init_190 - 656.0
        self._gr_k_863 = 0.0029761904761904762*flagPN1
        self._gr_k_864 = 40425.0*nu*(23247.0*tmp_init_230 + 1950016.0) + 1027703600.0*tmp_init_101 + 83738639600.0*tmp_init_57 - 1134796585384.0
        self._gr_k_865 = -121275.0*nu*(2583.0*tmp_init_230 + 63584.0) + 1477014000.0*tmp_init_101 - 5013531600.0*tmp_init_57 + 13948768680.0
        self._gr_k_866 = -7330862540800.0*nu + 32491778400.0*tmp_init_101 + tmp_init_231*(1927.0*nu - 5120.0) + tmp_init_233 + 2352501887200.0*tmp_init_57 + 898822771760.0
        self._gr_k_867 = -5699427531200.0*nu + 9147208800.0*tmp_init_101 + 76403250.0*tmp_init_230*(tmp_init_234 - 5120.0) + 478362931200.0*tmp_init_232 + 1097751423200.0*tmp_init_57 + 3504456665254.2916
        self._gr_k_868 = 22127822000.0*tmp_init_101 + tmp_init_24*(639036783.0*tmp_init_230 - 88091253280.0) + 1306529180400.0*tmp_init_57 - 706784882744.0
        self._gr_k_869 = 525.0*nu*(8950095.0*tmp_init_230 - 336795968.0) + 6974310000.0*tmp_init_101 + 218177828400.0*tmp_init_57 - 119671514760.0
        self._gr_k_870 = 1925.0*nu*(488187.0*tmp_init_230 - 5303152.0) + 142650200.0*tmp_init_101 - 4141075400.0*tmp_init_57 - 33580920912.0
        self._gr_k_871 = -673025502000.0*nu + 4751741000.0*tmp_init_101 + 38201625.0*tmp_init_230*(tmp_init_235 + 256.0) - 11959073280.0*tmp_init_232 + 318518465000.0*tmp_init_57 - 1458076493386.1573
        self._gr_k_872 = 76518536000.0*nu + 2249781000.0*tmp_init_101 + 7640325.0*tmp_init_230*(tmp_init_236 + 6400.0) - 59795366400.0*tmp_init_232 + 99362132200.0*tmp_init_57 - 1649106996922.7864
        self._gr_k_873 = -303391271200.0*nu + 3692360600.0*tmp_init_101 + 12843386325.0*tmp_init_237 + 150610899000.0*tmp_init_57 - 520959078152.0
        self._gr_k_874 = -46868530050.0*nu + 489881325.0*tmp_init_101 + 19761620925.0*tmp_init_57 + 1748263800.0
        self._gr_k_875 = 745113600.0*tmp_init_104
        self._gr_k_876 = 16299360.0*flagTail
        self._gr_k_877 = -84111050800.0*nu + 1457309000.0*tmp_init_101 + 69793663800.0*tmp_init_57 - 2484856010.0
        self._gr_k_878 = -158615114000.0*nu + 1077216875.0*tmp_init_101 - 7972715520.0*tmp_init_232 - 20374200.0*tmp_init_238 + 39696230875.0*tmp_init_57 + 64675517238.895141
        self._gr_k_879 = -896.0*nu + 84.0*tmp_init_57 + 1715.0
        self._gr_k_880 = -20229224400.0*nu + 5128200.0*tmp_init_101 + 5971303800.0*tmp_init_57 - 4276509930.0
        self._gr_k_881 = -1089295752000.0*nu + 4925697200.0*tmp_init_101 - 31890862080.0*tmp_init_232 - 81496800.0*tmp_init_238 + 252640254800.0*tmp_init_57 + 621805525955.58056
        self._gr_k_882 = -547359716300.0*nu + 3533523300.0*tmp_init_101 + 192559333100.0*tmp_init_57 + 122018934925.0
        self._gr_k_883 = -1514967300.0*nu + 138253500.0*tmp_init_101 - 1253983500.0*tmp_init_57 - 879073965.0
        self._gr_k_884 = -2760263100.0*nu + 347287500.0*tmp_init_101 + 8485942500.0*tmp_init_57 + 535442355.0
        self._gr_k_885 = -665782469300.0*nu + 463918100.0*tmp_init_101 + 47836293120.0*tmp_init_232 + 122245200.0*tmp_init_238 - tmp_init_239 + 78430637500.0*tmp_init_57 + 986330853741.0
        self._gr_k_886 = -198893414400.0*nu + 1524121600.0*tmp_init_101 - 63781724160.0*tmp_init_232 - 162993600.0*tmp_init_238 + 59313203200.0*tmp_init_57 + 89950343711.161125
        self._gr_k_887 = 17068649.878071777*flagTail
        self._gr_k_888 = -340042839250.0*nu + 1659319075.0*tmp_init_101 - 39863577600.0*tmp_init_232 - 101871000.0*tmp_init_238 + 70547772875.0*tmp_init_57 + 302554064119.4757
        self._gr_k_889 = -130977000.0*nu + 20738025.0*tmp_init_101 - 316891575.0*tmp_init_57 + 3274425.0
        self._gr_k_890 = -4198057500.0*nu + 148698375.0*tmp_init_101 + 5505987375.0*tmp_init_57 + 1269321375.0
        self._gr_k_891 = -124185838700.0*nu + 163796325.0*tmp_init_101 + 16232972525.0*tmp_init_57 + 191445995125.0
        self._gr_k_892 = 505231650.0*nu - 5630625.0*tmp_init_101 - 66614625.0*tmp_init_57 - 1922243400.0
        self._gr_k_893 = -170556750.0*nu + 28940625.0*tmp_init_101 + 532121625.0*tmp_init_57 - 10804500.0
        self._gr_k_894 = -366968448000.0*nu + 40236134400.0*tmp_init_57 + 529030656000.0
        self._gr_k_895 = -1905.0*nu + 408.0*tmp_init_57 + 1845.0
        self._gr_k_896 = -6477.0*nu + tmp_init_240 + 9255.0
        self._gr_k_897 = -264.0*nu + 108.0*tmp_init_57 - 15.0
        self._gr_k_898 = 219552.0*nu - 42624.0*tmp_init_57 - 247200.0
        self._gr_k_899 = -7760.0*nu + tmp_init_162*tmp_init_230 + 288.0*tmp_init_57 - 110640.0
        self._gr_k_900 = 52320.0*nu - 17280.0*tmp_init_57 - 22800.0
        self._gr_k_901 = -207856.0*nu + 4305.0*tmp_init_237 + 4896.0*tmp_init_57 - 152160.0
        self._gr_k_902 = 861.0*nu*tmp_init_230 + 153808.0*nu - 60768.0*tmp_init_57 - 132960.0
        self._gr_k_903 = -248371200.0*tmp_init_241
        self._gr_k_904 = -15523200.0*nu*(861.0*tmp_init_230 - 54704.0) - tmp_init_242 + 7451136000.0
        self._gr_k_905 = 1853944035200.0*nu - 5640249600.0*tmp_init_101 - 30561300.0*tmp_init_230*(tmp_init_234 - 2560.0) - tmp_init_233 - 618641892800.0*tmp_init_57 + 851091640400.0
        self._gr_k_906 = 1848.0*flagTail
        self._gr_k_907 = 1293600.0*tmp_init_241
        self._gr_k_908 = 1164240.0*flagTail
        self._gr_k_909 = -849181132800.0*nu + 127563448320.0*tmp_init_232 + 325987200.0*tmp_init_238 + tmp_init_242 - 651858428990.32225
        self._gr_k_910 = 112185170800.0*nu - 9965894400.0*tmp_init_232 + tmp_init_243 - tmp_init_244*(tmp_init_234 - 3200.0) - 3322371600.0*tmp_init_57 + 67015219116.868926
        self._gr_k_911 = -38186838000.0*nu - 781707600.0*tmp_init_101 - 1993178880.0*tmp_init_232 - tmp_init_244*(tmp_init_235 - 640.0) - 1916355600.0*tmp_init_57 + 362684126712.97378
        self._gr_k_912 = ccomplex.complex[double](-270423268800.0*nu - 1923635200.0*tmp_init_101 + 15280650.0*tmp_init_230*(tmp_init_147 - 2560.0) - tmp_init_239 - 75891500800.0*tmp_init_57 + 1743706674272.0, 150282147040.75997)
        self._gr_k_913 = -10094611200.0*tmp_init_101 - tmp_init_110*(438554655.0*tmp_init_230 - 28287877216.0) - 690857254400.0*tmp_init_57 + 1156690418336.0
        self._gr_k_914 = 5775.0*nu*(54243.0*tmp_init_230 - 3052096.0) - 481588800.0*tmp_init_101 + 3654789600.0*tmp_init_57 - 9949641240.0
        self._gr_k_915 = -175.0*nu*(51910551.0*tmp_init_230 - 1445337920.0) - 4717720000.0*tmp_init_101 - 177736960800.0*tmp_init_57 + 291357217480.0
        self._gr_k_916 = -35206617600.0*flagTail
        self._gr_k_917 = -2200413600.0*flagTail
        self._gr_k_918 = ccomplex.complex[double](-49460361200.0*nu - 1474880800.0*tmp_init_101 - tmp_init_231*(tmp_init_236 + 1280.0) - 34432330400.0*tmp_init_57 + 842706449292.31458, 75141073520.379985)
        self._gr_k_919 = 100840714600.0*nu - 12733875.0*tmp_init_230*(tmp_init_80 - 256.0) - 3986357760.0*tmp_init_232 + tmp_init_243 - 4047896400.0*tmp_init_57 + 42319552677.94757
        self._gr_k_920 = 128166245000.0*nu - 1671210000.0*tmp_init_101 - 7204826475.0*tmp_init_237 - 56133194000.0*tmp_init_57 + 448806143396.0
        self._gr_k_921 = -3.1954219879389947e-11*flagPN3
        self._gr_k_922 = 3877951440.0*nu + 1613222160.0*tmp_init_101 + 14075160.0*tmp_init_102 - 6859009410.0*tmp_init_57 - 590832462.0
        self._gr_k_923 = 4679700080.0*nu + 1262866000.0*tmp_init_101 + 5876200.0*tmp_init_102 - 7886254750.0*tmp_init_57 - 730848338.0
        self._gr_k_924 = 12255300.0*nu - 40338000.0*tmp_init_101 + 1822500.0*tmp_init_102 - 905175.0*tmp_init_57 - 2513025.0
        self._gr_k_925 = 585900300.0*nu + 800260800.0*tmp_init_101 + 813300.0*tmp_init_102 - 1337304075.0*tmp_init_57 - 76465965.0
        self._gr_k_926 = -1212785100.0*nu - 266944320.0*tmp_init_101 - 2586420.0*tmp_init_102 + 1982557755.0*tmp_init_57 + 193921245.0
        self._gr_k_927 = 14269500.0*nu + 101833200.0*tmp_init_101 - 2211300.0*tmp_init_102 - 71616825.0*tmp_init_57 - 543375.0
        self._gr_k_928 = -6199479.0*nu + 7719030.0*tmp_init_101 + 5514300.0*tmp_init_57 + 1167903.0
        self._gr_k_929 = 7128000.0*tmp_init_105
        self._gr_k_930 = 375269568.0*nu + 148609380.0*tmp_init_101 + 953400.0*tmp_init_102 - 655699962.0*tmp_init_57 - 57776298.0
        self._gr_k_931 = -5287221.0*nu - 3090150.0*tmp_init_101 + 121500.0*tmp_init_102 + 9348615.0*tmp_init_57 + 836622.0
        self._gr_k_932 = 163015474.0*nu + 137002352.0*tmp_init_101 + 49472.0*tmp_init_102 - 322527512.0*tmp_init_57 - 23590990.0
        self._gr_k_933 = -47595042.0*nu - 4015440.0*tmp_init_101 + 74103480.0*tmp_init_57 + 7780014.0
        self._gr_k_934 = 9711333.0*nu + 27058050.0*tmp_init_101 - 442260.0*tmp_init_102 - 28163565.0*tmp_init_57 - 1104516.0
        self._gr_k_935 = -26272995.0*nu + 11955630.0*tmp_init_101 + 138480.0*tmp_init_102 + 33077955.0*tmp_init_57 + 4636785.0
        self._gr_k_936 = 36322825.0*nu + 72130850.0*tmp_init_101 + 954800.0*tmp_init_102 - 87438065.0*tmp_init_57 - 5062651.0
        self._gr_k_937 = 565317295.0*nu + 427914170.0*tmp_init_101 - 18880.0*tmp_init_102 - 1083162215.0*tmp_init_57 - 83952469.0
        self._gr_k_938 = 87802795.0*nu + 160846550.0*tmp_init_101 - 2111200.0*tmp_init_102 - 212638475.0*tmp_init_57 - 11567905.0
        self._gr_k_939 = 6422268.0*nu + 25422180.0*tmp_init_101 - 5880.0*tmp_init_102 - 20539122.0*tmp_init_57 - 793578.0
        self._gr_k_940 = 17913851.0*nu + 33369022.0*tmp_init_101 - 372008.0*tmp_init_102 - 43044322.0*tmp_init_57 - 2416349.0
        self._gr_k_941 = 134385110.0*nu - 24551900.0*tmp_init_101 + 247300.0*tmp_init_102 - 191293285.0*tmp_init_57 - 22636667.0
        self._gr_k_942 = 52596810.0*nu - 114106860.0*tmp_init_101 + 1382940.0*tmp_init_102 - 34840395.0*tmp_init_57 - 9499653.0
        self._gr_k_943 = -57024000.0*tmp_init_105
        self._gr_k_944 = 427224.0*nu - 155437.0
        self._gr_k_945 = 633024.0*nu - 215922.0
        self._gr_k_946 = 130368.0*nu - 40184.0
        self._gr_k_947 = -7290.0*nu + 90.0*tmp_init_57 + 2385.0
        self._gr_k_948 = 38850.0*nu - 5730.0*tmp_init_57 - 11793.0
        self._gr_k_949 = 53410.0*nu - 5770.0*tmp_init_57 - 16667.0
        self._gr_k_950 = 3750.0*nu - 1950.0*tmp_init_57 - 885.0
        self._gr_k_951 = 513.0*nu - 171.0
        self._gr_k_952 = -6360.0*nu + tmp_init_240 + 1956.0
        self._gr_k_953 = -1729.0*nu + 520.0*tmp_init_57 + 479.0
        self._gr_k_954 = 310.0*nu + 170.0*tmp_init_57 - 107.0
        self._gr_k_955 = 15390.0*nu - 3510.0*tmp_init_57 - 4473.0
        self._gr_k_956 = 834.0 - 72.0*nu
        self._gr_k_957 = tmp_init_129
        self._gr_k_958 = 2970.0*nu - 990.0
        self._gr_k_959 = 7425.0*nu - 2475.0
        self._gr_k_960 = 5940.0*nu - 1980.0
        self._gr_k_961 = 7909930.0*nu - 10779600.0*tmp_init_57 - 1438932.0
        self._gr_k_962 = 1485.0*nu - 495.0
        self._gr_k_963 = tmp_init_88
        self._gr_k_964 = 8173195.0*nu - 10729020.0*tmp_init_57 - 1532349.0
        self._gr_k_965 = 3960.0 - 11880.0*nu
        self._gr_k_966 = -7648795.0*nu + 10709130.0*tmp_init_57 + 1359669.0
        self._gr_k_967 = 10890.0 - 32670.0*nu
        self._gr_k_968 = 21120.0 - 63360.0*nu
        self._gr_k_969 = 168960.0 - 506880.0*nu
        self._gr_k_970 = tmp_init_85
        self._gr_k_971 = 220.0*tmp_init_96
        self._gr_k_972 = 118800.0*tmp_init_92
        self._gr_k_973 = -tmp_init_91
        self._gr_k_974 = tmp_init_192*tmp_init_47
        self._gr_k_975 = ccomplex.complex[double](-31680.0*cmath.pow(tmp_init_47, 2)*tmp_init_76, -31680.0*tmp_init_47*(21.0 - 21.0*tmp_init_9))
        self._gr_k_976 = 0.00066666666666666667*tmp_init_245
        self._gr_k_977 = 0.00011431184270690444*tmp_init_245
        self._gr_k_978 = -8964888.0*nu + 2182584.0*tmp_init_57 + 3795138.0
        self._gr_k_979 = -5333528.0*nu + 1814144.0*tmp_init_57 + 2121884.0
        self._gr_k_980 = -313608.0*nu + 150784.0*tmp_init_57 + 154452.0
        self._gr_k_981 = -236128.0*nu + 5236.0*tmp_init_57 + 117441.0
        self._gr_k_982 = -1006120.0*nu + 500180.0*tmp_init_57 + 359885.0
        self._gr_k_983 = -4024320.0*nu + 522240.0*tmp_init_57 + 1804800.0
        self._gr_k_984 = -304152.0*nu + 3864.0*tmp_init_57 + 149514.0
        self._gr_k_985 = -506936.0*nu + 384024.0*tmp_init_57 + 191162.0
        self._gr_k_986 = 1572.0*nu + 24548.0*tmp_init_57 - 4293.0
        self._gr_k_987 = -1304220.0*nu + 574372.0*tmp_init_57 + 503115.0
        self._gr_k_988 = -378196.0*nu + 296868.0*tmp_init_57 + 150215.0
        self._gr_k_989 = -80532.0*nu + 14756.0*tmp_init_57 + 38151.0
        self._gr_k_990 = -3640124.0*nu + 1239084.0*tmp_init_57 + 1471813.0
        self._gr_k_991 = -1261788.0*nu + 562828.0*tmp_init_57 + 473709.0
        self._gr_k_992 = -264996.0*nu + 7140.0*tmp_init_57 + 129663.0
        self._gr_k_993 = -12180952.0*nu + 2108824.0*tmp_init_57 + 5334122.0
        self._gr_k_994 = -2749188.0*nu + 1043588.0*tmp_init_57 + 1055639.0
        self._gr_k_995 = -384332.0*nu - 32084.0*tmp_init_57 + 217581.0
        self._gr_k_996 = -8208008.0*nu + 2064776.0*tmp_init_57 + 3424750.0
        self._gr_k_997 = -358956.0*nu + 212940.0*tmp_init_57 + 118053.0
        self._gr_k_998 = -3122576.0*nu + 587088.0*tmp_init_57 + 1347756.0
        self._gr_k_999 = -58380.0*nu + 2380.0*tmp_init_57 + 28245.0
        self._gr_k_1000 = -6370116.0*nu + 856020.0*tmp_init_57 + 2849499.0
        self._gr_k_1001 = -313560.0*nu + 143000.0*tmp_init_57 + 112970.0
        self._gr_k_1002 = -1305172.0*nu + 374132.0*tmp_init_57 + 528651.0
        self._gr_k_1003 = -30940.0*nu + 23660.0*tmp_init_57 + 8645.0
        self._gr_k_1004 = 1694.0*nu + tmp_init_240 - 963.0
        self._gr_k_1005 = -17014.0*nu + 10752.0*tmp_init_57 + 6587.0
        self._gr_k_1006 = -2.4730927508705286e-6*tmp_init_221
        self._gr_k_1007 = 0.0041666666666666667*tmp_init_246
        self._gr_k_1008 = -735550.0*nu + 222840.0*tmp_init_57 + 217455.0
        self._gr_k_1009 = -244630.0*nu + 85680.0*tmp_init_57 + 72411.0
        self._gr_k_1010 = -58010.0*nu + 1800.0*tmp_init_57 + 19181.0
        self._gr_k_1011 = -203570.0*nu + 97200.0*tmp_init_57 + 56081.0
        self._gr_k_1012 = -1249375270661.0*nu - 263476921680.0*tmp_init_101 + 16676067240.0*tmp_init_102 + 2024133173874.0*tmp_init_57 + 201105561909.0
        self._gr_k_1013 = -40471582725.0*nu - 20011468960.0*tmp_init_101 + 1011755640.0*tmp_init_102 + 70646799230.0*tmp_init_57 + 6369550845.0
        self._gr_k_1014 = -26623306605.0*nu - 56211187200.0*tmp_init_101 + 5671058400.0*tmp_init_102 + 67937456720.0*tmp_init_57 + 3337679835.0
        self._gr_k_1015 = -38915415.0*nu - 3219678000.0*tmp_init_101 + 258627600.0*tmp_init_102 + 1572846660.0*tmp_init_57 - 45734535.0
        self._gr_k_1016 = -334430555935.0*nu - 97639903200.0*tmp_init_101 - 394228800.0*tmp_init_102 + 554433295080.0*tmp_init_57 + 53494194201.0
        self._gr_k_1017 = -165848760391.0*nu - 177218045760.0*tmp_init_101 + 6215637960.0*tmp_init_102 + 340786141794.0*tmp_init_57 + 23904456207.0
        self._gr_k_1018 = -516565909021.0*nu - 267803211760.0*tmp_init_101 + 4371623760.0*tmp_init_102 + 919301620804.0*tmp_init_57 + 79907942931.0
        self._gr_k_1019 = -1300221615.0*nu - 6077979600.0*tmp_init_101 + 1669714200.0*tmp_init_102 + 4744326510.0*tmp_init_57 + 110756415.0
        self._gr_k_1020 = 8456448.0*tmp_init_104*(84.0*nu + tmp_init_17*tmp_init_93 - 21.0)
        self._gr_k_1021 = -1268467200.0*tmp_init_105
        self._gr_k_1022 = -96180.0*nu + 38100.0*tmp_init_57 + 27189.0
        self._gr_k_1023 = -19300.0*nu + 1700.0*tmp_init_57 + 6225.0
        self._gr_k_1024 = -1026600.0*nu + 102600.0*tmp_init_57 + 326122.0
        self._gr_k_1025 = -367160.0*nu + 74120.0*tmp_init_57 + 112130.0
        self._gr_k_1026 = -155340.0*nu + 14100.0*tmp_init_57 + 49889.0
        self._gr_k_1027 = -10300.0*nu + 8100.0*tmp_init_57 + 2445.0
        self._gr_k_1028 = -1061760.0*nu + 544320.0*tmp_init_57 + 295600.0
        self._gr_k_1029 = -913920.0*nu + 69120.0*tmp_init_57 + 292608.0
        self._gr_k_1030 = -13420.0*nu + 660.0*tmp_init_57 + 4411.0
        self._gr_k_1031 = -141220.0*nu + 64440.0*tmp_init_57 + 40190.0
        self._gr_k_1032 = -160720.0*nu + 70740.0*tmp_init_57 + 45223.0
        self._gr_k_1033 = -624020.0*nu + 201840.0*tmp_init_57 + 182268.0
        self._gr_k_1034 = -81660.0*nu + 5100.0*tmp_init_57 + 26595.0
        self._gr_k_1035 = -294540.0*nu + 49680.0*tmp_init_57 + 92964.0
        self._gr_k_1036 = -1431220.0*nu + 258840.0*tmp_init_57 + 441630.0
        self._gr_k_1037 = -97840.0*nu + 56700.0*tmp_init_57 + 25695.0
        self._gr_k_1038 = -56320.0*nu + 10560.0*tmp_init_57 + 17776.0
        self._gr_k_1039 = -3886590981404.0*nu - 3137262583560.0*tmp_init_101 + 109045026720.0*tmp_init_102 + 7454955646096.0*tmp_init_57 + 582048988434.0
        self._gr_k_1040 = -39125807670.0*nu - 44371911780.0*tmp_init_101 + 3029571720.0*tmp_init_102 + 79549198050.0*tmp_init_57 + 5809143435.0
        self._gr_k_1041 = -1613722100870.0*nu - 2054791491060.0*tmp_init_101 + 169718943240.0*tmp_init_102 + 3454979220770.0*tmp_init_57 + 228028343631.0
        self._gr_k_1042 = -2156020435172.0*nu - 1499227649640.0*tmp_init_101 - 21311211600.0*tmp_init_102 + 4017478901548.0*tmp_init_57 + 328078544454.0
        self._gr_k_1043 = -311608512002.0*nu - 95953926180.0*tmp_init_101 + 6157640160.0*tmp_init_102 + 513644884508.0*tmp_init_57 + 50275799853.0
        self._gr_k_1044 = -190174734210.0*nu - 417632394900.0*tmp_init_101 + 74778782400.0*tmp_init_102 + 486052274820.0*tmp_init_57 + 23930465505.0
        self._gr_k_1045 = 59083398717.0*nu + 38956585920.0*tmp_init_101 + 31829588280.0*tmp_init_102 - 115516259558.0*tmp_init_57 - 8695187235.0
        self._gr_k_1046 = -509839634739.0*nu - 665214102000.0*tmp_init_101 + 85431941280.0*tmp_init_102 + 1089095902916.0*tmp_init_57 + 72519322419.0
        self._gr_k_1047 = 1788473723.0*nu + 27662869920.0*tmp_init_101 + 407478960.0*tmp_init_102 - 16878647912.0*tmp_init_57 + 249658005.0
        self._gr_k_1048 = -241489286437.0*nu - 420974872080.0*tmp_init_101 + 73960873560.0*tmp_init_102 + 564157596898.0*tmp_init_57 + 32490993675.0
        self._gr_k_1049 = -159704154666.0*nu - 244174159740.0*tmp_init_101 + 21351447180.0*tmp_init_102 + 362170692079.0*tmp_init_57 + 21773306961.0
        self._gr_k_1050 = -4142712420.0*nu - 9404599050.0*tmp_init_101 + 646569000.0*tmp_init_102 + 10674452250.0*tmp_init_57 + 535190040.0
        self._gr_k_1051 = -7712613964.0*nu + 39228420.0*tmp_init_101 + tmp_init_247 + 11557279426.0*tmp_init_57 + 1284791013.0
        self._gr_k_1052 = 7738208552.0*nu + 1985310180.0*tmp_init_101 + 5762715840.0*tmp_init_102 - 13941957288.0*tmp_init_57 - 1174983417.0
        self._gr_k_1053 = 5572674804.0*nu + 22531531680.0*tmp_init_101 + 10056790800.0*tmp_init_102 - 22420997536.0*tmp_init_57 - 325019064.0
        self._gr_k_1054 = -798513757740.0*nu - 948603516000.0*tmp_init_101 + 83724001200.0*tmp_init_102 + 1672830975760.0*tmp_init_57 + 114401100984.0
        self._gr_k_1055 = -9824847135.0*nu - 1259688780.0*tmp_init_101 + 701923320.0*tmp_init_102 + 14864799810.0*tmp_init_57 + 1661299350.0
        self._gr_k_1056 = -749764054730.0*nu - 718202915640.0*tmp_init_101 + 30858528960.0*tmp_init_102 + 1489295471580.0*tmp_init_57 + 110663924340.0
        self._gr_k_1057 = -160248504927.0*nu - 297486385980.0*tmp_init_101 + 49477889160.0*tmp_init_102 + 384097725298.0*tmp_init_57 + 21145875702.0
        self._gr_k_1058 = -12222995400.0*nu - 35560359450.0*tmp_init_101 + 7513713900.0*tmp_init_102 + 35179185195.0*tmp_init_57 + 1389821895.0
        self._gr_k_1059 = -659903707828.0*nu - 568353296070.0*tmp_init_101 + 19294158240.0*tmp_init_102 + 1285424716352.0*tmp_init_57 + 97954253934.0
        self._gr_k_1060 = -100465665930.0*nu - 38813521740.0*tmp_init_101 + 1646947260.0*tmp_init_102 + 170119621875.0*tmp_init_57 + 16003569645.0
        self._gr_k_1061 = -1204596047180.0*nu - 574872035640.0*tmp_init_101 + 9169707960.0*tmp_init_102 + 2115509466910.0*tmp_init_57 + 187652542530.0
        self._gr_k_1062 = -510392955808.0*nu - 231888240150.0*tmp_init_101 - 9973301940.0*tmp_init_102 + 889043088507.0*tmp_init_57 + 80060253747.0
        self._gr_k_1063 = -5100859819.0*nu + 5479763520.0*tmp_init_101 + tmp_init_247 + 4927556326.0*tmp_init_57 + 949340898.0
        self._gr_k_1064 = 17178074190.0*nu + 8547450240.0*tmp_init_101 + 11235236040.0*tmp_init_102 - 32646915770.0*tmp_init_57 - 2553893670.0
        self._gr_k_1065 = -29846527459.0*nu - 60245542560.0*tmp_init_101 + 15158646720.0*tmp_init_102 + 72462427156.0*tmp_init_57 + 3941678916.0
        self._gr_k_1066 = -2607017167872.0*nu - 188596961280.0*tmp_init_101 + 70497423360.0*tmp_init_102 + 4036793499648.0*tmp_init_57 + 426583093248.0
        self._gr_k_1067 = -440440.0*tmp_init_103
        self._gr_k_1068 = -253693440.0*tmp_init_106
        self._gr_k_1069 = -20295475200.0*tmp_init_105
        self._gr_k_1070 = -3.9457070707070707e-7*tmp_init_98
        self._gr_k_1071 = -9.8544132635041726e-11*tmp_init_107
        self._gr_k_1072 = -3.5511363636363636e-5*tmp_init_229
        self._gr_k_1073 = 38062080.0*nu + 6451200.0*tmp_init_101 - 45158400.0*tmp_init_57 - 7188480.0
        self._gr_k_1074 = 4806585.0*nu + 226380.0*tmp_init_101 - 4909730.0*tmp_init_57 - 958944.0
        self._gr_k_1075 = -456757.0*nu + tmp_init_226 + 410830.0*tmp_init_57 + 92372.0
        self._gr_k_1076 = -2045876.0*nu + 1021160.0*tmp_init_101 + 2038190.0*tmp_init_57 + 380658.0
        self._gr_k_1077 = -936054.0*nu + 259140.0*tmp_init_101 + 817985.0*tmp_init_57 + 189927.0
        self._gr_k_1078 = 4728234.0*nu + 2711380.0*tmp_init_101 - 6794095.0*tmp_init_57 - 846041.0
        self._gr_k_1079 = 3115.0*nu + tmp_init_248 - 112490.0*tmp_init_57 - 3748.0
        self._gr_k_1080 = 50953644.0*nu + 16855020.0*tmp_init_101 - 65980110.0*tmp_init_57 - 9384318.0
        self._gr_k_1081 = -18917885.0*nu + 2502920.0*tmp_init_101 + 21266560.0*tmp_init_57 + 3510750.0
        self._gr_k_1082 = 37697324.0*nu + 15828120.0*tmp_init_101 - 51815120.0*tmp_init_57 - 6781600.0
        self._gr_k_1083 = -8782676.0*nu - 564200.0*tmp_init_101 + 10736320.0*tmp_init_57 + 1605200.0
        self._gr_k_1084 = 245294.0*nu + 216930.0*tmp_init_101 - 346325.0*tmp_init_57 - 46597.0
        self._gr_k_1085 = 7732942.0*nu + 4498690.0*tmp_init_101 - 11655245.0*tmp_init_57 - 1338925.0
        self._gr_k_1086 = 334990698.0*nu + 107331000.0*tmp_init_101 - 437244500.0*tmp_init_57 - 61300128.0
        self._gr_k_1087 = 368706282.0*nu + 88379480.0*tmp_init_101 - 456038660.0*tmp_init_57 - 68780752.0
        self._gr_k_1088 = 126071981.0*nu + 56808220.0*tmp_init_101 - 178086650.0*tmp_init_57 - 22379088.0
        self._gr_k_1089 = -32868115.0*nu - 8175860.0*tmp_init_101 + 43907710.0*tmp_init_57 + 5860968.0
        self._gr_k_1090 = 17258745.0*nu + 11637500.0*tmp_init_101 - 27540170.0*tmp_init_57 - 2902984.0
        self._gr_k_1091 = -2266110.0*nu + 2294775.0*tmp_init_57 + 450765.0
        self._gr_k_1092 = 64265523.0*nu + 29288280.0*tmp_init_101 - 88106760.0*tmp_init_57 - 11646378.0
        self._gr_k_1093 = -2193135.0*nu + 1007160.0*tmp_init_101 + 1728720.0*tmp_init_57 + 449658.0
        self._gr_k_1094 = -91035.0*nu + 103488.0*tmp_init_101 + 924.0*tmp_init_57 + 22974.0
        self._gr_k_1095 = -89229.0*nu + 91140.0*tmp_init_57 + 17682.0
        self._gr_k_1096 = 6565132.0*nu + 3563840.0*tmp_init_101 - 10285730.0*tmp_init_57 - 1095942.0
        self._gr_k_1097 = 27143361.0*nu + 14643720.0*tmp_init_101 - 39492600.0*tmp_init_57 - 4788558.0
        self._gr_k_1098 = -87724994.0*nu - 30657760.0*tmp_init_101 + 117406345.0*tmp_init_57 + 15876819.0
        self._gr_k_1099 = -396459.0*nu - 254800.0*tmp_init_101 + 632100.0*tmp_init_57 + 66374.0
        self._gr_k_1100 = -22811124.0*nu - 11854080.0*tmp_init_101 + 33759390.0*tmp_init_57 + 3962490.0
        self._gr_k_1101 = -196727202.0*nu - 37038960.0*tmp_init_101 + 236557545.0*tmp_init_57 + 36989667.0
        self._gr_k_1102 = -175368984.0*nu - 43606080.0*tmp_init_101 + 219888900.0*tmp_init_57 + 32503692.0
        self._gr_k_1103 = -2309790.0*nu - 1940400.0*tmp_init_101 + 4031895.0*tmp_init_57 + 369789.0
        self._gr_k_1104 = -16138017.0*nu - 2686992.0*tmp_init_101 + 19129404.0*tmp_init_57 + 3047970.0
        self._gr_k_1105 = -1995889.0*nu - 812000.0*tmp_init_101 + 2784740.0*tmp_init_57 + 354762.0
        self._gr_k_1106 = -9870679.0*nu - 2006928.0*tmp_init_101 + 12048596.0*tmp_init_57 + 1844798.0
        self._gr_k_1107 = -5514215.0*nu - 1529248.0*tmp_init_101 + 7085596.0*tmp_init_57 + 1011846.0
        self._gr_k_1108 = -30429.0*nu - 35280.0*tmp_init_101 + 61740.0*tmp_init_57 + 4410.0
        self._gr_k_1109 = tmp_init_190 - 81.0
        self._gr_k_1110 = 1406.0 - 4488.0*nu
        self._gr_k_1111 = 108864.0*nu - 32643.0
        self._gr_k_1112 = 1565760.0*nu - 476080.0
        self._gr_k_1113 = 1282176.0*nu - 400752.0
        self._gr_k_1114 = 9.3333333333333333e-7*tmp_init_249*(14664.0*nu - 6547.0)
        self._gr_k_1115 = -71527008.0*nu + 4032700.0*tmp_init_101 + 74789085.0*tmp_init_57 + 13910575.0
        self._gr_k_1116 = -47184186.0*nu + 1614060.0*tmp_init_101 + 49869645.0*tmp_init_57 + 9160539.0
        self._gr_k_1117 = -182152236.0*nu - 58696960.0*tmp_init_101 + 247342830.0*tmp_init_57 + 32519738.0
        self._gr_k_1118 = -63929754.0*nu - 31504620.0*tmp_init_101 + 97009605.0*tmp_init_57 + 10850667.0
        self._gr_k_1119 = -16262085.0*nu + tmp_init_248 + 17246460.0*tmp_init_57 + 3155542.0
        self._gr_k_1120 = -36014517.0*nu - 16470860.0*tmp_init_101 + 53308920.0*tmp_init_57 + 6191122.0
        self._gr_k_1121 = 0.10416666666666667*tmp_init_249
        self._gr_k_1122 = 0.00081380208333333333*tmp_init_246
        self._gr_k_1123 = -323412495.0*nu - 119905940.0*tmp_init_101 + 454178130.0*tmp_init_57 + 56899900.0
        self._gr_k_1124 = -75428773.0*nu + 3292660.0*tmp_init_101 + 79362710.0*tmp_init_57 + 14654484.0
        self._gr_k_1125 = -445547319.0*nu - 119851340.0*tmp_init_101 + 583237830.0*tmp_init_57 + 80731744.0
        self._gr_k_1126 = -62252925.0*nu - 34246100.0*tmp_init_101 + 98211330.0*tmp_init_57 + 10346896.0
        self._gr_k_1127 = -252273.0*nu + tmp_init_250 + 253890.0*tmp_init_57 + 49392.0
        self._gr_k_1128 = -2308733.0*nu - 1067220.0*tmp_init_101 + 3496150.0*tmp_init_57 + 390460.0
        self._gr_k_1129 = -6760915.0*nu - 2377060.0*tmp_init_101 + 9452870.0*tmp_init_57 + 1189360.0
        self._gr_k_1130 = -13032789.0*nu - 3501036.0*tmp_init_101 + 17006682.0*tmp_init_57 + 2365968.0
        self._gr_k_1131 = -12399135.0*nu - 3305148.0*tmp_init_101 + 16026066.0*tmp_init_57 + 2263380.0
        self._gr_k_1132 = -380583.0*nu - 279300.0*tmp_init_101 + 683550.0*tmp_init_57 + 58128.0
        self._gr_k_1133 = -11639243.0*nu - 3440668.0*tmp_init_101 + 15567706.0*tmp_init_57 + 2089428.0
        self._gr_k_1134 = -14553.0*nu - tmp_init_250 + 48510.0*tmp_init_57 + 924.0
        self._gr_k_1135 = -200297895.0*nu + 14908740.0*tmp_init_101 + 207517800.0*tmp_init_57 + 39014766.0
        self._gr_k_1136 = -608982675.0*nu - 289820300.0*tmp_init_101 + 923076840.0*tmp_init_57 + 103154758.0
        self._gr_k_1137 = -3487220982.0*nu - 991105080.0*tmp_init_101 + 4612502160.0*tmp_init_57 + 629308812.0
        self._gr_k_1138 = -2140269558.0*nu - 748397160.0*tmp_init_101 + 2972476920.0*tmp_init_57 + 378104628.0
        self._gr_k_1139 = -2643598419.0*nu - 639588740.0*tmp_init_101 + 3385876620.0*tmp_init_57 + 483369802.0
        self._gr_k_1140 = -57114855.0*nu - 46084500.0*tmp_init_101 + 105503580.0*tmp_init_57 + 8592066.0
        self._gr_k_1141 = -665672616.0*nu - 208695480.0*tmp_init_101 + 899175690.0*tmp_init_57 + 119082702.0
        self._gr_k_1142 = -623541996.0*nu - 153907040.0*tmp_init_101 + 802874100.0*tmp_init_57 + 113734420.0
        self._gr_k_1143 = -271733196.0*nu - 112925120.0*tmp_init_101 + 394288860.0*tmp_init_57 + 47068300.0
        self._gr_k_1144 = -35372736.0*nu - 22756580.0*tmp_init_101 + 59267145.0*tmp_init_57 + 5676643.0
        self._gr_k_1145 = 105336.0*nu - 43629.0
        self._gr_k_1146 = 115584.0*nu - 50156.0
        self._gr_k_1147 = 2.0931123542524005e-5*tmp_init_251*(tmp_init_126 - 19.0)
        self._gr_k_1148 = -27721764.0*nu - 15781920.0*tmp_init_101 + 44826390.0*tmp_init_57 + 4529154.0
        self._gr_k_1149 = -271452132.0*nu - 67737600.0*tmp_init_101 + 348452790.0*tmp_init_57 + 49625730.0
        self._gr_k_1150 = -294200312.0*nu - 92383200.0*tmp_init_101 + 398650420.0*tmp_init_57 + 52526716.0
        self._gr_k_1151 = -129148866.0*nu - 51037140.0*tmp_init_101 + 185852205.0*tmp_init_57 + 22427691.0
        self._gr_k_1152 = -12254270.0*nu + 1236900.0*tmp_init_101 + 12513235.0*tmp_init_57 + 2393317.0
        self._gr_k_1153 = -374659362.0*nu - 101421180.0*tmp_init_101 + 490337085.0*tmp_init_57 + 67914387.0
        self._gr_k_1154 = -1832670.0*nu - 2102100.0*tmp_init_101 + 4039875.0*tmp_init_57 + 237405.0
        self._gr_k_1155 = -44513280.0*nu - 12902400.0*tmp_init_101 + 58060800.0*tmp_init_57 + 8110080.0
        self._gr_k_1156 = -2.3921284048598863e-8*tmp_init_223
        self._gr_k_1157 = -4.3402777777777778e-5*flagPN52*tmp_init_228
        self._gr_k_1158 = -2.1433470507544582e-5*flagPN52
        self._gr_k_1159 = -tmp_init_99
        self._gr_k_1160 = -1.0960297418630752e-8*tmp_init_107
        self._gr_k_1161 = -tmp_init_217
        self._gr_k_1162 = -9.5367431640625e-7*flagPN3
        self._gr_k_1163 = -0.0022321428571428571*flagPN32
        self._gr_k_1164 = -2.7247837611607143e-7*tmp_init_223
        self._gr_k_1165 = -8.6805555555555556e-5*tmp_init_229
        self._gr_k_1166 = -0.16666666666666667*tmp_init_245
        self._gr_k_1167 = -1.7438616071428571e-5*tmp_init_251
        self._gr_k_1168 = -9.5367431640625e-6*tmp_init_251
        self._gr_k_1169 = -3.9691612051008484e-5*tmp_init_249


    cdef void _compute(self
            , double e=-1
            , double x=-1
            , double z=-1
        ):
        """
        values being computed:
        - h21EccCorrResum
        - h22EccCorrResum
        - h31EccCorrResum
        - h32EccCorrResum
        - h33EccCorrResum
        - h41EccCorrResum
        - h42EccCorrResum
        - h43EccCorrResum
        - h44EccCorrResum
        - h51EccCorrResum
        - h52EccCorrResum
        - h53EccCorrResum
        - h54EccCorrResum
        - h55EccCorrResum
        - h61EccCorrResum
        - h62EccCorrResum
        - h63EccCorrResum
        - h64EccCorrResum
        - h65EccCorrResum
        - h66EccCorrResum
        - h71EccCorrResum
        - h72EccCorrResum
        - h73EccCorrResum
        - h74EccCorrResum
        - h75EccCorrResum
        - h76EccCorrResum
        - h77EccCorrResum
        - h81EccCorrResum
        - h82EccCorrResum
        - h83EccCorrResum
        - h84EccCorrResum
        - h85EccCorrResum
        - h86EccCorrResum
        - h87EccCorrResum
        - h88EccCorrResum
        """

        # internal computations intermediate variables declaration/initialisation
        cdef:
            ccomplex.complex[double] tmp_0 = ccomplex.complex[double](0, z)
            ccomplex.complex[double] tmp_1 = 2.0*tmp_0
            ccomplex.complex[double] tmp_2 = ccomplex.exp(-tmp_1)
            double tmp_3 = cmath.pow(e, 2)
            double tmp_4 = cmath.fabs(tmp_3 - 1)
            double tmp_5 = cmath.pow(tmp_4, 2)
            double tmp_6 = cmath.pow(tmp_5, -1)
            ccomplex.complex[double] tmp_7 = ccomplex.exp(tmp_0)
            ccomplex.complex[double] tmp_8 = 2.0*tmp_7
            ccomplex.complex[double] tmp_9 = ccomplex.exp(tmp_1)
            ccomplex.complex[double] tmp_10 = e*tmp_9
            ccomplex.complex[double] tmp_11 = e + tmp_10 + tmp_8
            ccomplex.complex[double] tmp_12 = ccomplex.pow(tmp_11, <double>2)
            ccomplex.complex[double] tmp_13 = 28.0*tmp_9
            ccomplex.complex[double] tmp_14 = 8.0*tmp_7
            ccomplex.complex[double] tmp_15 = tmp_14*(self._gr_k_245 + tmp_13)
            ccomplex.complex[double] tmp_16 = 4.0*tmp_0
            ccomplex.complex[double] tmp_17 = ccomplex.exp(tmp_16)
            ccomplex.complex[double] tmp_18 = self._gr_k_247*tmp_17
            ccomplex.complex[double] tmp_19 = ccomplex.exp(-tmp_16)
            double tmp_20 = cmath.pow(tmp_4, 3)
            double tmp_21 = cmath.pow(tmp_20, -1)
            ccomplex.complex[double] tmp_22 = tmp_12*tmp_19*tmp_21
            double tmp_23 = e*x
            double tmp_24 = cmath.pow(x, 2)
            double tmp_25 = cmath.pow(tmp_4, 1.5)
            ccomplex.complex[double] tmp_26 = tmp_17*tmp_25
            ccomplex.complex[double] tmp_27 = 3.0*tmp_0
            ccomplex.complex[double] tmp_28 = ccomplex.exp(tmp_27)
            ccomplex.complex[double] tmp_29 = 16.0*tmp_28
            ccomplex.complex[double] tmp_30 = e*tmp_29
            double tmp_31 = 2.0*tmp_3
            ccomplex.complex[double] tmp_32 = tmp_31*tmp_9
            ccomplex.complex[double] tmp_33 = 6.0*tmp_0
            ccomplex.complex[double] tmp_34 = ccomplex.exp(tmp_33)
            double tmp_35 = cmath.pow(e, 3)
            ccomplex.complex[double] tmp_36 = tmp_14*tmp_35
            double tmp_37 = cmath.pow(e, 4)
            ccomplex.complex[double] tmp_38 = ccomplex.exp(8*tmp_0)
            ccomplex.complex[double] tmp_39 = ccomplex.exp(-tmp_33)
            double tmp_40 = cmath.pow(tmp_4, 4)
            double tmp_41 = cmath.pow(tmp_40, -1)
            ccomplex.complex[double] tmp_42 = tmp_39*tmp_41
            ccomplex.complex[double] tmp_43 = tmp_12*tmp_42
            double tmp_44 = 6.2831853071795865
            double tmp_45 = 0.69314718055994531
            double tmp_46 = 4.0*tmp_45
            ccomplex.complex[double] tmp_47 = ccomplex.complex[double](tmp_44, -tmp_46 - 1.0)
            ccomplex.complex[double] tmp_48 = tmp_12*tmp_26
            double tmp_49 = cmath.log(4*x*cmath.exp(-17.0/18.0 + (2.0/3.0)*M_EULER_GAMA))
            ccomplex.complex[double] tmp_50 = ccomplex.complex[double](0, 17280.0)
            ccomplex.complex[double] tmp_51 = ccomplex.pow(tmp_11, <double>3)
            ccomplex.complex[double] tmp_52 = tmp_51*tmp_9
            double tmp_53 = 18.849555921538759
            double tmp_54 = -tmp_46
            ccomplex.complex[double] tmp_55 = 5.0*tmp_0
            ccomplex.complex[double] tmp_56 = ccomplex.exp(tmp_55)
            ccomplex.complex[double] tmp_57 = e*tmp_56
            double tmp_58 = 37.699111843077519
            double tmp_59 = 1.0986122886681097
            double tmp_60 = 81.0*tmp_59
            double tmp_61 = -tmp_60
            ccomplex.complex[double] tmp_62 = ccomplex.complex[double](0, 3.0)
            ccomplex.complex[double] tmp_63 = -tmp_62
            ccomplex.complex[double] tmp_64 = ccomplex.complex[double](0, tmp_45)
            ccomplex.complex[double] tmp_65 = 20.0*tmp_64
            ccomplex.complex[double] tmp_66 = 2.0*tmp_9
            ccomplex.complex[double] tmp_67 = tmp_17*tmp_3
            double tmp_68 = 94.247779607693797
            double tmp_69 = 12.566370614359173
            ccomplex.complex[double] tmp_70 = ccomplex.complex[double](0, -tmp_46)
            ccomplex.complex[double] tmp_71 = 3.0*tmp_17
            double tmp_72 = -162.0*tmp_59
            ccomplex.complex[double] tmp_73 = 9.0*tmp_9
            ccomplex.complex[double] tmp_74 = tmp_28*tmp_35
            ccomplex.complex[double] tmp_75 = 960.0*tmp_74
            ccomplex.complex[double] tmp_76 = 3.1415926535897932*tmp_34
            double tmp_77 = 1.6094379124341004
            double tmp_78 = 15625.0*tmp_77
            double tmp_79 = -tmp_78
            double tmp_80 = -243.0*tmp_59
            double tmp_81 = 6561.0*tmp_59
            ccomplex.complex[double] tmp_82 = 8.0*tmp_9
            double tmp_83 = 40.840704496667312
            double tmp_84 = 2187.0*tmp_59
            double tmp_85 = -tmp_84
            ccomplex.complex[double] tmp_86 = 6.0*tmp_17
            ccomplex.complex[double] tmp_87 = tmp_37*tmp_9
            ccomplex.complex[double] tmp_88 = 120.0*tmp_87
            double tmp_89 = 390625.0*tmp_77
            ccomplex.complex[double] tmp_90 = ccomplex.complex[double](0, 943328.0*tmp_45 - 22842.0*tmp_59 - tmp_89)
            double tmp_91 = 45927.0*tmp_59
            ccomplex.complex[double] tmp_92 = 10.0*tmp_17
            double tmp_93 = 3159.0*tmp_59
            ccomplex.complex[double] tmp_94 = 10.0*tmp_34
            ccomplex.complex[double] tmp_95 = ccomplex.exp(10*tmp_0)
            double tmp_96 = 21.991148575128553
            double tmp_97 = 3645.0*tmp_59
            double tmp_98 = 15.707963267948966
            double tmp_99 = 729.0*tmp_59
            ccomplex.complex[double] tmp_100 = ccomplex.complex[double](0, tmp_99)
            ccomplex.complex[double] tmp_101 = 5.0*tmp_38
            double tmp_102 = 78125.0*tmp_77
            ccomplex.complex[double] tmp_103 = 5.0*tmp_9
            ccomplex.complex[double] tmp_104 = ccomplex.complex[double](0, 1)*tmp_103
            ccomplex.complex[double] tmp_105 = tmp_101*(tmp_100 - 864.0*tmp_64 + tmp_98) + tmp_104*(-tmp_102 + 367456.0*tmp_45 - 118098.0*tmp_59) - tmp_90 + tmp_92*(-74400.0*tmp_64 + ccomplex.complex[double](0, 1)*tmp_91 + 3.1415926535897932) - tmp_94*(-4512.0*tmp_64 + ccomplex.complex[double](0, 1)*tmp_93 + 3.1415926535897932) + tmp_95*(5856.0*tmp_64 + tmp_96 - ccomplex.complex[double](0, 1)*tmp_97)
            double tmp_106 = cmath.pow(e, 5)
            ccomplex.complex[double] tmp_107 = tmp_106*tmp_7
            ccomplex.complex[double] tmp_108 = 24.0*tmp_107
            double tmp_109 = cmath.pow(e, 6)
            double tmp_110 = 59049.0*tmp_59
            double tmp_111 = 1.9459101490553133
            double tmp_112 = -5764801.0*tmp_111
            ccomplex.complex[double] tmp_113 = ccomplex.complex[double](0, tmp_59)
            double tmp_114 = 1328125.0*tmp_77
            ccomplex.complex[double] tmp_115 = 15.0*tmp_17
            ccomplex.complex[double] tmp_116 = ccomplex.exp(12*tmp_0)
            double tmp_117 = 116.23892818282235
            double tmp_118 = 3125.0*tmp_77
            ccomplex.complex[double] tmp_119 = ccomplex.complex[double](0, 25.0)
            ccomplex.complex[double] tmp_120 = tmp_115*(2104137.0*tmp_113 + ccomplex.complex[double](0, 1)*tmp_114 - 6405632.0*tmp_64 + tmp_98) + tmp_116*(tmp_117 - tmp_119*(tmp_118 + tmp_81) + 441856.0*tmp_64) + 160.0*tmp_34*(-ccomplex.complex[double](0, 1)*tmp_110 + 94944.0*tmp_64 + 3.1415926535897932) - 4241.1500823462209*tmp_38 + 144.0*tmp_9*tmp_90 - 48.0*tmp_95*(tmp_62*(4000.0*tmp_45 - 2511.0*tmp_59) + tmp_96) - ccomplex.complex[double](0, 1)*(tmp_112 + 54811648.0*tmp_45 + 15114357.0*tmp_59 - 26953125.0*tmp_77)
            double tmp_121 = cmath.pow(x, 1.5)
            double tmp_122 = cmath.pow(tmp_4, -3.5)
            ccomplex.complex[double] tmp_123 = tmp_122*tmp_39
            ccomplex.complex[double] tmp_124 = tmp_121*tmp_123
            ccomplex.complex[double] tmp_125 = e*tmp_15
            ccomplex.complex[double] tmp_126 = tmp_25*tmp_56
            ccomplex.complex[double] tmp_127 = tmp_9 - 1.0
            ccomplex.complex[double] tmp_128 = tmp_9 + 1.0
            ccomplex.complex[double] tmp_129 = e*tmp_17
            double tmp_130 = 400.0*tmp_3
            double tmp_131 = 600.0*tmp_35
            ccomplex.complex[double] tmp_132 = tmp_37*tmp_7
            ccomplex.complex[double] tmp_133 = tmp_106*tmp_128
            ccomplex.complex[double] tmp_134 = e*tmp_28
            ccomplex.complex[double] tmp_135 = ccomplex.complex[double](0, tmp_25)
            ccomplex.complex[double] tmp_136 = tmp_135*tmp_56
            ccomplex.complex[double] tmp_137 = tmp_3*tmp_8
            ccomplex.complex[double] tmp_138 = 7.0*tmp_0
            ccomplex.complex[double] tmp_139 = ccomplex.exp(tmp_138)
            ccomplex.complex[double] tmp_140 = 80.0*tmp_34
            ccomplex.complex[double] tmp_141 = tmp_3*tmp_56
            ccomplex.complex[double] tmp_142 = tmp_17*tmp_35
            ccomplex.complex[double] tmp_143 = 9.0*tmp_95
            ccomplex.complex[double] tmp_144 = tmp_106*tmp_9
            ccomplex.complex[double] tmp_145 = tmp_28*tmp_37
            double tmp_146 = cmath.pow(e, 7)
            ccomplex.complex[double] tmp_147 = 385.0*tmp_95
            ccomplex.complex[double] tmp_148 = ccomplex.exp(14*tmp_0)
            ccomplex.complex[double] tmp_149 = tmp_109*tmp_7
            double tmp_150 = cmath.pow(e, 8)
            ccomplex.complex[double] tmp_151 = tmp_17 - 1.0
            ccomplex.complex[double] tmp_152 = 28.0*tmp_38 + 28.0
            ccomplex.complex[double] tmp_153 = 140.0*tmp_7
            ccomplex.complex[double] tmp_154 = 120.0*tmp_116 + 120.0
            ccomplex.complex[double] tmp_155 = 2880.0*tmp_67
            ccomplex.complex[double] tmp_156 = 240.0*tmp_9
            ccomplex.complex[double] tmp_157 = tmp_156*tmp_37
            ccomplex.complex[double] tmp_158 = 9.0*tmp_0
            double tmp_159 = cmath.pow(x, 2.5)
            double tmp_160 = cmath.pow(tmp_4, -4.5)
            double tmp_161 = tmp_159*tmp_160
            double tmp_162 = cmath.pow(tmp_4, -1)
            ccomplex.complex[double] tmp_163 = ccomplex.exp(-tmp_0)
            ccomplex.complex[double] tmp_164 = 5.0*tmp_7
            ccomplex.complex[double] tmp_165 = tmp_35*tmp_56
            ccomplex.complex[double] tmp_166 = 24.0*tmp_9
            ccomplex.complex[double] tmp_167 = self._gr_k_3*tmp_34
            ccomplex.complex[double] tmp_168 = ccomplex.exp(-tmp_27)
            ccomplex.complex[double] tmp_169 = tmp_168*tmp_6
            ccomplex.complex[double] tmp_170 = tmp_109*tmp_139
            ccomplex.complex[double] tmp_171 = 16.0*tmp_17
            ccomplex.complex[double] tmp_172 = e*tmp_171
            ccomplex.complex[double] tmp_173 = tmp_29*tmp_3
            ccomplex.complex[double] tmp_174 = tmp_35*tmp_82
            ccomplex.complex[double] tmp_175 = 4.0*tmp_7
            ccomplex.complex[double] tmp_176 = tmp_175*tmp_37
            ccomplex.complex[double] tmp_177 = e + 5.0*tmp_10 + tmp_175 + tmp_28*tmp_31
            ccomplex.complex[double] tmp_178 = ccomplex.exp(-tmp_55)
            ccomplex.complex[double] tmp_179 = tmp_178*tmp_21
            double tmp_180 = cmath.pow(tmp_4, -2.5)
            double tmp_181 = tmp_3*(tmp_31 + 13.0)
            ccomplex.complex[double] tmp_182 = e*(tmp_103 + 3.0) + tmp_14
            ccomplex.complex[double] tmp_183 = tmp_12*tmp_182
            ccomplex.complex[double] tmp_184 = tmp_183*tmp_28
            ccomplex.complex[double] tmp_185 = 8.0*tmp_34
            ccomplex.complex[double] tmp_186 = 122.52211349000194*tmp_17 + 3.1415926535897932*tmp_185 + 103.67255756846318*tmp_9 + 28.274333882308139
            double tmp_187 = 8991.0*tmp_59
            double tmp_188 = tmp_187 - 15198.0*tmp_45
            double tmp_189 = 12914.0*tmp_45
            double tmp_190 = -tmp_99
            ccomplex.complex[double] tmp_191 = tmp_71*(tmp_190 + 826.0*tmp_45)
            double tmp_192 = 2.0*tmp_45
            ccomplex.complex[double] tmp_193 = tmp_192*tmp_34
            ccomplex.complex[double] tmp_194 = tmp_118 - tmp_189 - tmp_191 + tmp_193 + tmp_97
            ccomplex.complex[double] tmp_195 = 240.0*tmp_74
            ccomplex.complex[double] tmp_196 = 12.0*tmp_9
            ccomplex.complex[double] tmp_197 = 7.0*tmp_17
            ccomplex.complex[double] tmp_198 = 3.1415926535897932*tmp_196 + 3.1415926535897932*tmp_197 + 15.707963267948966
            double tmp_199 = 84.0*tmp_45 + tmp_61
            double tmp_200 = 145.0*tmp_45
            ccomplex.complex[double] tmp_201 = tmp_45*tmp_71
            ccomplex.complex[double] tmp_202 = tmp_200 + tmp_201 + tmp_61
            ccomplex.complex[double] tmp_203 = ccomplex.complex[double](0, 2.0)
            double tmp_204 = 60794800.0*tmp_45
            double tmp_205 = 26381781.0*tmp_59
            double tmp_206 = 23281250.0*tmp_77
            double tmp_207 = -tmp_206
            double tmp_208 = 17294403.0*tmp_111
            double tmp_209 = -tmp_208
            double tmp_210 = 149744.0*tmp_45
            double tmp_211 = 92583.0*tmp_59
            ccomplex.complex[double] tmp_212 = 15.0*tmp_38
            double tmp_213 = 160.22122533307946
            double tmp_214 = 1472.0*tmp_45
            double tmp_215 = 972.0*tmp_59
            ccomplex.complex[double] tmp_216 = 6.0*tmp_95
            ccomplex.complex[double] tmp_217 = tmp_115*(9126288.0*tmp_45 + 34506.0*tmp_59 - 3953125.0*tmp_77)
            double tmp_218 = 97.38937226128359
            double tmp_219 = 1377.0*tmp_59
            double tmp_220 = -tmp_219
            ccomplex.complex[double] tmp_221 = ccomplex.complex[double](0, 9.0)
            double tmp_222 = tmp_112 + 48304000.0*tmp_45 + 15557589.0*tmp_59 - 24453125.0*tmp_77
            ccomplex.complex[double] tmp_223 = tmp_62*tmp_9
            double tmp_224 = 4286848.0*tmp_45
            double tmp_225 = 1361043.0*tmp_59
            double tmp_226 = 921875.0*tmp_77
            ccomplex.complex[double] tmp_227 = tmp_116*(tmp_218 + tmp_221*(tmp_220 + 2128.0*tmp_45)) + tmp_212*(ccomplex.complex[double](0, 1)*tmp_210 - ccomplex.complex[double](0, 1)*tmp_211 + tmp_96) + tmp_216*(tmp_213 - ccomplex.complex[double](0, 1)*tmp_214 + ccomplex.complex[double](0, 1)*tmp_215) + ccomplex.complex[double](0, 1)*tmp_217 - tmp_222*tmp_223 + tmp_94*(-ccomplex.complex[double](0, 1)*tmp_224 + ccomplex.complex[double](0, 1)*tmp_225 + ccomplex.complex[double](0, 1)*tmp_226 + tmp_96) + ccomplex.complex[double](0, 1)*(tmp_204 + tmp_205 + tmp_207 + tmp_209)
            double tmp_228 = 35644.0*tmp_45
            double tmp_229 = 405.0*tmp_59
            double tmp_230 = -2673.0*tmp_59
            ccomplex.complex[double] tmp_231 = tmp_86*(tmp_230 + 4384.0*tmp_45)
            double tmp_232 = 1053.0*tmp_59
            ccomplex.complex[double] tmp_233 = tmp_9*(68344.0*tmp_45 - 20412.0*tmp_59 - tmp_78)
            double tmp_234 = -3593750.0*tmp_77
            double tmp_235 = -823543.0*tmp_111
            double tmp_236 = tmp_234 + tmp_235 + 7107328.0*tmp_45 + 2239245.0*tmp_59
            double tmp_237 = 180736.0*tmp_45
            double tmp_238 = 166.50441064025904
            double tmp_239 = 1792.0*tmp_45
            double tmp_240 = 265625.0*tmp_77
            ccomplex.complex[double] tmp_241 = tmp_92*(-tmp_240 + 1207808.0*tmp_45 - 375030.0*tmp_59)
            double tmp_242 = 810.0*tmp_59
            double tmp_243 = tmp_242 + 165632.0*tmp_45 - 71875.0*tmp_77
            ccomplex.complex[double] tmp_244 = -tmp_212*(-tmp_100 + 768.0*ccomplex.complex[double](0, 1)*tmp_45 + tmp_83) - ccomplex.complex[double](0, 1)*tmp_236 - ccomplex.complex[double](0, 1)*tmp_241 + 105.0*ccomplex.complex[double](0, 1)*tmp_243*tmp_9 - tmp_94*(-ccomplex.complex[double](0, 1)*tmp_237 + 111537.0*ccomplex.complex[double](0, 1)*tmp_59 + tmp_96) + tmp_95*(ccomplex.complex[double](0, 1)*tmp_219 + tmp_238 - ccomplex.complex[double](0, 1)*tmp_239)
            ccomplex.complex[double] tmp_245 = 3.0*tmp_107
            ccomplex.complex[double] tmp_246 = 40.840704496667312*tmp_9 + 34.557519189487726
            ccomplex.complex[double] tmp_247 = tmp_109*tmp_227 - tmp_157*(-ccomplex.complex[double](0, 1)*tmp_231 + ccomplex.complex[double](0, 1)*tmp_233 + tmp_34*(-ccomplex.complex[double](0, 1)*tmp_232 + 1288.0*ccomplex.complex[double](0, 1)*tmp_45 + tmp_96) + tmp_38*(tmp_70 + tmp_98) - ccomplex.complex[double](0, 1)*(tmp_228 + tmp_229 + tmp_79)) + tmp_244*tmp_245 + 2880.0*tmp_57*(tmp_246 + 6.0*ccomplex.complex[double](0, 1)*(tmp_45*tmp_9 - 9.0*tmp_45 + 9.0*tmp_59)) + 23040.0*tmp_76
            ccomplex.complex[double] tmp_248 = tmp_121*tmp_39
            ccomplex.complex[double] tmp_249 = 4.0*tmp_10
            ccomplex.complex[double] tmp_250 = tmp_3*tmp_7
            ccomplex.complex[double] tmp_251 = tmp_12*tmp_7
            ccomplex.complex[double] tmp_252 = tmp_106*tmp_34
            ccomplex.complex[double] tmp_253 = 3.0*tmp_9
            ccomplex.complex[double] tmp_254 = tmp_28*(tmp_253 - 1.0)
            ccomplex.complex[double] tmp_255 = tmp_37*tmp_56
            ccomplex.complex[double] tmp_256 = tmp_127*tmp_255
            ccomplex.complex[double] tmp_257 = 16.0*tmp_10
            ccomplex.complex[double] tmp_258 = 14.0*tmp_9
            ccomplex.complex[double] tmp_259 = tmp_250*(-17.0*tmp_17 + tmp_258 + 29.0*tmp_34 + 6.0)
            ccomplex.complex[double] tmp_260 = 252.0*tmp_9
            double tmp_261 = 16.0*tmp_35
            ccomplex.complex[double] tmp_262 = 6.0*tmp_10
            ccomplex.complex[double] tmp_263 = self._gr_k_25*tmp_9 + self._gr_k_26
            ccomplex.complex[double] tmp_264 = self._gr_k_22*tmp_17 + self._gr_k_23*tmp_9 + self._gr_k_24
            ccomplex.complex[double] tmp_265 = self._gr_k_4*tmp_9 + self._gr_k_5*tmp_17 + self._gr_k_6 + tmp_167
            ccomplex.complex[double] tmp_266 = 5.0*tmp_17
            ccomplex.complex[double] tmp_267 = -1843200.0*tmp_34
            double tmp_268 = 3.0*tmp_109
            ccomplex.complex[double] tmp_269 = tmp_146*tmp_7
            ccomplex.complex[double] tmp_270 = ccomplex.exp(tmp_158)
            ccomplex.complex[double] tmp_271 = tmp_150*tmp_270
            ccomplex.complex[double] tmp_272 = e*tmp_34
            ccomplex.complex[double] tmp_273 = tmp_270*tmp_3*(7568.0*tmp_3 + 371.0*tmp_37 + 5568.0)
            ccomplex.complex[double] tmp_274 = tmp_177*tmp_20*tmp_34
            ccomplex.complex[double] tmp_275 = ccomplex.exp(11*tmp_0)
            ccomplex.complex[double] tmp_276 = ccomplex.exp(13*tmp_0)
            ccomplex.complex[double] tmp_277 = 10.0*tmp_7
            ccomplex.complex[double] tmp_278 = 18.0*tmp_9
            ccomplex.complex[double] tmp_279 = -5.0*tmp_17
            ccomplex.complex[double] tmp_280 = tmp_279 + 15.0
            ccomplex.complex[double] tmp_281 = -e*tmp_277*(tmp_9 + 5.0) + tmp_3*(-tmp_278 - tmp_280) - 32.0*tmp_9
            ccomplex.complex[double] tmp_282 = tmp_281*tmp_52
            ccomplex.complex[double] tmp_283 = tmp_181*tmp_270
            ccomplex.complex[double] tmp_284 = ccomplex.complex[double](0, 3.1415926535897932)
            double tmp_285 = -27.0*tmp_59
            double tmp_286 = -tmp_97
            ccomplex.complex[double] tmp_287 = ccomplex.complex[double](0, tmp_98)
            ccomplex.complex[double] tmp_288 = ccomplex.complex[double](0, tmp_96)
            ccomplex.complex[double] tmp_289 = 105.0*tmp_9
            ccomplex.complex[double] tmp_290 = tmp_106*tmp_253
            double tmp_291 = -tmp_226
            ccomplex.complex[double] tmp_292 = 105.0*tmp_284
            double tmp_293 = 17010.0*tmp_59
            double tmp_294 = 630.0*tmp_45
            double tmp_295 = tmp_294 + 3103.0
            ccomplex.complex[double] tmp_296 = 192.0*tmp_272
            double tmp_297 = 51030.0*tmp_59
            double tmp_298 = tmp_294 + 1819.0
            double tmp_299 = 8610.0*tmp_45
            double tmp_300 = 8505.0*tmp_59
            ccomplex.complex[double] tmp_301 = 6.0*tmp_9
            ccomplex.complex[double] tmp_302 = 192.0*tmp_141
            double tmp_303 = 560.0*tmp_45
            double tmp_304 = 636160.0*tmp_45 - 348705.0*tmp_59 + 10272.0
            ccomplex.complex[double] tmp_305 = 840.0*tmp_284
            double tmp_306 = 11600.0*tmp_45
            double tmp_307 = 10935.0*tmp_59
            double tmp_308 = -tmp_307
            ccomplex.complex[double] tmp_309 = 21.0*tmp_17
            double tmp_310 = -1789760.0*tmp_45 + 382725.0*tmp_59 + 546875.0*tmp_77 + 13696.0
            ccomplex.complex[double] tmp_311 = 48.0*tmp_142
            double tmp_312 = 2734375.0*tmp_77
            ccomplex.complex[double] tmp_313 = 70.0*tmp_284
            double tmp_314 = 331695.0*tmp_59
            double tmp_315 = -tmp_314 + 566930.0*tmp_45 + 2033.0
            double tmp_316 = -1432000.0*tmp_45 + 349920.0*tmp_59 + tmp_89 + 1712.0
            ccomplex.complex[double] tmp_317 = 7.0*tmp_9
            ccomplex.complex[double] tmp_318 = 48.0*tmp_145
            double tmp_319 = 46484375.0*tmp_77
            double tmp_320 = tmp_319 - 186945080.0*tmp_45 + 50910930.0*tmp_59 + 20116.0
            ccomplex.complex[double] tmp_321 = 2.0*tmp_17
            double tmp_322 = 76545.0*tmp_59
            double tmp_323 = -tmp_322
            ccomplex.complex[double] tmp_324 = 3.0*tmp_38
            double tmp_325 = 26841640.0*tmp_45 + 2447010.0*tmp_59 - 13203125.0*tmp_77 + 1284.0
            ccomplex.complex[double] tmp_326 = 21.0*tmp_9
            double tmp_327 = 40353607.0*tmp_111
            double tmp_328 = tmp_327 - 237535032.0*tmp_45 - 105810705.0*tmp_59 + 125781250.0*tmp_77 + 6420.0
            double tmp_329 = 60960.0*tmp_45
            double tmp_330 = 37179.0*tmp_59
            double tmp_331 = -121060821.0*tmp_111
            double tmp_332 = 9477.0*tmp_59
            double tmp_333 = 4609375.0*tmp_77
            double tmp_334 = tmp_333 - 19910720.0*tmp_45 + 5867721.0*tmp_59
            double tmp_335 = 45528544.0*tmp_45 + 3007854.0*tmp_59 - 21640625.0*tmp_77
            ccomplex.complex[double] tmp_336 = tmp_115*tmp_335
            double tmp_337 = 353565.0*tmp_59
            double tmp_338 = 116406250.0*tmp_77
            double tmp_339 = tmp_331 - tmp_338 + 349936352.0*tmp_45 + 164175903.0*tmp_59
            double tmp_340 = 9.8696044010893586
            double tmp_341 = 14700.0*tmp_340 - 515063.0
            double tmp_342 = cmath.pow(tmp_45, 2)
            double tmp_343 = 529200.0*tmp_342 - 539280.0*tmp_45
            double tmp_344 = 2430.0*tmp_59
            double tmp_345 = cmath.pow(tmp_59, 2)
            double tmp_346 = 14288400.0*tmp_345 + 14560560.0*tmp_59
            double tmp_347 = 279300.0*tmp_340 - 9786197.0
            double tmp_348 = 48.0*tmp_3
            double tmp_349 = tmp_45*tmp_59
            double tmp_350 = tmp_45*tmp_77
            double tmp_351 = cmath.pow(tmp_77, 2)
            double tmp_352 = 59920.0*tmp_45
            double tmp_353 = 210.0*tmp_45
            ccomplex.complex[double] tmp_354 = 35.0*tmp_284
            double tmp_355 = 515063.0 - 14700.0*tmp_340
            ccomplex.complex[double] tmp_356 = 96.0*tmp_74
            double tmp_357 = -tmp_102
            double tmp_358 = tmp_111*tmp_45
            double tmp_359 = cmath.pow(tmp_111, 2)
            ccomplex.complex[double] tmp_360 = 2.0*tmp_34
            double tmp_361 = 229635.0*tmp_59
            double tmp_362 = 433755.0*tmp_59
            double tmp_363 = 140.0*tmp_45
            ccomplex.complex[double] tmp_364 = 210.0*tmp_284
            double tmp_365 = 105.0*tmp_59 + 107.0
            double tmp_366 = 10.0*tmp_45
            double tmp_367 = -tmp_327
            double tmp_368 = tmp_367 + 249531456.0*tmp_45 + 106864839.0*tmp_59 - 131640625.0*tmp_77
            ccomplex.complex[double] tmp_369 = ccomplex.exp(-tmp_138)
            ccomplex.complex[double] tmp_370 = tmp_369*tmp_41
            ccomplex.complex[double] tmp_371 = 24.0*tmp_74
            ccomplex.complex[double] tmp_372 = 34.0*tmp_9
            ccomplex.complex[double] tmp_373 = tmp_266 + tmp_372
            ccomplex.complex[double] tmp_374 = -4.0*e*tmp_7*(tmp_9 - 5.0) + tmp_3*(tmp_373 + 5.0) + tmp_371 - 4.0*tmp_9
            ccomplex.complex[double] tmp_375 = tmp_2/tmp_25
            ccomplex.complex[double] tmp_376 = 64.0*tmp_28
            ccomplex.complex[double] tmp_377 = self._gr_k_277*tmp_34
            ccomplex.complex[double] tmp_378 = tmp_175*tmp_3
            ccomplex.complex[double] tmp_379 = tmp_180*tmp_19
            ccomplex.complex[double] tmp_380 = tmp_23*tmp_379
            ccomplex.complex[double] tmp_381 = tmp_139*tmp_146
            ccomplex.complex[double] tmp_382 = 256.0*tmp_56
            ccomplex.complex[double] tmp_383 = e*tmp_382
            ccomplex.complex[double] tmp_384 = 64.0*tmp_67
            ccomplex.complex[double] tmp_385 = 64.0*tmp_74
            ccomplex.complex[double] tmp_386 = 4.0*tmp_9
            ccomplex.complex[double] tmp_387 = tmp_37*tmp_386
            ccomplex.complex[double] tmp_388 = tmp_106*tmp_175
            ccomplex.complex[double] tmp_389 = tmp_26*tmp_374
            ccomplex.complex[double] tmp_390 = tmp_123*tmp_24
            ccomplex.complex[double] tmp_391 = e*tmp_139
            double tmp_392 = tmp_366 + 7.0
            ccomplex.complex[double] tmp_393 = ccomplex.complex[double](0, tmp_392)
            ccomplex.complex[double] tmp_394 = -tmp_393 + tmp_98
            ccomplex.complex[double] tmp_395 = e*tmp_7
            ccomplex.complex[double] tmp_396 = 40.0*tmp_395
            ccomplex.complex[double] tmp_397 = tmp_3*(-tmp_280 - tmp_372) + tmp_386 - tmp_396
            ccomplex.complex[double] tmp_398 = tmp_12*tmp_9
            ccomplex.complex[double] tmp_399 = tmp_397*tmp_398
            double tmp_400 = 141.3716694115407
            ccomplex.complex[double] tmp_401 = ccomplex.complex[double](0, tmp_366 + 63.0)
            ccomplex.complex[double] tmp_402 = tmp_394*tmp_9
            double tmp_403 = -1215.0*tmp_59
            ccomplex.complex[double] tmp_404 = 1152.0*tmp_67
            ccomplex.complex[double] tmp_405 = ccomplex.complex[double](0, 5.0)
            ccomplex.complex[double] tmp_406 = 30.0*tmp_64
            ccomplex.complex[double] tmp_407 = ccomplex.complex[double](0, 21.0)
            ccomplex.complex[double] tmp_408 = 5.0*tmp_34
            double tmp_409 = 361.28315516282622
            ccomplex.complex[double] tmp_410 = 384.0*tmp_74
            double tmp_411 = 2827.4333882308139
            double tmp_412 = -65610.0*tmp_59
            double tmp_413 = 549.77871437821382
            ccomplex.complex[double] tmp_414 = ccomplex.complex[double](0, 12.0)
            double tmp_415 = 172.78759594743863
            double tmp_416 = 32805.0*tmp_59
            double tmp_417 = -tmp_416
            ccomplex.complex[double] tmp_418 = tmp_166*tmp_37
            double tmp_419 = -1484375.0*tmp_77
            double tmp_420 = 25515.0*tmp_59
            double tmp_421 = 365958.0*tmp_59
            double tmp_422 = 34.557519189487726
            ccomplex.complex[double] tmp_423 = -tmp_101*(5103.0*tmp_113 - 7104.0*tmp_64 + 436.68137884898126) - tmp_104*(-tmp_421 + 1263488.0*tmp_45 - 296875.0*tmp_77) - tmp_92*(-tmp_221*(26368.0*tmp_45 - 15957.0*tmp_59) + tmp_422) - tmp_94*(tmp_221*(-tmp_232 + 1216.0*tmp_45) + 493.23004661359754) + tmp_95*(ccomplex.complex[double](0, 1)*tmp_420 - 40704.0*tmp_64 + tmp_96) + ccomplex.complex[double](0, 1)*(tmp_419 + 3333952.0*tmp_45 + 71442.0*tmp_59)
            double tmp_424 = 235.61944901923449
            ccomplex.complex[double] tmp_425 = 8192.0*tmp_64
            ccomplex.complex[double] tmp_426 = 90.0*tmp_38
            ccomplex.complex[double] tmp_427 = tmp_21*tmp_248
            ccomplex.complex[double] tmp_428 = tmp_106*tmp_275
            ccomplex.complex[double] tmp_429 = tmp_109*tmp_116
            ccomplex.complex[double] tmp_430 = tmp_37*tmp_95
            ccomplex.complex[double] tmp_431 = tmp_270*tmp_35
            ccomplex.complex[double] tmp_432 = tmp_3*tmp_38
            ccomplex.complex[double] tmp_433 = tmp_106*tmp_56
            ccomplex.complex[double] tmp_434 = self._gr_k_270*tmp_9
            ccomplex.complex[double] tmp_435 = self._gr_k_271 + tmp_434
            ccomplex.complex[double] tmp_436 = tmp_3*(self._gr_k_273*tmp_17 + self._gr_k_274*tmp_9 + self._gr_k_275)
            ccomplex.complex[double] tmp_437 = self._gr_k_276*tmp_9 + self._gr_k_278*tmp_17 + self._gr_k_279 + tmp_377
            ccomplex.complex[double] tmp_438 = tmp_175*tmp_35
            ccomplex.complex[double] tmp_439 = tmp_37*(self._gr_k_263*tmp_9 + self._gr_k_264*tmp_34 + self._gr_k_265*tmp_38 + self._gr_k_266*tmp_17 + self._gr_k_267)
            ccomplex.complex[double] tmp_440 = tmp_146*tmp_56
            ccomplex.complex[double] tmp_441 = tmp_109*tmp_34
            ccomplex.complex[double] tmp_442 = tmp_3*tmp_386
            ccomplex.complex[double] tmp_443 = tmp_35*tmp_7
            ccomplex.complex[double] tmp_444 = tmp_34*tmp_40
            ccomplex.complex[double] tmp_445 = tmp_106*tmp_139
            ccomplex.complex[double] tmp_446 = 16.0*tmp_57
            ccomplex.complex[double] tmp_447 = 4.0*tmp_17
            ccomplex.complex[double] tmp_448 = 4.0*tmp_74
            ccomplex.complex[double] tmp_449 = 64.0*tmp_56
            ccomplex.complex[double] tmp_450 = e*tmp_449
            ccomplex.complex[double] tmp_451 = tmp_171*tmp_3
            ccomplex.complex[double] tmp_452 = 20.0*tmp_395
            ccomplex.complex[double] tmp_453 = tmp_12*tmp_122*tmp_178
            ccomplex.complex[double] tmp_454 = tmp_28*tmp_3
            ccomplex.complex[double] tmp_455 = tmp_166*tmp_35
            ccomplex.complex[double] tmp_456 = tmp_12*tmp_160*tmp_369
            ccomplex.complex[double] tmp_457 = tmp_184*tmp_25
            ccomplex.complex[double] tmp_458 = 224.0*tmp_9
            ccomplex.complex[double] tmp_459 = tmp_51*tmp_7
            double tmp_460 = 358.14156250923643
            double tmp_461 = -tmp_54 - 9.0
            ccomplex.complex[double] tmp_462 = 15.0*tmp_9
            double tmp_463 = -tmp_114
            double tmp_464 = 204.20352248333656
            ccomplex.complex[double] tmp_465 = tmp_248*tmp_41
            ccomplex.complex[double] tmp_466 = 36.0*tmp_9
            ccomplex.complex[double] tmp_467 = 35.0*tmp_17
            ccomplex.complex[double] tmp_468 = 8.0*tmp_165 + tmp_3*(tmp_258 + tmp_467 + 3.0) + tmp_452*(tmp_253 + 1.0) + tmp_466
            ccomplex.complex[double] tmp_469 = self._gr_k_373*tmp_38
            double tmp_470 = 3.0*tmp_35
            ccomplex.complex[double] tmp_471 = tmp_146*tmp_270
            ccomplex.complex[double] tmp_472 = tmp_26*tmp_468
            ccomplex.complex[double] tmp_473 = ccomplex.complex[double](tmp_98, -tmp_392 + 10.0*tmp_59)
            ccomplex.complex[double] tmp_474 = tmp_3*(tmp_467 + 42.0*tmp_9 + 15.0) + tmp_396*(tmp_253 + 2.0) + 108.0*tmp_9
            ccomplex.complex[double] tmp_475 = tmp_398*tmp_474
            ccomplex.complex[double] tmp_476 = 41472.0*tmp_34
            double tmp_477 = 738.27427359360141
            ccomplex.complex[double] tmp_478 = 1536.0*tmp_57
            ccomplex.complex[double] tmp_479 = 384.0*tmp_67
            double tmp_480 = 47385.0*tmp_59
            double tmp_481 = -tmp_480
            ccomplex.complex[double] tmp_482 = tmp_37*tmp_82
            double tmp_483 = -28824005.0*tmp_111
            ccomplex.complex[double] tmp_484 = ccomplex.complex[double](0, tmp_84)
            ccomplex.complex[double] tmp_485 = ccomplex.complex[double](0, 1)*tmp_94
            ccomplex.complex[double] tmp_486 = tmp_101*(tmp_424 - 23.0*ccomplex.complex[double](0, 1)*(11776.0*tmp_45 - 7047.0*tmp_59)) + tmp_104*(tmp_112 + 30295232.0*tmp_45 + 15126507.0*tmp_59 - 16406250.0*tmp_77) + tmp_485*(-tmp_226 + 3685248.0*tmp_45 - 992898.0*tmp_59) - ccomplex.complex[double](0, 1)*tmp_92*(13367488.0*tmp_45 + 1538190.0*tmp_59 - 6796875.0*tmp_77) + tmp_95*(tmp_484 - 2752.0*tmp_64 + 1435.7078426905355) - ccomplex.complex[double](0, 1)*(tmp_207 + 76115584.0*tmp_45 + tmp_483 + 37137447.0*tmp_59)
            double tmp_487 = -334765625.0*tmp_77
            double tmp_488 = 9765625.0*tmp_77
            ccomplex.complex[double] tmp_489 = tmp_101*(tmp_424 + ccomplex.complex[double](0, 1)*(42178560.0*tmp_45 - tmp_488 - 12428721.0*tmp_59)) + 2.0*tmp_116*(-6642.0*tmp_113 + tmp_425 + 644.02649398590761) - ccomplex.complex[double](0, 1)*tmp_140*(21889024.0*tmp_45 + 1746198.0*tmp_59 - 10609375.0*tmp_77) + ccomplex.complex[double](0, 1)*tmp_266*(-144120025.0*tmp_111 + 814882816.0*tmp_45 + 380979045.0*tmp_59 - 436815625.0*tmp_77) - ccomplex.complex[double](0, 1)*tmp_82*(-233062669.0*tmp_111 + 630246656.0*tmp_45 + 303817311.0*tmp_59 - 197031250.0*tmp_77) + 8.0*tmp_95*(102789.0*tmp_113 - 171776.0*tmp_64 + 1077.5662801812991) + ccomplex.complex[double](0, 1)*(-893544155.0*tmp_111 + 2509146112.0*tmp_45 + tmp_487 + 490012497.0*tmp_59)
            ccomplex.complex[double] tmp_490 = tmp_106*tmp_270
            double tmp_491 = cmath.sqrt(tmp_4)
            ccomplex.complex[double] tmp_492 = 32.0*tmp_28
            ccomplex.complex[double] tmp_493 = e*tmp_492
            double tmp_494 = 3.0*tmp_37
            ccomplex.complex[double] tmp_495 = tmp_3*tmp_82
            ccomplex.complex[double] tmp_496 = tmp_35*tmp_9
            ccomplex.complex[double] tmp_497 = 40.0*tmp_9
            ccomplex.complex[double] tmp_498 = tmp_106*tmp_14
            ccomplex.complex[double] tmp_499 = tmp_14*tmp_3
            ccomplex.complex[double] tmp_500 = tmp_23*tmp_43
            ccomplex.complex[double] tmp_501 = tmp_25*tmp_399
            double tmp_502 = 47.123889803846899
            double tmp_503 = 15.0*tmp_45
            double tmp_504 = tmp_503 + 16.0
            ccomplex.complex[double] tmp_505 = -tmp_203*tmp_504 + tmp_502
            double tmp_506 = 6691.5923521462596
            ccomplex.complex[double] tmp_507 = 420.0*tmp_64
            double tmp_508 = 984150.0*tmp_59
            ccomplex.complex[double] tmp_509 = ccomplex.complex[double](0, 10.0)
            ccomplex.complex[double] tmp_510 = 80.0*tmp_9
            ccomplex.complex[double] tmp_511 = 12.0*tmp_95
            double tmp_512 = 688905.0*tmp_59
            ccomplex.complex[double] tmp_513 = tmp_160*tmp_248
            double tmp_514 = cmath.sqrt(x)
            double tmp_515 = tmp_180*tmp_514
            ccomplex.complex[double] tmp_516 = 19.0*tmp_9
            ccomplex.complex[double] tmp_517 = tmp_249*(tmp_317 - 8.0) - 24.0*tmp_255 + tmp_29 + tmp_378*(4.0*tmp_17 - tmp_516 - 5.0) - tmp_470*(tmp_301 + tmp_309 + 1.0)
            ccomplex.complex[double] tmp_518 = tmp_179*tmp_23
            double tmp_519 = tmp_25*tmp_49
            ccomplex.complex[double] tmp_520 = tmp_106*tmp_196
            double tmp_521 = 9.0*tmp_146
            ccomplex.complex[double] tmp_522 = -tmp_517
            ccomplex.complex[double] tmp_523 = tmp_24*tmp_370
            ccomplex.complex[double] tmp_524 = tmp_25*tmp_28*tmp_522
            ccomplex.complex[double] tmp_525 = -tmp_249*(tmp_317 - 24.0) + tmp_470*(tmp_278 + tmp_309 + 5.0) - tmp_492 + tmp_499*(tmp_516 + 10.0)
            ccomplex.complex[double] tmp_526 = 31.415926535897932 - tmp_407
            ccomplex.complex[double] tmp_527 = 2304.0*tmp_34
            ccomplex.complex[double] tmp_528 = ccomplex.complex[double](tmp_68, -tmp_54 - 63.0)
            ccomplex.complex[double] tmp_529 = 288.0*tmp_57
            ccomplex.complex[double] tmp_530 = 288.0*tmp_67
            double tmp_531 = -62500.0*tmp_77
            double tmp_532 = -312500.0*tmp_77
            double tmp_533 = 19683.0*tmp_59
            double tmp_534 = 11640625.0*tmp_77
            ccomplex.complex[double] tmp_535 = 30.0*tmp_38
            ccomplex.complex[double] tmp_536 = 128.0*tmp_28
            ccomplex.complex[double] tmp_537 = tmp_25*tmp_475
            ccomplex.complex[double] tmp_538 = 12.0*tmp_10
            double tmp_539 = 47500.880922277674
            double tmp_540 = -tmp_89
            ccomplex.complex[double] tmp_541 = 45.0*tmp_17
            double tmp_542 = 10781250.0*tmp_77
            double tmp_543 = 201768035.0*tmp_111
            double tmp_544 = -tmp_333
            double tmp_545 = 1673828125.0*tmp_77
            ccomplex.complex[double] tmp_546 = tmp_17*tmp_37
            ccomplex.complex[double] tmp_547 = 512.0*tmp_28
            ccomplex.complex[double] tmp_548 = 48.0*tmp_139
            ccomplex.complex[double] tmp_549 = tmp_249*(269.0*tmp_9 + 113.0)
            ccomplex.complex[double] tmp_550 = 215.0*tmp_17
            ccomplex.complex[double] tmp_551 = tmp_550 + 35.0
            ccomplex.complex[double] tmp_552 = 105.0*tmp_34
            ccomplex.complex[double] tmp_553 = 63.0*tmp_17 + tmp_552 + 27.0*tmp_9 + 5.0
            ccomplex.complex[double] tmp_554 = tmp_37*tmp_548 + tmp_378*(tmp_551 + 142.0*tmp_9) + tmp_470*tmp_553 + tmp_547 + tmp_549
            ccomplex.complex[double] tmp_555 = tmp_150*tmp_275
            ccomplex.complex[double] tmp_556 = tmp_275*tmp_37
            ccomplex.complex[double] tmp_557 = tmp_109*tmp_38
            ccomplex.complex[double] tmp_558 = tmp_3*tmp_9
            double tmp_559 = 3.0*tmp_106
            ccomplex.complex[double] tmp_560 = tmp_28*tmp_491*(tmp_109*tmp_548 + tmp_35*(887.0*tmp_17 - 315.0*tmp_34 + 371.0*tmp_9 - 15.0) + tmp_37*(tmp_153 + 568.0*tmp_28 - tmp_548 + 860.0*tmp_56) - tmp_378*(tmp_258 + tmp_551) - tmp_547 - tmp_549 + tmp_553*tmp_559)
            ccomplex.complex[double] tmp_561 = tmp_249*(807.0*tmp_9 + 565.0) + 2048.0*tmp_28 + tmp_470*(189.0*tmp_17 + tmp_552 + 135.0*tmp_9 + 35.0) + tmp_499*(tmp_550 + 284.0*tmp_9 + 105.0)
            double tmp_562 = -477495.0*tmp_59
            ccomplex.complex[double] tmp_563 = ccomplex.complex[double](0, 405.0)
            double tmp_564 = 3428125.0*tmp_77
            double tmp_565 = -48828125.0*tmp_77
            double tmp_566 = -720600125.0*tmp_111
            double tmp_567 = tmp_21*tmp_514
            ccomplex.complex[double] tmp_568 = 186.0*tmp_9
            ccomplex.complex[double] tmp_569 = 1074.0*tmp_17
            ccomplex.complex[double] tmp_570 = tmp_171 + tmp_30*(tmp_9 - 49.0) + 1920.0*tmp_433 + tmp_438*(-239.0*tmp_17 + tmp_552 + 791.0*tmp_9 + 175.0) + tmp_494*(tmp_260 + 252.0*tmp_34 + 35.0*tmp_38 + tmp_569 + 35.0) - tmp_495*(tmp_197 + tmp_568 - 133.0)
            ccomplex.complex[double] tmp_571 = 768.0*tmp_56
            ccomplex.complex[double] tmp_572 = tmp_123*tmp_23
            ccomplex.complex[double] tmp_573 = tmp_25*tmp_9
            ccomplex.complex[double] tmp_574 = tmp_570*tmp_573
            ccomplex.complex[double] tmp_575 = 4.0*tmp_34
            ccomplex.complex[double] tmp_576 = 64.0*tmp_34
            ccomplex.complex[double] tmp_577 = tmp_3*tmp_376
            ccomplex.complex[double] tmp_578 = tmp_35*tmp_386
            double tmp_579 = 219.91148575128553
            double tmp_580 = tmp_363 + 181.0
            ccomplex.complex[double] tmp_581 = ccomplex.complex[double](tmp_579, -tmp_580)
            ccomplex.complex[double] tmp_582 = 2304.0*tmp_56
            ccomplex.complex[double] tmp_583 = ccomplex.complex[double](0, 35.0)
            ccomplex.complex[double] tmp_584 = 7.0*tmp_34
            ccomplex.complex[double] tmp_585 = 384.0*tmp_454
            ccomplex.complex[double] tmp_586 = ccomplex.complex[double](0, 7.0)
            ccomplex.complex[double] tmp_587 = 40.0*tmp_34
            double tmp_588 = 317115.0*tmp_59
            double tmp_589 = -tmp_588
            ccomplex.complex[double] tmp_590 = 7.0*tmp_38
            ccomplex.complex[double] tmp_591 = 7.0*tmp_95
            ccomplex.complex[double] tmp_592 = 84.0*tmp_95
            ccomplex.complex[double] tmp_593 = 84.0*tmp_9
            ccomplex.complex[double] tmp_594 = 7.0*tmp_116
            ccomplex.complex[double] tmp_595 = tmp_3*tmp_497
            ccomplex.complex[double] tmp_596 = tmp_23*tmp_456
            ccomplex.complex[double] tmp_597 = 79.0*tmp_9
            ccomplex.complex[double] tmp_598 = 112.0*tmp_134
            ccomplex.complex[double] tmp_599 = -959.0*tmp_17
            ccomplex.complex[double] tmp_600 = 2604.0*tmp_34
            ccomplex.complex[double] tmp_601 = -105.0*tmp_38
            ccomplex.complex[double] tmp_602 = 3888.0*tmp_17 + tmp_438*(-3647.0*tmp_17 + 745.0*tmp_34 - 1897.0*tmp_9 - 385.0) - 1920.0*tmp_445 + tmp_494*(-1098.0*tmp_17 - tmp_600 - tmp_601 - 364.0*tmp_9 - 55.0) + tmp_495*(-tmp_599 - 1518.0*tmp_9 - 581.0) + tmp_598*(tmp_597 - 31.0)
            ccomplex.complex[double] tmp_603 = tmp_109*tmp_270
            ccomplex.complex[double] tmp_604 = tmp_573*tmp_602
            ccomplex.complex[double] tmp_605 = tmp_146*tmp_34
            ccomplex.complex[double] tmp_606 = -252.0*tmp_34
            double tmp_607 = -490008085.0*tmp_111
            ccomplex.complex[double] tmp_608 = tmp_14*tmp_37
            ccomplex.complex[double] tmp_609 = tmp_433*tmp_514
            ccomplex.complex[double] tmp_610 = 475.0*tmp_34
            ccomplex.complex[double] tmp_611 = 10000.0*tmp_17 + 28.0*tmp_443*(467.0*tmp_17 + tmp_610 + 229.0*tmp_9 + 45.0) + 384.0*tmp_490 + tmp_494*(594.0*tmp_17 + 924.0*tmp_34 + 1155.0*tmp_38 + 220.0*tmp_9 + 35.0) + tmp_495*(3241.0*tmp_17 + 2694.0*tmp_9 + 725.0) + tmp_598*(227.0*tmp_9 + 109.0)
            double tmp_612 = 15.0*tmp_106
            ccomplex.complex[double] tmp_613 = tmp_573*tmp_611
            ccomplex.complex[double] tmp_614 = 224.0*tmp_134
            ccomplex.complex[double] tmp_615 = 56.0*tmp_443
            ccomplex.complex[double] tmp_616 = tmp_12*(50000.0*tmp_17 + tmp_37*(8910.0*tmp_17 + 8316.0*tmp_34 + 3465.0*tmp_38 + 4620.0*tmp_9 + 945.0) + tmp_495*(9723.0*tmp_17 + 13470.0*tmp_9 + 5075.0) + tmp_614*(454.0*tmp_9 + 327.0) + tmp_615*(934.0*tmp_17 + tmp_610 + 687.0*tmp_9 + 180.0))
            double tmp_617 = -2187500.0*tmp_77
            double tmp_618 = tmp_122*tmp_514
            ccomplex.complex[double] tmp_619 = 123.0*tmp_9
            ccomplex.complex[double] tmp_620 = tmp_28*tmp_348
            ccomplex.complex[double] tmp_621 = 4335.0*tmp_17
            ccomplex.complex[double] tmp_622 = 10554.0*tmp_34
            ccomplex.complex[double] tmp_623 = tmp_35*tmp_492
            ccomplex.complex[double] tmp_624 = tmp_23*tmp_370
            ccomplex.complex[double] tmp_625 = 11281.0*tmp_9
            ccomplex.complex[double] tmp_626 = -4193.0*tmp_17
            ccomplex.complex[double] tmp_627 = -14245.0*tmp_34
            ccomplex.complex[double] tmp_628 = 4389.0*tmp_38
            ccomplex.complex[double] tmp_629 = tmp_146*tmp_275
            double tmp_630 = 15.0*tmp_109
            ccomplex.complex[double] tmp_631 = 57673.0*tmp_17
            ccomplex.complex[double] tmp_632 = 56665.0*tmp_38
            ccomplex.complex[double] tmp_633 = 3003.0*tmp_95
            ccomplex.complex[double] tmp_634 = tmp_146*tmp_276
            ccomplex.complex[double] tmp_635 = tmp_17*tmp_348
            double tmp_636 = 45.0*tmp_109
            double tmp_637 = tmp_41*tmp_514
            ccomplex.complex[double] tmp_638 = 36.0*tmp_107
            ccomplex.complex[double] tmp_639 = 48.0*tmp_129
            ccomplex.complex[double] tmp_640 = -40845.0*tmp_95
            ccomplex.complex[double] tmp_641 = -3003.0*tmp_116
            ccomplex.complex[double] tmp_642 = e*tmp_576
            ccomplex.complex[double] tmp_643 = tmp_3*tmp_449
            ccomplex.complex[double] tmp_644 = tmp_171*tmp_35
            ccomplex.complex[double] tmp_645 = tmp_37*tmp_536
            ccomplex.complex[double] tmp_646 = tmp_106*tmp_386
            ccomplex.complex[double] tmp_647 = 36.0*tmp_149
            double tmp_648 = 45.0*tmp_146
            double tmp_649 = 315.0*tmp_146

        # 21/math.m: hFactEccCorrResumExceptPAv5Flag[2,1]
        self.h21EccCorrResum = self._gr_k_1163*tmp_22*tmp_23*(e*(self._gr_k_956*tmp_9 + self._gr_k_957 + tmp_18) - tmp_15) + self._gr_k_254*tmp_161*(self._gr_k_251*tmp_28*(self._gr_k_211*tmp_120*tmp_150 + self._gr_k_229*tmp_105*tmp_146*tmp_7 + self._gr_k_230*tmp_34 - 48.0*tmp_107*(self._gr_k_212*tmp_38 + self._gr_k_224*tmp_9 + self._gr_k_225*tmp_17 + self._gr_k_226*tmp_34 + self._gr_k_227*tmp_95 + self._gr_k_228) - 2.0*tmp_109*(self._gr_k_217*tmp_95 + self._gr_k_218*tmp_9 + self._gr_k_219*tmp_34 + self._gr_k_220*tmp_38 + self._gr_k_221*tmp_17 + self._gr_k_222*tmp_116 + self._gr_k_223) - tmp_155*(self._gr_k_0*tmp_9 + self._gr_k_1*tmp_17 + self._gr_k_2) - tmp_157*(self._gr_k_13*tmp_34 + self._gr_k_14*tmp_17 + self._gr_k_15*tmp_9 + self._gr_k_16*tmp_38 + self._gr_k_17) - 23040.0*tmp_57*(self._gr_k_18*tmp_9 + self._gr_k_19) - tmp_75*(self._gr_k_213*tmp_17 + self._gr_k_214*tmp_9 + self._gr_k_215*tmp_34 + self._gr_k_216)) + self._gr_k_252*tmp_12*(8.0*cmath.pow(e, 11)*tmp_127*(tmp_154 + 10004.0*tmp_17 + 129984.0*tmp_34 + 10004.0*tmp_38 + 1989.0*tmp_9 + 1989.0*tmp_95) - 70.0*cmath.pow(e, 10)*tmp_151*tmp_7*(tmp_152 + 4852.0*tmp_17 + 417.0*tmp_34 + 417.0*tmp_9) - 16.0*cmath.pow(e, 9)*tmp_127*(tmp_154 + 7932.0*tmp_17 + 84372.0*tmp_34 + 7932.0*tmp_38 + 1737.0*tmp_9 + 1737.0*tmp_95) + e*tmp_140*(4634.0*tmp_9 + 5671.0) + 210000.0*tmp_139 + 40.0*tmp_141*(682.0*tmp_17 + 9945.0*tmp_9 + 19703.0) + 20.0*tmp_142*(7874.0*tmp_17 + 3163.0*tmp_34 + 11431.0*tmp_9 + 17192.0) + 448.0*tmp_144*(tmp_143 + 50.0*tmp_38 - 50.0*tmp_9 - 9.0) + 15.0*tmp_145*(1380.0*tmp_17 - 2206.0*tmp_34 + 415.0*tmp_38 + 8686.0*tmp_9 + 4685.0) + 24.0*tmp_146*(287.0*tmp_116 + tmp_147 + 40.0*tmp_148 - 385.0*tmp_17 - 287.0*tmp_9 - 40.0) - 70.0*tmp_149*(28.0*tmp_116 - 120.0*tmp_17 + 120.0*tmp_38 - 177.0*tmp_9 + 177.0*tmp_95 - 28.0) + tmp_150*tmp_151*tmp_153*(tmp_152 + 3384.0*tmp_17 + 357.0*tmp_34 + 357.0*tmp_9)) + self._gr_k_253*tmp_127*tmp_134*tmp_40*(7537600.0*tmp_128*tmp_129 - tmp_128*tmp_131*tmp_9*(-1065.0*tmp_17 - 115007.0*tmp_9 - 1065.0) - tmp_130*tmp_28*(-9535.0*tmp_17 - 131527.0*tmp_9 - 9535.0) - 30.0*tmp_132*(-9160088.0*tmp_17 - 1133558.0*tmp_34 - 13.0*tmp_38 - 1133558.0*tmp_9 - 13.0) + tmp_133*(321428808.0*tmp_17 + 5687282.0*tmp_34 + 33.0*tmp_38 + 5687282.0*tmp_9 + 33.0) + 4934400.0*tmp_56) + 18000.0*tmp_12*tmp_126*(self._gr_k_244*tmp_9 - tmp_125*tmp_47 + tmp_3*(self._gr_k_241*tmp_9 + self._gr_k_242*tmp_17 + self._gr_k_243)) + tmp_49*(self._gr_k_250*tmp_28*tmp_51*(self._gr_k_240*tmp_28 + 8.0*tmp_10*(self._gr_k_235*tmp_9 + self._gr_k_236) + tmp_137*(self._gr_k_237*tmp_17 + self._gr_k_238*tmp_9 + self._gr_k_239) + tmp_35*(self._gr_k_231*tmp_9 + self._gr_k_232*tmp_17 + self._gr_k_233*tmp_34 + self._gr_k_234)) + 108000.0*tmp_12*tmp_136*(self._gr_k_249*tmp_9 - tmp_125 + tmp_3*(self._gr_k_246*tmp_9 + self._gr_k_248 + tmp_18))))*ccomplex.exp(-tmp_158) + self._gr_k_56*tmp_24*tmp_43*(self._gr_k_53*tmp_17 + self._gr_k_55*tmp_26 - tmp_30*(self._gr_k_49*tmp_9 + self._gr_k_54) + tmp_32*(self._gr_k_50*tmp_9 + self._gr_k_51*tmp_17 + self._gr_k_52) + tmp_36*(self._gr_k_45*tmp_17 + self._gr_k_46*tmp_34 + self._gr_k_47*tmp_9 + self._gr_k_48) + tmp_37*(self._gr_k_40*tmp_38 + self._gr_k_41*tmp_34 + self._gr_k_42*tmp_9 + self._gr_k_43*tmp_17 + self._gr_k_44)) + self._gr_k_586*tmp_124*(<double>(self._gr_k_28)*(-tmp_105*tmp_108 + tmp_109*tmp_120 + 11520.0*tmp_34*tmp_47 + 11520.0*tmp_57*(tmp_47*tmp_9 + tmp_53 + ccomplex.complex[double](0, 1)*(-tmp_54 - 3.0)) + 5760.0*tmp_67*(3.1415926535897932*tmp_17 + tmp_58 + tmp_66*(tmp_53 + tmp_63 + tmp_65) - ccomplex.complex[double](0, 1)*(88.0*tmp_45 + tmp_61 + 6.0)) + tmp_75*(tmp_34*(tmp_62 + tmp_69 + tmp_70) + tmp_68 + tmp_71*(tmp_63 + 68.0*tmp_64 + 25.132741228718346) + tmp_73*(tmp_53 - ccomplex.complex[double](0, 1)*(188.0*tmp_45 + tmp_72 + 3.0)) + ccomplex.complex[double](0, 1)*(2516.0*tmp_45 - 1458.0*tmp_59 - 15.0)) + tmp_88*(tmp_38*(ccomplex.complex[double](0, 1)*(280.0*tmp_45 + tmp_80 + 6.0) + 53.407075111026485) + 56.0*tmp_76 + tmp_82*(tmp_58 + ccomplex.complex[double](0, 1)*(10856.0*tmp_45 - tmp_81 - 6.0)) + tmp_86*(tmp_83 - ccomplex.complex[double](0, 1)*(2840.0*tmp_45 + tmp_85 + 6.0)) - ccomplex.complex[double](0, 1)*(70472.0*tmp_45 - 21870.0*tmp_59 + tmp_79 + 18.0) + 113.09733552923256)) - 2880.0*tmp_47*tmp_48 + tmp_49*(self._gr_k_29*tmp_52*(e*(tmp_9 - 3.0) - tmp_8) - tmp_48*tmp_50)) + self._gr_k_598*tmp_12*tmp_2*tmp_6
        # 22/math.m: hFactEccCorrResumExceptPAv5Flag[2,2]
        self.h22EccCorrResum = self._gr_k_136*tmp_123*tmp_159*(self._gr_k_129*tmp_10*(self._gr_k_127*tmp_252 - 21504.0*tmp_254 - 2688.0*tmp_256 - tmp_257*(self._gr_k_128*tmp_17 - 504.0*tmp_9 - 1386.0) + 1344.0*tmp_259 + tmp_261*(self._gr_k_126*tmp_34 + 126.0*tmp_17 + tmp_260 + 315.0*tmp_38 + 63.0)) + self._gr_k_130*(self._gr_k_107*tmp_34 + self._gr_k_88*tmp_150*tmp_227 + self._gr_k_99*tmp_244*tmp_269 - tmp_155*(self._gr_k_103*tmp_17 + self._gr_k_110*tmp_9 + self._gr_k_111) + tmp_245*(self._gr_k_100*tmp_95 + self._gr_k_101*tmp_34 + self._gr_k_102 + self._gr_k_96*tmp_9 + self._gr_k_97*tmp_38 + self._gr_k_98*tmp_17) - tmp_268*(self._gr_k_104*tmp_95 + self._gr_k_105*tmp_34 + self._gr_k_106*tmp_17 + self._gr_k_108*tmp_116 + self._gr_k_109 + self._gr_k_89*tmp_9 + self._gr_k_95*tmp_38) - 8640.0*tmp_57*(self._gr_k_11*tmp_9 + self._gr_k_12) + 720.0*tmp_74*(self._gr_k_10 + self._gr_k_7*tmp_34 + self._gr_k_8*tmp_9 + self._gr_k_9*tmp_17) + 720.0*tmp_87*(self._gr_k_90*tmp_17 + self._gr_k_91*tmp_9 + self._gr_k_92*tmp_34 + self._gr_k_93*tmp_38 + self._gr_k_94)) + self._gr_k_135*tmp_251*(-tmp_249*(-3477.0*tmp_9 - 9283.0) + 16.0*tmp_250*(101.5*tmp_17 + 1288.0*tmp_9 + 1462.5) + 18816.0*tmp_28 + tmp_35*(-1347.0*tmp_17 + 781.0*tmp_34 + 10299.0*tmp_9 + 3675.0)) + tmp_25*(self._gr_k_121*tmp_34 + self._gr_k_122*tmp_5*(tmp_106*tmp_164*(19265250.0*tmp_17 - 121109530.0*tmp_34 - 22228215.0*tmp_38 + 1534115.0*tmp_9 - 639.0*tmp_95 + 6899.0) + tmp_109*(16155.0*tmp_116 + 90636975.0*tmp_17 - 458313800.0*tmp_34 - 841263775.0*tmp_38 - 230820.0*tmp_9 - 20428476.0*tmp_95 - 16811.0) + tmp_267 + 19200.0*tmp_57*(195.5 - 633.5*tmp_9) + 3200.0*tmp_67*(-6266.5*tmp_17 - 7020.0*tmp_9 + 822.5) + 200.0*tmp_74*(-626476.0*tmp_17 - 64459.0*tmp_34 + 133932.0*tmp_9 + 5639.0) + 200.0*tmp_87*(-611632.0*tmp_17 - 929883.0*tmp_34 - 11655.0*tmp_38 + 106179.0*tmp_9 + 35.0)) + self._gr_k_133*tmp_37*tmp_38 + 3024000.0*tmp_57*(self._gr_k_125*(tmp_103 + 1.0) + 3.1415926535897932*tmp_263) + 1008000.0*tmp_67*(self._gr_k_131*tmp_9*(tmp_9 - 2.0) + 3.1415926535897932*tmp_264) - 504000.0*tmp_74*(self._gr_k_132*(tmp_266 + tmp_9) + 3.1415926535897932*tmp_265)) + tmp_49*(self._gr_k_134*tmp_251*(self._gr_k_120*tmp_28 - tmp_137*(self._gr_k_116*tmp_9 + self._gr_k_117 + 399.0*tmp_17) + tmp_262*(self._gr_k_118*tmp_9 + self._gr_k_119) + tmp_35*(self._gr_k_112*tmp_34 + self._gr_k_113*tmp_9 + self._gr_k_114*tmp_17 + self._gr_k_115)) - 1512000.0*tmp_135*tmp_28*(self._gr_k_123*tmp_255 + self._gr_k_124*tmp_28 - tmp_137*tmp_264 - tmp_262*tmp_263 + tmp_265*tmp_35))) + self._gr_k_163*tmp_180*tmp_248*(self._gr_k_160*(tmp_155*(tmp_198 - tmp_203*(tmp_199*tmp_9 - tmp_202)) + tmp_195*(tmp_186 - ccomplex.complex[double](0, 1)*(tmp_188*tmp_9 - tmp_194)) + tmp_247) + self._gr_k_162*tmp_181*tmp_38 - 126669.01579274046*tmp_126*tmp_177 + tmp_49*(self._gr_k_161*tmp_184 - 120960.0*tmp_136*tmp_177)) + self._gr_k_585*tmp_179*tmp_24*(self._gr_k_577*tmp_56 + self._gr_k_578*tmp_170 + self._gr_k_584*tmp_177*tmp_26 + tmp_106*(self._gr_k_562*tmp_95 + self._gr_k_563*tmp_9 + self._gr_k_564*tmp_38 + self._gr_k_565*tmp_34 + self._gr_k_566*tmp_17 + self._gr_k_567) + tmp_172*(self._gr_k_579*tmp_9 + self._gr_k_580) + tmp_173*(self._gr_k_568*tmp_9 + self._gr_k_569*tmp_17 + self._gr_k_570) - tmp_174*(self._gr_k_576*tmp_34 + self._gr_k_581*tmp_17 + self._gr_k_582*tmp_9 + self._gr_k_583) + tmp_176*(self._gr_k_571*tmp_38 + self._gr_k_572*tmp_9 + self._gr_k_573*tmp_34 + self._gr_k_574*tmp_17 + self._gr_k_575)) + self._gr_k_863*tmp_169*tmp_23*(self._gr_k_862*tmp_165 + e*tmp_8*(self._gr_k_857*tmp_17 + self._gr_k_860*tmp_9 + self._gr_k_861) + tmp_166*(self._gr_k_858*tmp_9 + self._gr_k_859) + tmp_3*(self._gr_k_854*tmp_9 + self._gr_k_855*tmp_17 + self._gr_k_856 + tmp_167)) + self._gr_k_921*tmp_370*cmath.pow(x, 3)*(self._gr_k_886*tmp_271 + self._gr_k_906*tmp_7*(3.0*tmp_106*tmp_7*(8635671898.0*tmp_111 - tmp_321*(tmp_320*tmp_364 + 690900.0*tmp_340 - 1686678000.0*tmp_342 - 10691295300.0*tmp_345 - 9761718750.0*tmp_351 + 140.0*tmp_45*(152732790.0*tmp_59 + 139453125.0*tmp_77 + 285758908.0) - 10894939020.0*tmp_59 - 9947656250.0*tmp_77 - 24207961.0) - tmp_324*(-tmp_313*(-tmp_361 + 235410.0*tmp_45 - 10379.0) - 18992400.0*tmp_342 - 16074450.0*tmp_345 - tmp_355 + 140.0*tmp_45*(tmp_361 + 138244.0) - 16380630.0*tmp_59) - tmp_326*(tmp_325*tmp_364 + 44100.0*tmp_340 + 1232725200.0*tmp_342 - 513872100.0*tmp_345 + 2772656250.0*tmp_351 - tmp_363*(9455130.0*tmp_59 + 39609375.0*tmp_77 + 41029364.0) - 523660140.0*tmp_59 + 2825468750.0*tmp_77 - 1545189.0) - 659.73445725385658*ccomplex.complex[double](0, 1)*tmp_328 - 220500.0*tmp_340 + 24834497520.0*tmp_342 - 22220248050.0*tmp_345 - 4940214300.0*tmp_349 - 52828125000.0*tmp_350 + 26414062500.0*tmp_351 - 16948514940.0*tmp_358 + 8474257470.0*tmp_359 - tmp_360*(tmp_313*(71700090.0*tmp_45 - 43401015.0*tmp_59 + 35417.0) + 396900.0*tmp_340 + 1165592400.0*tmp_342 + 3038071050.0*tmp_345 - 420.0*tmp_45*(14467005.0*tmp_59 + 12177884.0) + 3095939070.0*tmp_59 - 13906701.0) - 50832496848.0*tmp_45 - 22643490870.0*tmp_59 + 26917187500.0*tmp_77 + tmp_95*(73500.0*tmp_340 - 6903120.0*tmp_342 - 6072570.0*tmp_345 + 28.0*tmp_45*(tmp_362 + 251236.0) - 6188238.0*tmp_59 + 43.982297150257105*ccomplex.complex[double](0, 1)*(-tmp_362 + 477330.0*tmp_45 - 51467.0) - 2575315.0) + 7725945.0) + 14.0*tmp_109*(12711386205.0*tmp_111*tmp_284 - 12953507847.0*tmp_111 - tmp_115*(-tmp_192*(610063650.0*tmp_59 + 2272265625.0*tmp_77 + 2435777104.0) + tmp_292*tmp_335 + 985041120.0*tmp_342 - 315824670.0*tmp_345 + 2272265625.0*tmp_351 - 321840378.0*tmp_59 + 2315546875.0*tmp_77) + tmp_116*(-tmp_284*(6387570.0*tmp_45 - 3903795.0*tmp_59 - 107.0) + tmp_330*tmp_365 + 1669920.0*tmp_342 - 30.0*tmp_45*(260253.0*tmp_59 + 217424.0)) + tmp_212*(-tmp_287*(12075126.0*tmp_45 - 7424865.0*tmp_59 + 1391.0) - tmp_337*tmp_365 - 15990240.0*tmp_342 + tmp_45*(74248650.0*tmp_59 + 61436832.0)) + tmp_216*(-tmp_332*tmp_365 - 1179360.0*tmp_342 + 702.0*tmp_45*(2835.0*tmp_59 + 1712.0) + 12.566370614359173*ccomplex.complex[double](0, 1)*(-tmp_322 + 95970.0*tmp_45 - 3103.0)) + tmp_253*(4317835949.0*tmp_111 + tmp_292*tmp_368 + 12696290880.0*tmp_342 - 11220808095.0*tmp_345 + 13822265625.0*tmp_351 + 4237128735.0*tmp_359 - 6.0*tmp_45*(1412376245.0*tmp_111 + 462969675.0*tmp_59 + 4607421875.0*tmp_77 + 4449977632.0) - 11434537773.0*tmp_59 + 14085546875.0*tmp_77) - 36743316960.0*tmp_284*tmp_45 - 17238469815.0*tmp_284*tmp_59 + 12222656250.0*tmp_284*tmp_77 - 7314232800.0*tmp_342 + 17238469815.0*tmp_345 + 1235895570.0*tmp_349 + 24445312500.0*tmp_350 - 12222656250.0*tmp_351 + 25422772410.0*tmp_358 - 12711386205.0*tmp_359 + 37443189664.0*tmp_45 + 17566821621.0*tmp_59 - 12455468750.0*tmp_77 + tmp_94*(tmp_288*(298660590.0*tmp_45 - 88015815.0*tmp_59 - 69140625.0*tmp_77 + 107.0) + 113977920.0*tmp_342 + 616110705.0*tmp_345 + 483984375.0*tmp_351 - tmp_366*(123222141.0*tmp_59 + 96796875.0*tmp_77 + 213044704.0) + 627846147.0*tmp_59 + 493203125.0*tmp_77)) - 768.0*tmp_34*(89880.0*tmp_284 + tmp_341) - tmp_348*(6.0*tmp_34*(-tmp_305*(tmp_299 - tmp_300 - 1391.0) + 191100.0*tmp_340 - 7232400.0*tmp_342 - 7144200.0*tmp_345 + 1680.0*tmp_45*(tmp_300 + 4387.0) - 7280280.0*tmp_59 - 6695819.0) + tmp_38*(tmp_298*tmp_305 + 249900.0*tmp_340 + tmp_343 - 8756071.0) + tmp_71*(-tmp_303*(tmp_297 + 53393.0) + tmp_305*(-tmp_293 + 34930.0*tmp_45 + 2033.0) - 764400.0*tmp_342 + tmp_346 + tmp_347)) - tmp_356*(tmp_253*(tmp_292*tmp_304 + 176400.0*tmp_340 + 6585600.0*tmp_342 + 36614025.0*tmp_345 - 70.0*tmp_45*(1046115.0*tmp_59 + 972416.0) + 37311435.0*tmp_59 - 6180756.0) + tmp_292*tmp_310 + tmp_309*(-tmp_292*(tmp_306 - tmp_307 - 856.0) - 1218000.0*tmp_342 - 1148175.0*tmp_345 - tmp_355 + 50.0*tmp_45*(tmp_91 + 24824.0) - 1170045.0*tmp_59) + tmp_34*(tmp_341 + 58800.0*tmp_342 - tmp_352 + tmp_354*(tmp_353 + 3317.0)) + 235200.0*tmp_340 - 7291200.0*tmp_342 - 40186125.0*tmp_345 + 80372250.0*tmp_349 + 114843750.0*tmp_350 - 57421875.0*tmp_351 + 191504320.0*tmp_45 - 40951575.0*tmp_59 - 58515625.0*tmp_77 - 8241008.0) + 12.0*tmp_37*tmp_9*(-tmp_185*(58800.0*tmp_340 - 12818400.0*tmp_342 - 11609325.0*tmp_345 - tmp_354*(-tmp_314 + 367710.0*tmp_45 - 11021.0) + 70.0*tmp_45*(tmp_314 + 186608.0) - 11830455.0*tmp_59 - 2060252.0) - 29400.0*tmp_284*(tmp_357 + 152774.0*tmp_45 + 18711.0*tmp_59 + 107.0) - 514500.0*tmp_340 - 1059517200.0*tmp_342 + 550103400.0*tmp_345 - 2296875000.0*tmp_351 + 5.0*tmp_38*(-112.0*tmp_284*(tmp_353 + 107.0) + 58800.0*tmp_342 - tmp_352 - tmp_355) + 560.0*tmp_45*(1709505.0*tmp_59 + 8203125.0*tmp_77 + 8173409.0) + 560581560.0*tmp_59 - 2340625000.0*tmp_77 - tmp_86*(tmp_305*tmp_315 + 84848400.0*tmp_342 + 278623800.0*tmp_345 + tmp_347 - 50960.0*tmp_45*(tmp_307 + 9523.0) + 283930920.0*tmp_59) - 56.0*tmp_9*(tmp_292*tmp_316 + 29400.0*tmp_340 - 5208000.0*tmp_342 - 36741600.0*tmp_345 - 41015625.0*tmp_351 + 50.0*tmp_45*(1469664.0*tmp_59 + 1640625.0*tmp_77 + 3064480.0) - 37441440.0*tmp_59 - 41796875.0*tmp_77 - 1030126.0) + 18027205.0) - 48.0*tmp_57*(-5880.0*tmp_284*(-tmp_344 + 2430.0*tmp_45 - 749.0) + 720300.0*tmp_340 - 14288400.0*tmp_342 - tmp_346 + 136080.0*tmp_45*(210.0*tmp_59 + 107.0) + tmp_9*(tmp_295*tmp_305 + 426300.0*tmp_340 + tmp_343 - 14936827.0) - 25238087.0)) + self._gr_k_907*tmp_273 + self._gr_k_909*tmp_139 + 4.0*tmp_109*(self._gr_k_877*tmp_28 + self._gr_k_880*tmp_275 + self._gr_k_881*tmp_139 + self._gr_k_882*tmp_56 + self._gr_k_883*tmp_276 + self._gr_k_884*tmp_7 + self._gr_k_885*tmp_270) - 16.0*tmp_141*(self._gr_k_918*tmp_9 + self._gr_k_919*tmp_17 + self._gr_k_920) + 8.0*tmp_142*(self._gr_k_870*tmp_34 + self._gr_k_871*tmp_9 + self._gr_k_872*tmp_17 + self._gr_k_873) + tmp_144*(self._gr_k_864*tmp_38 + self._gr_k_865*tmp_95 + self._gr_k_866*tmp_17 + self._gr_k_867*tmp_34 + self._gr_k_868*tmp_9 + self._gr_k_869) - 4.0*tmp_145*(self._gr_k_905*tmp_17 + self._gr_k_912*tmp_34 + self._gr_k_913*tmp_9 + self._gr_k_914*tmp_38 + self._gr_k_915) + 4.0*tmp_146*(self._gr_k_874*tmp_17 + self._gr_k_878*tmp_34 + self._gr_k_888*tmp_38 + self._gr_k_889*tmp_148 + self._gr_k_890*tmp_9 + self._gr_k_891*tmp_95 + self._gr_k_892*tmp_116 + self._gr_k_893) + tmp_25*(self._gr_k_887*tmp_7*(tmp_155*(tmp_198 + tmp_203*(-tmp_199*tmp_9 + tmp_202)) + tmp_195*(tmp_186 + ccomplex.complex[double](0, 1)*(tmp_194 + tmp_253*(5066.0*tmp_45 - 2997.0*tmp_59))) + tmp_247) + self._gr_k_894*tmp_270*tmp_37 + self._gr_k_903*tmp_283 + self._gr_k_904*tmp_139 - 7761600.0*tmp_141*(self._gr_k_898*tmp_9 + self._gr_k_899*tmp_17 + self._gr_k_900) + 186278400.0*tmp_142*(self._gr_k_879*tmp_34 + self._gr_k_895*tmp_9 + self._gr_k_896*tmp_17 + self._gr_k_897) - 3880800.0*tmp_272*(self._gr_k_901*tmp_9 + self._gr_k_902)) - 16.0*tmp_272*(self._gr_k_910*tmp_9 + self._gr_k_911) - 31890862080.0*tmp_274*cmath.log(16*x) + 996589440.0*tmp_282*ccomplex.log(2*tmp_162*x*((1.0/2.0)*e*(tmp_163 + tmp_7) + 1)*cmath.exp(-1.0/2.0)) + cmath.pow(tmp_49, 2)*(self._gr_k_916*tmp_183*tmp_26 + self._gr_k_917*tmp_282 + 140826470400.0*tmp_274) + tmp_49*(self._gr_k_908*(3072.0*tmp_139*(107.0 - tmp_292) + 7.0*tmp_149*(tmp_116*(tmp_284 - tmp_329 + tmp_330) + tmp_212*(-65.0*tmp_284 - tmp_337 + 574176.0*tmp_45) + tmp_216*(-116.0*tmp_284 - tmp_332 + 11232.0*tmp_45) + tmp_336 + tmp_339 + tmp_9*(-tmp_331 - 748594368.0*tmp_45 - 320594517.0*tmp_59 + 394921875.0*tmp_77) + tmp_94*(tmp_288 + tmp_334)) + tmp_290*(-6300.0*tmp_284 + tmp_321*(-19740.0*tmp_284 + tmp_320) + tmp_324*(-3395.0*tmp_284 - tmp_323 - 90440.0*tmp_45 + 428.0) + tmp_326*(-1260.0*tmp_284 + tmp_325) + tmp_328 + tmp_34*(-23170.0*tmp_284 + 47801040.0*tmp_45 - 28934010.0*tmp_59 + 23112.0) + tmp_95*(-3367.0*tmp_284 + 32872.0*tmp_45 - 28917.0*tmp_59 - 2140.0)) + tmp_296*(-5145.0*tmp_284 + tmp_293 - 17010.0*tmp_45 + tmp_9*(-3045.0*tmp_284 + tmp_295) + 5243.0) + tmp_302*(tmp_17*(-1785.0*tmp_284 + tmp_298) - 5985.0*tmp_284 - tmp_297 + tmp_301*(-1365.0*tmp_284 - tmp_299 + tmp_300 + 1391.0) + 104790.0*tmp_45 + 6099.0) + tmp_311*(tmp_253*(-10080.0*tmp_284 + tmp_304) - 13440.0*tmp_284 + tmp_309*(-tmp_305 - tmp_306 - tmp_308 + 856.0) + tmp_310 + tmp_34*(-1085.0*tmp_284 + tmp_303 + 856.0)) + tmp_318*(-3675.0*tmp_284 - tmp_312 + tmp_317*(-1680.0*tmp_284 + tmp_316) + tmp_34*(-3605.0*tmp_284 - 122080.0*tmp_45 + 110565.0*tmp_59 + 3424.0) + tmp_38*(-tmp_313 - 350.0*tmp_45 - 535.0) + 5347090.0*tmp_45 + 654885.0*tmp_59 + tmp_86*(-1995.0*tmp_284 + tmp_315) + 3745.0)) + tmp_25*(self._gr_k_875*tmp_283 + self._gr_k_876*(46080.0*tmp_139*tmp_284 + 5760.0*tmp_141*(ccomplex.complex[double](0, 1)*tmp_198 + tmp_199*tmp_9 - tmp_200 - tmp_201 + tmp_60) + 240.0*tmp_142*(-tmp_118 + tmp_188*tmp_9 + tmp_189 + tmp_191 - tmp_193 + tmp_284*(78.0*tmp_17 + 23.0*tmp_34 + 66.0*tmp_9 + 18.0) + tmp_286) + 240.0*tmp_145*(-tmp_228 - tmp_229 - tmp_231 + tmp_233 + tmp_34*(-tmp_232 - tmp_288 + 1288.0*tmp_45) + tmp_38*(-tmp_287 - tmp_46) + tmp_78) + tmp_149*(tmp_116*(ccomplex.complex[double](0, 1)*tmp_218 - 19152.0*tmp_45 + 12393.0*tmp_59) - tmp_204 - tmp_205 + tmp_206 + tmp_208 + tmp_212*(-tmp_210 + tmp_211 + tmp_288) + tmp_216*(ccomplex.complex[double](0, 1)*tmp_213 + tmp_214 - tmp_215) - tmp_217 + tmp_222*tmp_253 + tmp_94*(tmp_224 - tmp_225 + tmp_288 + tmp_291)) + 5760.0*tmp_272*(ccomplex.complex[double](0, 1)*tmp_246 - tmp_253*tmp_45 + tmp_285 + 27.0*tmp_45) + tmp_290*(tmp_212*(768.0*tmp_45 - ccomplex.complex[double](0, 1)*tmp_83 - tmp_99) + tmp_236 + tmp_241 - tmp_243*tmp_289 + tmp_94*(-tmp_237 - tmp_288 + 111537.0*tmp_59) + tmp_95*(tmp_220 + ccomplex.complex[double](0, 1)*tmp_238 + tmp_239)))) - 93884313600.0*tmp_274*tmp_284)) + 0.25*tmp_162*(e*(tmp_163 + tmp_164) + tmp_32 + 4.0)
        # 31/math.m: hFactEccCorrResumExceptPAv5Flag[3,1]
        self.h31EccCorrResum = self._gr_k_363*tmp_159*tmp_42*(self._gr_k_282*tmp_109 + self._gr_k_341*tmp_430*(-2491.0*tmp_3 - 69834.0) + self._gr_k_344*tmp_432*(-16112.0*tmp_3 - 217.0*tmp_37 - 62032.0) + self._gr_k_348*tmp_431*(-3285.0*tmp_3 - 23224.0) + self._gr_k_351*tmp_107 + self._gr_k_352*tmp_428 + self._gr_k_353*tmp_395*(self._gr_k_349*tmp_441 + 1056.0*tmp_134*(3143.0*tmp_17 - 12602.0*tmp_9 - 1181.0) - tmp_171*(self._gr_k_350*tmp_9 + 119196.0) + tmp_37*(self._gr_k_268*tmp_34 - 243540.0*tmp_17 + 131670.0*tmp_38 - 56133.0*tmp_9 + 12573.0*tmp_95 - 6930.0) - 132.0*tmp_433*(653.0*tmp_9 + 45811.0) + 1533312.0*tmp_440 + tmp_442*(self._gr_k_340*tmp_17 + 251262.0*tmp_34 - 722898.0*tmp_9 - 113355.0) + 264.0*tmp_443*(27315.0*tmp_17 + 4005.0*tmp_34 + 638.0*tmp_38 - 2364.0*tmp_9 - 330.0)) + self._gr_k_354*tmp_10*tmp_40*(1600.0*tmp_10*(2545.0*tmp_17 + 33192.0*tmp_9 + 25007.0) + tmp_106*(928436400.0*tmp_17 - 447565875.0*tmp_34 - 43033510.0*tmp_38 + 1679811375.0*tmp_9 + 38962318.0) + tmp_130*tmp_7*(153024.0*tmp_17 - 42323.0*tmp_34 + 392928.0*tmp_9 + 48905.0) + tmp_131*(434816.0*tmp_17 - 65243.0*tmp_34 - 12233.0*tmp_38 + 611759.0*tmp_9 + 8637.0) + 150.0*tmp_132*(346846.0*tmp_17 - 911405.0*tmp_34 + 6656798.0*tmp_9 + 1141287.0) + 19200.0*tmp_28*(629.0*tmp_9 + 397.0)) + self._gr_k_355*tmp_87*(-16125.0*tmp_3 - 115174.0) + self._gr_k_356*tmp_391*(-13876.0*tmp_3 - 1693.0*tmp_37 - 6728.0) + self._gr_k_357*tmp_57*(-50204.0*tmp_3 - 10847.0*tmp_37 - 21144.0) + self._gr_k_358*(self._gr_k_338*tmp_269*tmp_423 + self._gr_k_339*tmp_34 + tmp_108*(self._gr_k_310*tmp_17 + self._gr_k_311*tmp_95 + self._gr_k_312*tmp_34 + self._gr_k_313*tmp_9 + self._gr_k_314*tmp_38 + self._gr_k_315) + 4.0*tmp_150*(self._gr_k_283*tmp_38 + self._gr_k_323*tmp_9 + self._gr_k_324*tmp_95 + self._gr_k_325*tmp_34 + self._gr_k_326*tmp_17 + self._gr_k_327*tmp_116 + self._gr_k_328) + tmp_268*(self._gr_k_286*tmp_34 + self._gr_k_317*tmp_38 + self._gr_k_329*tmp_95 + self._gr_k_330*tmp_9 + self._gr_k_331*tmp_116 + self._gr_k_332*tmp_17 + self._gr_k_333) + tmp_404*(self._gr_k_318*tmp_17 + self._gr_k_319*tmp_9 + self._gr_k_320) + tmp_418*(self._gr_k_285*tmp_34 + self._gr_k_306*tmp_17 + self._gr_k_307*tmp_38 + self._gr_k_308*tmp_9 + self._gr_k_309) + 4608.0*tmp_57*(self._gr_k_321*tmp_9 + self._gr_k_322) + 1152.0*tmp_74*(self._gr_k_334*tmp_9 + self._gr_k_335*tmp_34 + self._gr_k_336*tmp_17 + self._gr_k_337)) + self._gr_k_359*tmp_67*(-598736.0*tmp_3 - 38499.0*tmp_37 - 548208.0) + self._gr_k_360*tmp_74*(-28089.0*tmp_3 - 59864.0) + self._gr_k_361*tmp_106*tmp_40*(e*(50455.0 - 42339.0*tmp_116) + tmp_277*(-74107.0*tmp_95 - 943.0)) + self._gr_k_362*tmp_429 + 79200.0*tmp_25*tmp_402*(self._gr_k_280*tmp_433 + self._gr_k_281*tmp_17 + tmp_30*tmp_435 + tmp_386*tmp_436 + tmp_437*tmp_438 + tmp_439) - 79200.0*tmp_34*(self._gr_k_255*tmp_109 + self._gr_k_316*tmp_37 + self._gr_k_345*tmp_150 + self._gr_k_346*tmp_3 + self._gr_k_347) + tmp_49*(self._gr_k_342*(self._gr_k_287*tmp_150*tmp_34 + self._gr_k_299*tmp_34 + tmp_109*(self._gr_k_256*tmp_116 + self._gr_k_257*tmp_38 + self._gr_k_258*tmp_34 + self._gr_k_259*tmp_9 + self._gr_k_260*tmp_95 + self._gr_k_261*tmp_17 + self._gr_k_262) + tmp_133*tmp_175*(self._gr_k_288*tmp_38 + self._gr_k_289*tmp_17 + self._gr_k_290*tmp_34 + self._gr_k_291*tmp_9 + self._gr_k_292) + tmp_29*tmp_35*(self._gr_k_298*tmp_34 + self._gr_k_300*tmp_9 + self._gr_k_301*tmp_17 + self._gr_k_302) + tmp_387*(self._gr_k_293*tmp_34 + self._gr_k_294*tmp_17 + self._gr_k_295*tmp_9 + self._gr_k_296*tmp_38 + self._gr_k_297) - tmp_450*(self._gr_k_305 + tmp_434) + tmp_451*(self._gr_k_284*tmp_17 + self._gr_k_303*tmp_9 + self._gr_k_304)) + self._gr_k_343*tmp_444 + 1188000.0*tmp_135*(self._gr_k_269*tmp_445 + self._gr_k_272*tmp_34 + tmp_435*tmp_446 + tmp_436*tmp_447 + tmp_437*tmp_448 + tmp_439*tmp_9))) + self._gr_k_589*tmp_427*(<double>(self._gr_k_27)*(4608.0*e*tmp_56*(-tmp_400 + tmp_401 + tmp_402) - tmp_108*tmp_423 + tmp_109*(-tmp_115*(6708987.0*tmp_113 + tmp_415 - 22282240.0*tmp_64 + 5046875.0*ccomplex.complex[double](0, 1)*tmp_77) + tmp_116*(tmp_117 + tmp_119*(19375.0*tmp_77 + tmp_91) - 2945024.0*tmp_64) + tmp_140*(373977.0*tmp_113 - 609024.0*tmp_64 + 147.65485471872028) + tmp_426*(4374.0*tmp_113 + tmp_424 - tmp_425) - 72.0*ccomplex.complex[double](0, 1)*tmp_9*(6761984.0*tmp_45 + 106434.0*tmp_59 - 2984375.0*tmp_77) - 24.0*tmp_95*(-81.0*ccomplex.complex[double](0, 1)*(2048.0*tmp_45 - 1287.0*tmp_59) + 411.54863762026291) + ccomplex.complex[double](0, 1)*(-23882747.0*tmp_111 + 201052160.0*tmp_45 + 65498463.0*tmp_59 - 102421875.0*tmp_77)) + 4608.0*tmp_34*tmp_394 - tmp_404*(tmp_17*(6.0*tmp_393 + 78.539816339744831) + tmp_82*(tmp_400 - tmp_401) - ccomplex.complex[double](0, 1)*(tmp_403 + 1180.0*tmp_45 + 378.0) + 848.23001646924417) - tmp_410*(tmp_17*(tmp_406 + 2356.1944901923449 - 819.0*ccomplex.complex[double](0, 1)) + tmp_405*(3962.0*tmp_45 - tmp_84 - 105.0) + tmp_408*(tmp_407 - 34.0*tmp_64 + 50.265482457436692) + tmp_73*(tmp_409 - ccomplex.complex[double](0, 1)*(tmp_403 + 1190.0*tmp_45 + 161.0)) + 1178.0972450961725) - tmp_418*(tmp_101*(tmp_415 + ccomplex.complex[double](0, 1)*(2168.0*tmp_45 - 1701.0*tmp_59 + 84.0)) + tmp_185*(tmp_413 - tmp_414*(110.0*tmp_45 + 21.0)) - tmp_405*(tmp_412 + 239464.0*tmp_45 - 59375.0*tmp_77 + 252.0) + tmp_411 + tmp_82*(ccomplex.complex[double](0, 1)*(172160.0*tmp_45 - 100845.0*tmp_59 - 672.0) + 1507.9644737231008) + tmp_86*(-ccomplex.complex[double](0, 1)*(tmp_417 + 33560.0*tmp_45 + 1092.0) + 2623.2298657474774))) + self._gr_k_588*tmp_391*(7614.0*tmp_3 + 653.0*tmp_37 + 4928.0) + 1152.0*tmp_389*tmp_394 + tmp_49*(self._gr_k_587*tmp_399 + tmp_389*tmp_50)) - self._gr_k_600*tmp_374*tmp_375 + self._gr_k_705*tmp_380*(self._gr_k_697*tmp_255 - tmp_249*(self._gr_k_699*tmp_9 + self._gr_k_700*tmp_17 + self._gr_k_701) + tmp_35*(self._gr_k_692*tmp_9 + self._gr_k_693*tmp_34 + self._gr_k_694*tmp_17 + self._gr_k_695*tmp_38 + self._gr_k_696) + tmp_376*(self._gr_k_698 + tmp_253) - tmp_378*(self._gr_k_702*tmp_9 + self._gr_k_703*tmp_17 + self._gr_k_704 + tmp_377)) + self._gr_k_736*tmp_390*(self._gr_k_733*tmp_381 + self._gr_k_734*tmp_34 + self._gr_k_735*tmp_389 + tmp_109*(self._gr_k_706*tmp_116 + self._gr_k_707*tmp_95 + self._gr_k_708*tmp_34 + self._gr_k_709*tmp_38 + self._gr_k_710*tmp_17 + self._gr_k_711*tmp_9 + self._gr_k_712) + tmp_383*(self._gr_k_713*tmp_9 + self._gr_k_721) + tmp_384*(self._gr_k_718*tmp_9 + self._gr_k_719*tmp_17 + self._gr_k_720) + tmp_385*(self._gr_k_714*tmp_9 + self._gr_k_715*tmp_34 + self._gr_k_716*tmp_17 + self._gr_k_717) + tmp_387*(self._gr_k_728*tmp_34 + self._gr_k_729*tmp_17 + self._gr_k_730*tmp_9 + self._gr_k_731*tmp_38 + self._gr_k_732) + tmp_388*(self._gr_k_722*tmp_95 + self._gr_k_723*tmp_38 + self._gr_k_724*tmp_34 + self._gr_k_725*tmp_17 + self._gr_k_726*tmp_9 + self._gr_k_727))
        # 32/math.m: hFactEccCorrResumExceptPAv5Flag[3,2]
        self.h32EccCorrResum = self._gr_k_1157*tmp_465*(self._gr_k_30*(tmp_109*(tmp_116*(ccomplex.complex[double](0, 1)*tmp_329 - ccomplex.complex[double](0, 1)*tmp_330 + 3.1415926535897932) - tmp_212*(-ccomplex.complex[double](0, 1)*tmp_337 + 574176.0*ccomplex.complex[double](0, 1)*tmp_45 + tmp_464) - tmp_216*(351.0*ccomplex.complex[double](0, 1)*(tmp_285 + 32.0*tmp_45) + 364.42474781641602) - ccomplex.complex[double](0, 1)*tmp_336 - ccomplex.complex[double](0, 1)*tmp_339 + 10.0*tmp_34*(-ccomplex.complex[double](0, 1)*tmp_334 + tmp_96) + 3.0*ccomplex.complex[double](0, 1)*tmp_368*tmp_9) - tmp_195*(tmp_17*(-tmp_62*(2320.0*tmp_45 + tmp_85 + 252.0) + 1583.3626974092558) + tmp_253*(ccomplex.complex[double](0, 1)*(18176.0*tmp_45 - 9963.0*tmp_59 - 432.0) + 904.77868423386045) + tmp_34*(tmp_218 + 4.0*ccomplex.complex[double](0, 1)*tmp_461) - ccomplex.complex[double](0, 1)*(tmp_308 + 51136.0*tmp_45 + tmp_79 + 576.0) + 1206.3715789784806) - tmp_245*(tmp_212*(-ccomplex.complex[double](0, 1)*(2584.0*tmp_45 + tmp_85 + 18.0) + 304.73448739820994) + tmp_411 + tmp_462*(ccomplex.complex[double](0, 1)*(5368328.0*tmp_45 + 489402.0*tmp_59 - 2640625.0*tmp_77 - 378.0) + 791.6813487046279) + tmp_92*(-ccomplex.complex[double](0, 1)*(5341288.0*tmp_45 + tmp_463 - 1454598.0*tmp_59 + 846.0) + 1771.8582566246434) + tmp_94*(tmp_62*(227624.0*tmp_45 - 137781.0*tmp_59 - 162.0) + 1039.8671683382216) + tmp_95*(ccomplex.complex[double](0, 1)*(4696.0*tmp_45 - 4131.0*tmp_59 + 450.0) + 1511.1060663766905) - ccomplex.complex[double](0, 1)*(tmp_112 + 33933576.0*tmp_45 + 15115815.0*tmp_59 - 17968750.0*tmp_77 + 1350.0)) - 23040.0*tmp_34*(tmp_44 + tmp_63) - 1440.0*tmp_57*(-tmp_62*(108.0*tmp_45 - 108.0*tmp_59 + 49.0) + tmp_9*(tmp_62*(-tmp_54 - 29.0) + 182.21237390820801) + 307.87608005179974) - 1440.0*tmp_67*(tmp_17*(tmp_62*(-tmp_54 - 17.0) + 106.81415022205297) + tmp_301*(-ccomplex.complex[double](0, 1)*(164.0*tmp_45 + tmp_72 + 39.0) + 81.681408993334624) + tmp_460 + ccomplex.complex[double](0, 1)*(-tmp_215 + 1996.0*tmp_45 - 171.0)) - tmp_88*(tmp_34*(-tmp_203*(3488.0*tmp_45 - tmp_93 + 144.0) + 647.16808663949741) + tmp_38*(-tmp_405*tmp_461 + tmp_69) + tmp_66*(-ccomplex.complex[double](0, 1)*(tmp_357 + 286400.0*tmp_45 - 69984.0*tmp_59 + 504.0) + 1055.5751316061705) + tmp_86*(-tmp_221*(2106.0*tmp_59 + 19.0) + tmp_460 + 32396.0*tmp_64) + ccomplex.complex[double](0, 1)*(305548.0*tmp_45 + 37422.0*tmp_59 - 156250.0*tmp_77 - 315.0) + 659.73445725385658)) + self._gr_k_33*tmp_457 + self._gr_k_34*tmp_459*(tmp_3*(35.0*tmp_17 - 86.0*tmp_9 - 145.0) - 30.0*tmp_395*(tmp_9 + 13.0) - tmp_458) + tmp_49*(self._gr_k_31*tmp_281*tmp_459 + self._gr_k_32*tmp_457)) + self._gr_k_1160*tmp_24*tmp_456*(self._gr_k_929*tmp_182*tmp_26 + self._gr_k_943*tmp_56 + tmp_106*(self._gr_k_922*tmp_17 + self._gr_k_923*tmp_34 + self._gr_k_924*tmp_95 + self._gr_k_925*tmp_9 + self._gr_k_926*tmp_38 + self._gr_k_927) + 40.0*tmp_132*(self._gr_k_930*tmp_17 + self._gr_k_931*tmp_38 + self._gr_k_932*tmp_9 + self._gr_k_933*tmp_34 + self._gr_k_934) - tmp_172*(self._gr_k_941*tmp_9 + self._gr_k_942) + 160.0*tmp_454*(self._gr_k_928*tmp_17 + self._gr_k_939*tmp_9 + self._gr_k_940) + tmp_455*(self._gr_k_935*tmp_34 + self._gr_k_936*tmp_17 + self._gr_k_937*tmp_9 + self._gr_k_938)) + self._gr_k_1165*tmp_23*tmp_453*(tmp_3*(self._gr_k_947*tmp_34 + self._gr_k_948*tmp_9 + self._gr_k_949*tmp_17 + self._gr_k_950) + tmp_386*(self._gr_k_954*tmp_9 + self._gr_k_955) - tmp_452*(self._gr_k_951*tmp_17 + self._gr_k_952*tmp_9 + self._gr_k_953)) + self._gr_k_601*tmp_168*tmp_180*tmp_183
        # 33/math.m: hFactEccCorrResumExceptPAv5Flag[3,3]
        self.h33EccCorrResum = self._gr_k_446*tmp_161*tmp_39*(tmp_49*(self._gr_k_445*tmp_12*tmp_491*(self._gr_k_423*tmp_17 - tmp_36*(self._gr_k_442*tmp_9 + self._gr_k_443*tmp_17 + self._gr_k_444 + 648.0*tmp_34) + tmp_442*(self._gr_k_420*tmp_9 + self._gr_k_421*tmp_17 + self._gr_k_422) + tmp_493*(self._gr_k_435*tmp_9 + self._gr_k_436) + tmp_494*(self._gr_k_437*tmp_38 + self._gr_k_438*tmp_17 + self._gr_k_439*tmp_34 + self._gr_k_440*tmp_9 + self._gr_k_441)) - 2138400.0*ccomplex.complex[double](0, 1)*tmp_5*(self._gr_k_369*tmp_490 + self._gr_k_384*tmp_34 + tmp_253*tmp_37*(self._gr_k_370*tmp_17 + self._gr_k_371*tmp_9 + self._gr_k_372*tmp_34 + self._gr_k_374 + tmp_469) - tmp_3*tmp_447*(self._gr_k_381*tmp_9 + self._gr_k_382*tmp_17 + self._gr_k_383) - tmp_446*(self._gr_k_379*tmp_9 + self._gr_k_380) - tmp_448*(self._gr_k_375*tmp_17 + self._gr_k_376*tmp_34 + self._gr_k_377*tmp_9 + self._gr_k_378))) + tmp_491*(self._gr_k_399*tmp_7*(self._gr_k_398*tmp_146*tmp_38 + 5.0*tmp_106*(self._gr_k_396*tmp_38 + 77220.0*tmp_17 + 70488.0*tmp_34 + 33660.0*tmp_9 + 25740.0*tmp_95 + 5643.0) - 52140.0*tmp_127*tmp_170 + 63360.0*tmp_129*(tmp_9 + 33.0) + 528.0*tmp_132*(3900.0*tmp_17 + 685.0*tmp_34 + 1985.0*tmp_38 + 2604.0*tmp_9 + 570.0) - 2112.0*tmp_454*(1815.0*tmp_17 - 1850.0*tmp_9 - 1149.0) - 120.0*tmp_496*(self._gr_k_397*tmp_34 - 18018.0*tmp_17 - 32373.0*tmp_9 - 10362.0) + 464640.0*tmp_56) + self._gr_k_400*tmp_40*(tmp_106*tmp_8*(111775290.0*tmp_17 - 432863430.0*tmp_34 - 120431395.0*tmp_38 + 13713305.0*tmp_9 - 696041.0*tmp_95 + 19531.0) + tmp_109*(7841.0*tmp_116 + 279586345.0*tmp_17 - 523205040.0*tmp_34 - 1460169725.0*tmp_38 + 3779906.0*tmp_9 - 92388762.0*tmp_95 - 9877.0) + tmp_267 + tmp_37*tmp_497*(-3233984.0*tmp_17 - 7515715.0*tmp_34 - 252553.0*tmp_38 + 1521791.0*tmp_9 + 12677.0) + 3840.0*tmp_57*(1505.0 - 3983.0*tmp_9) + 320.0*tmp_67*(-95105.0*tmp_17 - 69528.0*tmp_9 + 21345.0) + 80.0*tmp_74*(-2086112.0*tmp_17 - 320051.0*tmp_34 + 627616.0*tmp_9 + 41529.0)) + self._gr_k_418*(self._gr_k_409*tmp_34 + self._gr_k_410*tmp_150*tmp_489 + self._gr_k_411*tmp_269*tmp_486 - tmp_109*(self._gr_k_401*tmp_9 + self._gr_k_412*tmp_116 + self._gr_k_413*tmp_34 + self._gr_k_414*tmp_95 + self._gr_k_415*tmp_38 + self._gr_k_416*tmp_17 + self._gr_k_417) + tmp_410*(self._gr_k_402*tmp_17 + self._gr_k_403*tmp_9 + self._gr_k_404*tmp_34 + self._gr_k_405) + tmp_478*(self._gr_k_20*tmp_9 + self._gr_k_21) + tmp_479*(self._gr_k_406*tmp_9 + self._gr_k_407*tmp_17 + self._gr_k_408) + tmp_482*(self._gr_k_385*tmp_9 + self._gr_k_386*tmp_34 + self._gr_k_387*tmp_17 + self._gr_k_388*tmp_38 + self._gr_k_389) - tmp_498*(self._gr_k_390*tmp_17 + self._gr_k_391*tmp_34 + self._gr_k_392*tmp_95 + self._gr_k_393*tmp_9 + self._gr_k_394*tmp_38 + self._gr_k_395)) + self._gr_k_419*tmp_12*(247600.0*tmp_17 - tmp_30*(-19807.0*tmp_9 - 35715.0) + tmp_37*(64494.0*tmp_17 - 10760.0*tmp_34 + 4463.0*tmp_38 + 55920.0*tmp_9 + 22155.0) + tmp_438*(33145.0*tmp_17 + 4731.0*tmp_34 + 86305.0*tmp_9 + 42243.0) - tmp_495*(-16005.0*tmp_17 - 69688.0*tmp_9 - 60579.0))) - 1760.0*tmp_5*tmp_9*(self._gr_k_424*tmp_445 + self._gr_k_434*tmp_17 - tmp_30*(self._gr_k_425*tmp_9 + self._gr_k_426) - 4.0*tmp_35*(self._gr_k_427*tmp_7 + self._gr_k_428*tmp_56 + self._gr_k_429*tmp_28 + self._gr_k_430*tmp_139) + tmp_37*(self._gr_k_364*tmp_38 + self._gr_k_365*tmp_34 + self._gr_k_366*tmp_17 + self._gr_k_367*tmp_9 + self._gr_k_368) - tmp_442*(self._gr_k_431*tmp_9 + self._gr_k_432*tmp_17 + self._gr_k_433))) + self._gr_k_593*tmp_427*(self._gr_k_591*tmp_431*(79.0*tmp_3 + 1076.0) + self._gr_k_592*(8.0*tmp_106*tmp_486*tmp_7 + tmp_109*tmp_489 - tmp_410*(tmp_17*(ccomplex.complex[double](0, 1)*(73470.0*tmp_45 - 40095.0*tmp_59 - 1099.0) + 2466.1502330679877) + tmp_405*(27162.0*tmp_45 + 5967.0*tmp_59 + tmp_79 - 49.0) + tmp_408*(tmp_464 - ccomplex.complex[double](0, 1)*(370.0*tmp_45 - 351.0*tmp_59 + 91.0)) + tmp_413 + tmp_9*(-ccomplex.complex[double](0, 1)*(tmp_357 + 247110.0*tmp_45 + tmp_481 + 889.0) + 1994.9113350295187)) - tmp_473*tmp_476 - tmp_478*(tmp_253*(-ccomplex.complex[double](0, 1)*(270.0*tmp_45 - 270.0*tmp_59 + 133.0) + 298.45130209103036) + tmp_477 + ccomplex.complex[double](0, 1)*(-tmp_242 + 2090.0*tmp_45 - 329.0)) - tmp_479*(tmp_17*(-ccomplex.complex[double](0, 1)*(tmp_286 + 3740.0*tmp_45 + 1274.0) + 2858.8493147667118) + tmp_82*(ccomplex.complex[double](0, 1)*(-tmp_344 + 4990.0*tmp_45 - 259.0) + 581.19464091411175) - ccomplex.complex[double](0, 1)*(42180.0*tmp_45 - 6075.0*tmp_59 + tmp_79 + 854.0) + 1916.3715186897739) - tmp_482*(tmp_166*(ccomplex.complex[double](0, 1)*(1885400.0*tmp_45 + 303345.0*tmp_59 - 1015625.0*tmp_77 - 252.0) + 565.48667764616278) + tmp_185*(-tmp_407*(tmp_307 + 56.0) + 396080.0*tmp_64 + 2638.9378290154263) + tmp_212*(tmp_409 - ccomplex.complex[double](0, 1)*(tmp_190 + 792.0*tmp_45 + 196.0)) - tmp_405*(tmp_235 + 3943784.0*tmp_45 + 2127465.0*tmp_59 - 2156250.0*tmp_77 + 252.0) + tmp_411 + tmp_86*(-ccomplex.complex[double](0, 1)*(4788600.0*tmp_45 + tmp_463 - 1125090.0*tmp_59 + 1876.0) + 4209.7341558103229))) - 31104.0*tmp_472*tmp_473 + tmp_49*(self._gr_k_590*tmp_475 - 466560.0*ccomplex.complex[double](0, 1)*tmp_472)) + self._gr_k_599*tmp_375*tmp_468 + self._gr_k_751*tmp_380*(self._gr_k_737*tmp_139*tmp_37 + tmp_249*(self._gr_k_744*tmp_9 + self._gr_k_749*tmp_17 + self._gr_k_750) + tmp_376*(self._gr_k_742*tmp_9 + self._gr_k_743) + tmp_378*(self._gr_k_745*tmp_17 + self._gr_k_746*tmp_9 + self._gr_k_747*tmp_34 + self._gr_k_748) + tmp_470*(self._gr_k_738*tmp_9 + self._gr_k_739*tmp_17 + self._gr_k_740*tmp_34 + self._gr_k_741 + tmp_469)) + self._gr_k_87*tmp_390*(self._gr_k_78*tmp_471 + self._gr_k_85*tmp_34 + self._gr_k_86*tmp_472 + tmp_268*(self._gr_k_57*tmp_95 + self._gr_k_58*tmp_17 + self._gr_k_59*tmp_34 + self._gr_k_60*tmp_116 + self._gr_k_61*tmp_9 + self._gr_k_62*tmp_38 + self._gr_k_63) + tmp_383*(self._gr_k_77*tmp_9 + self._gr_k_84) + tmp_384*(self._gr_k_74*tmp_9 + self._gr_k_75*tmp_17 + self._gr_k_76) + tmp_385*(self._gr_k_70*tmp_17 + self._gr_k_71*tmp_9 + self._gr_k_72*tmp_34 + self._gr_k_73) + tmp_387*(self._gr_k_79*tmp_9 + self._gr_k_80*tmp_17 + self._gr_k_81*tmp_38 + self._gr_k_82*tmp_34 + self._gr_k_83) + tmp_388*(self._gr_k_64*tmp_95 + self._gr_k_65*tmp_34 + self._gr_k_66*tmp_17 + self._gr_k_67*tmp_9 + self._gr_k_68*tmp_38 + self._gr_k_69))
        # 41/math.m: hFactEccCorrResumExceptPAv5Flag[4,1]
        self.h41EccCorrResum = self._gr_k_164*tmp_22*tmp_397 + self._gr_k_447*tmp_513*(self._gr_k_35*(tmp_108*(-tmp_103*(-ccomplex.complex[double](0, 1)*(tmp_419 + 5315256.0*tmp_45 - 1255338.0*tmp_59 + 15488.0) + 22807.962665061899) + tmp_17*(2.0*ccomplex.complex[double](0, 1)*(-4411160.0*tmp_45 + 2547855.0*tmp_59 + 41344.0) - 122113.70644503526) + tmp_34*(2.0*ccomplex.complex[double](0, 1)*(137560.0*tmp_45 - 142155.0*tmp_59 + 12928.0) - 35342.917352885174) + tmp_38*(ccomplex.complex[double](0, 1)*(tmp_323 + 90280.0*tmp_45 + 1408.0) + 6644.4684623424127) + tmp_95*(ccomplex.complex[double](0, 1)*(tmp_322 - 131656.0*tmp_45 + 3200.0) + 4690.3978318095613) - ccomplex.complex[double](0, 1)*(14334072.0*tmp_45 + 1893942.0*tmp_59 - 7421875.0*tmp_77 - 22400.0) - 32986.722862692829) + tmp_109*(tmp_116*(-tmp_405*(tmp_512 + 484375.0*tmp_77 - 2304.0) + 11012656.0*tmp_64 + 17080.839257567706) - tmp_185*(tmp_62*(7711240.0*tmp_45 - 4625505.0*tmp_59 - 10368.0) + 43903.75733391736) - tmp_426*(-ccomplex.complex[double](0, 1)*(tmp_417 + 41320.0*tmp_45 + 384.0) + 1690.1768476313088) - tmp_466*(ccomplex.complex[double](0, 1)*(62802248.0*tmp_45 + 6372918.0*tmp_59 - 31328125.0*tmp_77 - 7296.0) + 10744.246875277093) - tmp_511*(tmp_62*(383880.0*tmp_45 - 234009.0*tmp_59 - 128.0) + 1297.4777659325846) - tmp_71*(-ccomplex.complex[double](0, 1)*(494270320.0*tmp_45 - 130203045.0*tmp_59 - 126171875.0*tmp_77 + 137472.0) + 201580.29261758908) + ccomplex.complex[double](0, 1)*(-167179229.0*tmp_111 + 963195632.0*tmp_45 + 438360093.0*tmp_59 - 512109375.0*tmp_77 + 57600.0) - 84823.001646924417) + 1536.0*tmp_34*tmp_505 + tmp_410*(tmp_17*(96.0*ccomplex.complex[double](0, 1)*(5.0*tmp_45 + 32.0) - 4005.5306333269864) + tmp_34*(tmp_477 - 32.0*ccomplex.complex[double](0, 1)*(5.0*tmp_45 - 16.0)) - tmp_405*(13712.0*tmp_45 - tmp_81 - 1792.0) + tmp_9*(3.0*ccomplex.complex[double](0, 1)*(tmp_308 + 10800.0*tmp_45 + 4352.0) - 19226.547039969535) - 13194.689145077132) + tmp_418*(tmp_140*(tmp_509*(16.0 - 17.0*tmp_45) + 9.4247779607693797) + tmp_38*(ccomplex.complex[double](0, 1)*(-tmp_420 + 27800.0*tmp_45 + 7808.0) + 11576.768928478388) - tmp_510*(tmp_203*(31069.0*tmp_45 - 16767.0*tmp_59 - 1232.0) + 3628.5395148962112) - tmp_86*(-ccomplex.complex[double](0, 1)*(98200.0*tmp_45 - 98415.0*tmp_59 + 20096.0) + 29421.015200868414) + ccomplex.complex[double](0, 1)*(tmp_419 + 4761240.0*tmp_45 - tmp_508 + 81536.0) - 120071.6712202019) + tmp_478*(tmp_406 + tmp_505*tmp_9 - 801.10612666539728 + 544.0*ccomplex.complex[double](0, 1)) + tmp_479*(tmp_9*(-tmp_506 + tmp_507 + 4544.0*ccomplex.complex[double](0, 1)) + tmp_92*(ccomplex.complex[double](0, 1)*tmp_503 + 28.274333882308139 + 16.0*ccomplex.complex[double](0, 1)) + ccomplex.complex[double](0, 1)*(tmp_308 + 10830.0*tmp_45 + 7456.0) - 10979.866324296327)) + self._gr_k_36*tmp_501 + self._gr_k_38*tmp_51*(tmp_137*(775.0*tmp_17 + 1346.0*tmp_9 + 1115.0) + tmp_249*(819.0*tmp_9 + 1523.0) + 4088.0*tmp_28 + tmp_35*(829.0*tmp_17 + 355.0*tmp_34 - 207.0*tmp_9 + 215.0)) + tmp_49*(self._gr_k_37*tmp_501 + self._gr_k_39*tmp_51*(tmp_137*(tmp_373 + 125.0) + tmp_249*(tmp_9 + 37.0) - 8.0*tmp_28 + tmp_470*(-13.0*tmp_17 + tmp_408 + 39.0*tmp_9 + 25.0)))) + self._gr_k_618*tmp_500*(tmp_249*(self._gr_k_613*tmp_17 + self._gr_k_614*tmp_9 + self._gr_k_615) + 1408.0*tmp_28*(self._gr_k_616*tmp_9 + self._gr_k_617) + tmp_35*(self._gr_k_604*tmp_34 + self._gr_k_605*tmp_38 + self._gr_k_606*tmp_9 + self._gr_k_607*tmp_17 + self._gr_k_608) + tmp_499*(self._gr_k_609*tmp_17 + self._gr_k_610*tmp_34 + self._gr_k_611*tmp_9 + self._gr_k_612))
        # 42/math.m: hFactEccCorrResumExceptPAv5Flag[4,2]
        self.h42EccCorrResum = self._gr_k_1007*tmp_181*tmp_515*tmp_9 + self._gr_k_1070*tmp_124*(self._gr_k_970*tmp_10*(self._gr_k_963*tmp_259 + self._gr_k_964*tmp_252 + self._gr_k_968*tmp_256 + self._gr_k_969*tmp_254 - tmp_257*(self._gr_k_965*tmp_9 + self._gr_k_966*tmp_17 + self._gr_k_967) + tmp_261*(self._gr_k_958*tmp_17 + self._gr_k_959*tmp_38 + self._gr_k_960*tmp_9 + self._gr_k_961*tmp_34 + self._gr_k_962)) + self._gr_k_971*(tmp_109*(tmp_116*(tmp_422 + ccomplex.complex[double](0, 1)*tmp_533 - 30576.0*tmp_64) - tmp_143*(tmp_484 - 3552.0*tmp_64 + 1203.2299863248908) + 135.0*ccomplex.complex[double](0, 1)*tmp_17*(tmp_421 + 1856272.0*tmp_45 - 1046875.0*tmp_77) - tmp_223*(-23059204.0*tmp_111 + 100781536.0*tmp_45 + 59980662.0*tmp_59 - 56484375.0*tmp_77) - tmp_535*(28431.0*tmp_113 - 54120.0*tmp_64 + 474.38049069205878) - tmp_94*(tmp_83 + ccomplex.complex[double](0, 1)*(6272288.0*tmp_45 - 1318761.0*tmp_59 - 1843750.0*tmp_77)) + 4.0*ccomplex.complex[double](0, 1)*(tmp_209 + 41887988.0*tmp_45 - tmp_534 + 21257640.0*tmp_59)) - tmp_245*(tmp_203*(-1647086.0*tmp_111 + tmp_234 + 6305796.0*tmp_45 + 4199040.0*tmp_59 + 945.0) + tmp_38*(6.0*ccomplex.complex[double](0, 1)*(620.0*tmp_45 + 1323.0) - 8042.4771931898707) - tmp_411 - tmp_447*(-ccomplex.complex[double](0, 1)*(3925580.0*tmp_45 + tmp_463 - 612360.0*tmp_59 + 5859.0) + 8765.0435035155231) + tmp_9*(-ccomplex.complex[double](0, 1)*(27013280.0*tmp_45 + 7413930.0*tmp_59 - 16609375.0*tmp_77 - 10584.0) - 15833.626974092558) - tmp_94*(tmp_62*(41264.0*tmp_45 - tmp_533 - 756.0) + 3433.760770373644) + tmp_95*(tmp_484 - 3072.0*tmp_64 + 323.5840433197487)) + 24.0*tmp_37*tmp_9*(tmp_34*(820.0*tmp_64 + 9927.4327853437466 - 6363.0*ccomplex.complex[double](0, 1)) + 10.0*tmp_38*(-12.0*tmp_64 + tmp_96) + tmp_405*(91684.0*tmp_45 + tmp_531 + 34992.0*tmp_59 - 441.0) + tmp_71*(-tmp_221*(tmp_344 + 497.0) + tmp_506 + 54860.0*tmp_64) + tmp_9*(-ccomplex.complex[double](0, 1)*(tmp_412 + 772180.0*tmp_45 + tmp_532 + 9261.0) + 13854.423602330988) + 3298.6722862692829) - tmp_371*(-225.0*tmp_17*tmp_528 - tmp_253*(-tmp_221*(tmp_242 + 763.0) + 27700.0*tmp_64 + 10273.007977238624) + tmp_34*(-ccomplex.complex[double](0, 1)*(100.0*tmp_45 + 441.0) + 251.32741228718346) + ccomplex.complex[double](0, 1)*(123940.0*tmp_45 + tmp_531 + 7497.0) - 11215.485773315562) - tmp_526*tmp_527 - tmp_529*(tmp_103*tmp_528 - 502.65482457436692 + 336.0*ccomplex.complex[double](0, 1)) - tmp_530*(tmp_17*(tmp_65 + 282.74333882308139 - 189.0*ccomplex.complex[double](0, 1)) - 1280.0*tmp_64 + tmp_9*(-tmp_65 - 1602.2122533307946 + 1071.0*ccomplex.complex[double](0, 1)) - 1319.4689145077132 + 882.0*ccomplex.complex[double](0, 1))) + self._gr_k_974*tmp_398*(-24.0*tmp_3*(tmp_258 + 1.0) + 12.0*tmp_395*(tmp_196 + 1.0) + tmp_443*(-tmp_279 - 198.0*tmp_9 - 99.0) + tmp_458) + self._gr_k_975*tmp_524 + tmp_49*(self._gr_k_972*tmp_251*tmp_525 + self._gr_k_973*tmp_524)) + self._gr_k_1071*tmp_523*(self._gr_k_1066*tmp_271 + self._gr_k_1067*tmp_273 + self._gr_k_1068*tmp_283*tmp_519 + self._gr_k_1069*tmp_139 + 24.0*tmp_109*(self._gr_k_1049*tmp_28 + self._gr_k_1050*tmp_276 + self._gr_k_1058*tmp_7 + self._gr_k_1059*tmp_56 + self._gr_k_1060*tmp_275 + self._gr_k_1061*tmp_139 + self._gr_k_1062*tmp_270) + tmp_25*(self._gr_k_1020*tmp_283 + self._gr_k_1021*tmp_17*tmp_522) + tmp_296*(self._gr_k_1051*tmp_9 + self._gr_k_1052) + tmp_302*(self._gr_k_1063*tmp_17 + self._gr_k_1064*tmp_9 + self._gr_k_1065) + tmp_311*(self._gr_k_1045*tmp_17 + self._gr_k_1046*tmp_9 + self._gr_k_1047*tmp_34 + self._gr_k_1048) + tmp_318*(self._gr_k_1053*tmp_34 + self._gr_k_1054*tmp_9 + self._gr_k_1055*tmp_38 + self._gr_k_1056*tmp_17 + self._gr_k_1057) + tmp_520*(self._gr_k_1039*tmp_17 + self._gr_k_1040*tmp_95 + self._gr_k_1041*tmp_9 + self._gr_k_1042*tmp_34 + self._gr_k_1043*tmp_38 + self._gr_k_1044) + tmp_521*(self._gr_k_1012*tmp_38 + self._gr_k_1013*tmp_116 + self._gr_k_1014*tmp_9 + self._gr_k_1015*tmp_148 + self._gr_k_1016*tmp_95 + self._gr_k_1017*tmp_17 + self._gr_k_1018*tmp_34 + self._gr_k_1019)) + self._gr_k_1072*tmp_518*(self._gr_k_1028*tmp_17 + self._gr_k_1029*tmp_445 + self._gr_k_1038*tmp_34 + tmp_30*(self._gr_k_1030*tmp_17 + self._gr_k_1031*tmp_9 + self._gr_k_1032) + tmp_438*(self._gr_k_1033*tmp_9 + self._gr_k_1034*tmp_38 + self._gr_k_1035*tmp_34 + self._gr_k_1036*tmp_17 + self._gr_k_1037) + tmp_494*(self._gr_k_1022*tmp_9 + self._gr_k_1023*tmp_95 + self._gr_k_1024*tmp_34 + self._gr_k_1025*tmp_17 + self._gr_k_1026*tmp_38 + self._gr_k_1027) + tmp_495*(self._gr_k_1008*tmp_9 + self._gr_k_1009*tmp_17 + self._gr_k_1010*tmp_34 + self._gr_k_1011)) + self._gr_k_603*tmp_169*tmp_517
        # 43/math.m: hFactEccCorrResumExceptPAv5Flag[4,3]
        self.h43EccCorrResum = self._gr_k_165*tmp_22*tmp_474 + self._gr_k_453*tmp_513*(self._gr_k_448*(tmp_109*(tmp_116*(tmp_509*(tmp_230 + 2888.0*tmp_45 + 896.0) + 15996.989792079227) + tmp_17*(-ccomplex.complex[double](0, 1)*(-5044200875.0*tmp_111 + 22908792400.0*tmp_45 + 12625430715.0*tmp_59 - 12394140625.0*tmp_77 + 314112.0) + 462568.10231456116) + tmp_185*(ccomplex.complex[double](0, 1)*(1131922040.0*tmp_45 + 152538390.0*tmp_59 - 590234375.0*tmp_77 - 40832.0) + 60130.083389708643) + tmp_38*(-ccomplex.complex[double](0, 1)*(960223760.0*tmp_45 - 254140335.0*tmp_59 - 244140625.0*tmp_77 + 151296.0) + 223979.84823768431) + tmp_386*(ccomplex.complex[double](0, 1)*(-3349349381.0*tmp_111 + 8198680168.0*tmp_45 + 3883790511.0*tmp_59 - 2132031250.0*tmp_77 - 36480.0) + 53721.234376385464) + tmp_511*(ccomplex.complex[double](0, 1)*(413880.0*tmp_45 - 238383.0*tmp_59 - 896.0) + 4344.822639914684) - ccomplex.complex[double](0, 1)*(-6254809085.0*tmp_111 + 17986982512.0*tmp_45 - tmp_545 + 2182238415.0*tmp_59 + 26880.0) + 39584.067435231395) + tmp_476*(-tmp_203*(tmp_504 - 15.0*tmp_59) + tmp_502) + tmp_482*(tmp_140*(tmp_203*(72307.0*tmp_45 - 39366.0*tmp_59 - 2096.0) + 6173.2295643039437) + tmp_156*(tmp_203*(-tmp_240 + 446441.0*tmp_45 + 112752.0*tmp_59 - 1424.0) + 4194.026192542374) + tmp_324*(-ccomplex.complex[double](0, 1)*(tmp_308 + 11000.0*tmp_45 + 6272.0) + 9628.9814832527163) - tmp_405*(tmp_112 + 20289992.0*tmp_45 - tmp_542 + 13100859.0*tmp_59 + 39808.0) + tmp_86*(-ccomplex.complex[double](0, 1)*(21048040.0*tmp_45 - 4084830.0*tmp_59 - 6640625.0*tmp_77 + 133504.0) + 196600.86826164926) + 293110.59457992771) - tmp_498*(-tmp_321*(ccomplex.complex[double](0, 1)*(331285880.0*tmp_45 + 61957710.0*tmp_59 - 183984375.0*tmp_77 - 147328.0) + 216958.38865691112) + tmp_38*(ccomplex.complex[double](0, 1)*(-4943800.0*tmp_45 + 2810295.0*tmp_59 + 50048.0) - 74879.860898312722) + tmp_9*(ccomplex.complex[double](0, 1)*(812946040.0*tmp_45 - tmp_543 + 485284365.0*tmp_59 - 438281250.0*tmp_77 + 178304.0) - 262574.31398703492) - tmp_94*(-ccomplex.complex[double](0, 1)*(16506584.0*tmp_45 + tmp_544 - 3862242.0*tmp_59 + 21120.0) + 31101.767270538953) + tmp_95*(-ccomplex.complex[double](0, 1)*(6376.0*tmp_45 - tmp_81 + 4480.0) + 2547.8316420613223) - ccomplex.complex[double](0, 1)*(-tmp_338 + 480670552.0*tmp_45 - tmp_543 + 224786421.0*tmp_59 - 40320.0) - 59376.101152847092) + 512.0*tmp_57*(tmp_203*(11325.0*tmp_45 - tmp_97 - 4304.0) + tmp_462*(-tmp_203*(243.0*tmp_45 + tmp_80 + 208.0) + 612.61056745000968) + 12676.326357234816) + 128.0*tmp_67*(tmp_17*(-ccomplex.complex[double](0, 1)*(tmp_417 + 33090.0*tmp_45 + 23008.0) + 33882.07676896592) - tmp_62*(tmp_357 + 178270.0*tmp_45 - 18225.0*tmp_59 + 15392.0) + tmp_66*(tmp_203*(108825.0*tmp_45 - tmp_480 - 18064.0) + 53202.871588543149) + 67999.772986951075) + 128.0*tmp_74*(tmp_253*(-ccomplex.complex[double](0, 1)*(1064480.0*tmp_45 + tmp_540 - 156735.0*tmp_59 + 26624.0) + 39207.07631680062) + tmp_34*(-ccomplex.complex[double](0, 1)*(16080.0*tmp_45 - 15795.0*tmp_59 + 9472.0) + 13948.671381938682) + tmp_539 + tmp_541*(ccomplex.complex[double](0, 1)*(-tmp_187 + 17872.0*tmp_45 - 1280.0) + 1884.9555921538759) + tmp_62*(612320.0*tmp_45 + tmp_540 + 206145.0*tmp_59 - 10752.0))) + self._gr_k_449*tmp_51*(-6.0*tmp_250*(275.0*tmp_17 + 10146.0*tmp_9 + 10415.0) - 47464.0*tmp_28 + tmp_35*(-7131.0*tmp_17 + 2195.0*tmp_34 - 21135.0*tmp_9 - 12705.0) - tmp_538*(2947.0*tmp_9 + 8195.0)) + self._gr_k_452*tmp_537 + tmp_49*(self._gr_k_450*tmp_51*(-tmp_137*(tmp_467 + 446.0*tmp_9 + 315.0) - tmp_249*(147.0*tmp_9 + 295.0) - 648.0*tmp_28 + tmp_35*(-147.0*tmp_17 + 35.0*tmp_34 - 255.0*tmp_9 - 105.0)) + self._gr_k_451*tmp_537)) + self._gr_k_633*tmp_500*(self._gr_k_624*tmp_129 + self._gr_k_629*tmp_10 + self._gr_k_630*tmp_272 + tmp_470*(self._gr_k_623*tmp_34 + self._gr_k_625*tmp_17 + self._gr_k_626*tmp_38 + self._gr_k_627*tmp_9 + self._gr_k_628) + tmp_499*(self._gr_k_619*tmp_34 + self._gr_k_620*tmp_9 + self._gr_k_621*tmp_17 + self._gr_k_622) - tmp_536*(self._gr_k_631*tmp_9 + self._gr_k_632))
        # 44/math.m: hFactEccCorrResumExceptPAv5Flag[4,4]
        self.h44EccCorrResum = self._gr_k_1122*tmp_515*tmp_546 + self._gr_k_154*tmp_124*(self._gr_k_148*tmp_9*(self._gr_k_137*tmp_127*tmp_445 + self._gr_k_138*tmp_557 + self._gr_k_139*tmp_443*(535.0*tmp_17 + 675.0*tmp_34 + 249.0*tmp_9 + 45.0) + self._gr_k_140*tmp_134*(tmp_462 + 7.0) + self._gr_k_141*tmp_558*(363.0*tmp_17 + 285.0*tmp_9 + 73.0) + self._gr_k_142*tmp_17 - 10.0*tmp_37*(self._gr_k_143*tmp_34 + self._gr_k_144*tmp_17 + self._gr_k_145*tmp_9 + self._gr_k_146*tmp_38 + self._gr_k_147)) + self._gr_k_151*tmp_560 + self._gr_k_152*tmp_251*(96.0*tmp_250*(61.0*tmp_17 + 96.0*tmp_9 + 48.0) + 7936.0*tmp_28 + tmp_35*(2097.0*tmp_17 + 1045.0*tmp_34 + 1995.0*tmp_9 + 735.0) + tmp_538*(967.0*tmp_9 + 845.0)) + self._gr_k_153*(tmp_109*(-ccomplex.complex[double](0, 1)*tmp_115*(-2957342913.0*tmp_111 + 6990275968.0*tmp_45 + 3305923875.0*tmp_59 - 1691281250.0*tmp_77) + tmp_116*(tmp_62*(-tmp_110 + 102272.0*tmp_45) + 18268.361280624648) - tmp_38*tmp_563*(15637248.0*tmp_45 + 1993086.0*tmp_59 - 8078125.0*tmp_77) + tmp_485*(-789777737.0*tmp_111 + 3467299712.0*tmp_45 + 1968280317.0*tmp_59 - 1883015625.0*tmp_77) + ccomplex.complex[double](0, 1)*tmp_73*(-5735976995.0*tmp_111 + 16626680320.0*tmp_45 + 1515912489.0*tmp_59 - 1260390625.0*tmp_77) + 3.0*tmp_95*(-tmp_405*(tmp_564 + 3636981.0*tmp_59) + 67823872.0*tmp_64 + 3427.4775850664644) - ccomplex.complex[double](0, 1)*(-15161426630.0*tmp_111 + 62574494976.0*tmp_45 - 13967096166.0*tmp_59 + 915765625.0*tmp_77)) - tmp_245*(tmp_103*(ccomplex.complex[double](0, 1)*(-779895221.0*tmp_111 + 1772106328.0*tmp_45 + 823346451.0*tmp_59 - 382093750.0*tmp_77 - 15498.0) + 23184.953783492674) + tmp_321*(-ccomplex.complex[double](0, 1)*(2708740440.0*tmp_45 + tmp_566 + 1717112115.0*tmp_59 - 1469468750.0*tmp_77 + 93366.0) + 139675.20937860221) + tmp_360*(ccomplex.complex[double](0, 1)*(980760440.0*tmp_45 + 187447770.0*tmp_59 - 547890625.0*tmp_77 - 116802.0) + 174735.3833926643) + tmp_38*(-ccomplex.complex[double](0, 1)*(176025880.0*tmp_45 + tmp_565 - 41924790.0*tmp_59 + 150822.0) + 225629.18438081895) + tmp_95*(-tmp_563*(tmp_99 + 98.0) + 558168.0*tmp_64 + 55948.623567780628) - ccomplex.complex[double](0, 1)*(-1787088310.0*tmp_111 + 5229772088.0*tmp_45 + tmp_487 + 356100462.0*tmp_59 + 13230.0) + 19792.033717615697) - 147456.0*tmp_34*(tmp_526 + tmp_65) - tmp_418*(tmp_101*(ccomplex.complex[double](0, 1)*(109972.0*tmp_45 - 56862.0*tmp_59 - 9387.0) + 14042.919161546376) + 12.0*tmp_17*(ccomplex.complex[double](0, 1)*(11214500.0*tmp_45 + 3105540.0*tmp_59 - 6875000.0*tmp_77 - 14427.0) + 21582.74153016188) + tmp_405*(-11529602.0*tmp_111 + 25232380.0*tmp_45 + 11337408.0*tmp_59 - 4656250.0*tmp_77 - 3969.0) + tmp_66*(-ccomplex.complex[double](0, 1)*(90545740.0*tmp_45 + tmp_483 + 64352475.0*tmp_59 - 48281250.0*tmp_77 + 47187.0) + 70591.586926162654) + tmp_94*(-ccomplex.complex[double](0, 1)*(tmp_291 + 2909668.0*tmp_45 - 564246.0*tmp_59 + 14553.0) + 21771.237089377267) + 29688.050576423546) - tmp_529*(tmp_9*(-tmp_62*(4860.0*tmp_59 + 9233.0) + 55540.0*tmp_64 + 41437.607100849373) - ccomplex.complex[double](0, 1)*(103460.0*tmp_45 + tmp_531 + 22617.0) + 33834.952879162073) - tmp_530*(tmp_266*(-tmp_221*(tmp_215 + 637.0) + 22060.0*tmp_64 + 8576.5479443001355) + tmp_386*(-ccomplex.complex[double](0, 1)*(tmp_308 + tmp_357 + 167140.0*tmp_45 + 11529.0) + 17247.343668207965) + ccomplex.complex[double](0, 1)*(409780.0*tmp_45 + tmp_532 + 233280.0*tmp_59 - 18963.0) + 28368.581661915833) - 48.0*tmp_74*(tmp_34*(-tmp_221*(tmp_416 + 9947.0) + 623860.0*tmp_64 + 133926.09482253289) - tmp_405*(tmp_235 + 2114660.0*tmp_45 + 1679616.0*tmp_59 - 1078125.0*tmp_77 + 9513.0) + tmp_71*(-ccomplex.complex[double](0, 1)*(3525940.0*tmp_45 + tmp_463 + tmp_562 + 70749.0) + 105840.25649944013) + tmp_73*(ccomplex.complex[double](0, 1)*(2225660.0*tmp_45 + 878445.0*tmp_59 - 1515625.0*tmp_77 - 19089.0) + 28557.077221131221) + 71157.073603808817)) + tmp_49*(self._gr_k_149*tmp_560 + self._gr_k_150*tmp_251*tmp_561)) + self._gr_k_210*tmp_523*(self._gr_k_169*tmp_555 + self._gr_k_178*tmp_556*(5287.0*tmp_3 + 6840.0) + self._gr_k_202*tmp_139 + self._gr_k_209*tmp_519*tmp_556 + 12.0*tmp_109*(self._gr_k_180*tmp_28 + self._gr_k_186*tmp_7 + self._gr_k_195*tmp_276 + self._gr_k_196*tmp_139 + self._gr_k_197*tmp_275 + self._gr_k_198*tmp_270 + self._gr_k_199*tmp_56) + tmp_25*(self._gr_k_200*tmp_556 + self._gr_k_201*tmp_17*tmp_554) + tmp_296*(self._gr_k_181*tmp_9 + self._gr_k_182) + tmp_302*(self._gr_k_183*tmp_9 + self._gr_k_184*tmp_17 + self._gr_k_185) + tmp_311*(self._gr_k_179*tmp_34 + self._gr_k_192*tmp_9 + self._gr_k_193*tmp_17 + self._gr_k_194) + tmp_318*(self._gr_k_187*tmp_9 + self._gr_k_188*tmp_17 + self._gr_k_189*tmp_34 + self._gr_k_190*tmp_38 + self._gr_k_191) + tmp_520*(self._gr_k_203*tmp_17 + self._gr_k_204*tmp_95 + self._gr_k_205*tmp_38 + self._gr_k_206*tmp_34 + self._gr_k_207*tmp_9 + self._gr_k_208) + tmp_521*(self._gr_k_170*tmp_95 + self._gr_k_171*tmp_17 + self._gr_k_172*tmp_9 + self._gr_k_173*tmp_38 + self._gr_k_174*tmp_148 + self._gr_k_175*tmp_116 + self._gr_k_176*tmp_34 + self._gr_k_177)) + self._gr_k_602*tmp_169*tmp_554 + self._gr_k_853*tmp_518*(self._gr_k_832*tmp_17 + self._gr_k_833*tmp_106*tmp_270 + self._gr_k_852*tmp_34 - tmp_30*(self._gr_k_835*tmp_17 + self._gr_k_846*tmp_9 + self._gr_k_847) - tmp_438*(self._gr_k_841*tmp_34 + self._gr_k_842*tmp_9 + self._gr_k_843*tmp_17 + self._gr_k_844*tmp_38 + self._gr_k_845) - tmp_494*(self._gr_k_834*tmp_95 + self._gr_k_840*tmp_34 + self._gr_k_848*tmp_9 + self._gr_k_849*tmp_38 + self._gr_k_850*tmp_17 + self._gr_k_851) - tmp_495*(self._gr_k_836*tmp_34 + self._gr_k_837*tmp_17 + self._gr_k_838*tmp_9 + self._gr_k_839))
        # 51/math.m: hFactEccCorrResumExceptPAv5Flag[5,1]
        self.h51EccCorrResum = self._gr_k_1166*tmp_395*tmp_567*(612.0*tmp_3 + 59.0*tmp_37 + 344.0) + self._gr_k_167*tmp_379*tmp_570 + self._gr_k_495*tmp_465*(self._gr_k_477*(self._gr_k_465*tmp_431*(90831.0*tmp_3 + 99104.0) + self._gr_k_466*tmp_430*(37793.0*tmp_3 + 169242.0) + self._gr_k_467*tmp_429 + self._gr_k_468*tmp_428 + self._gr_k_469*tmp_67*(2361520.0*tmp_3 + 353523.0*tmp_37 + 756592.0) + self._gr_k_470*tmp_391*(-10952.0*tmp_3 + 88439.0*tmp_37 + 18728.0) + self._gr_k_471*tmp_57*(251504.0*tmp_3 + 216057.0*tmp_37 + 24504.0) + self._gr_k_472*tmp_432*(-98192.0*tmp_3 + 10247.0*tmp_37 - 22352.0) + self._gr_k_473*tmp_74*(46443.0*tmp_3 + 53264.0) + self._gr_k_474*tmp_87*(59199.0*tmp_3 + 304646.0) + self._gr_k_475*tmp_107 + self._gr_k_476*tmp_109 + tmp_575*(self._gr_k_460*tmp_150 + self._gr_k_461*tmp_37 + self._gr_k_462*tmp_109 + self._gr_k_463*tmp_3 + self._gr_k_464)) + self._gr_k_490*tmp_395*(self._gr_k_482*tmp_440 + self._gr_k_484*tmp_443*(5481.0*tmp_17 + 1413.0*tmp_34 + 205.0*tmp_38 - 468.0*tmp_9 - 39.0) + self._gr_k_486*tmp_134*(1757.0*tmp_17 - 5807.0*tmp_9 - 446.0) + self._gr_k_487*tmp_441 + self._gr_k_488*tmp_433*(413.0*tmp_9 + 19555.0) - 32.0*tmp_17*(self._gr_k_483*tmp_9 + self._gr_k_485) + tmp_37*(self._gr_k_454*tmp_95 + self._gr_k_455*tmp_38 + self._gr_k_456*tmp_34 + self._gr_k_457*tmp_9 + self._gr_k_458*tmp_17 + self._gr_k_459) + tmp_495*(self._gr_k_478*tmp_34 + self._gr_k_479*tmp_9 + self._gr_k_480*tmp_17 + self._gr_k_481)) + self._gr_k_491*e*(tmp_106*(tmp_185*(-tmp_221*(tmp_512 + 743186.0) + 77831880.0*tmp_64 + 9271358.2835313222) + tmp_535*(tmp_62*(2000600.0*tmp_45 - 2066715.0*tmp_59 - 100998.0) + 978738.05848467137) - tmp_586*(-623422051.0*tmp_111 + 2063586448.0*tmp_45 + 1577179107.0*tmp_59 - 1212890625.0*tmp_77 + 488700.0) + tmp_592*(-tmp_221*(1639400.0*tmp_45 - 960093.0*tmp_59 - 11946.0) + 121507.37906289243) + tmp_593*(ccomplex.complex[double](0, 1)*(339885656.0*tmp_45 + 132440346.0*tmp_59 - 235234375.0*tmp_77 - 257382.0) + 312714.13273832802) + tmp_594*(-tmp_405*(8562105.0*tmp_59 + 7609375.0*tmp_77 - 58644.0) + 154691056.0*tmp_64 + 356372.84584526538) + tmp_71*(-tmp_62*(1650267920.0*tmp_45 - 78917895.0*tmp_59 - 697265625.0*tmp_77 + 5996892.0) + 21842378.455017808) + 4156327.0806992965) - 576.0*tmp_129*(tmp_266*(tmp_507 + 615.75216010359947 + 543.0*ccomplex.complex[double](0, 1)) + tmp_66*(1540.0*tmp_64 + 82026.984185229501 - 67513.0*ccomplex.complex[double](0, 1)) + ccomplex.complex[double](0, 1)*(tmp_297 - 50540.0*tmp_45 + 73667.0) - 89503.974700773209) + 24.0*tmp_132*(tmp_289*(-ccomplex.complex[double](0, 1)*(2240888.0*tmp_45 + 126846.0*tmp_59 - 1171875.0*tmp_77 + 59730.0) + 72570.790297924224) + tmp_321*(tmp_62*(4084360.0*tmp_45 + 841995.0*tmp_59 - 1693074.0) + 6174344.8296959681) + tmp_360*(ccomplex.complex[double](0, 1)*(1192520.0*tmp_45 - 995085.0*tmp_59 - 2084034.0) + 3975449.8836688641) + 21.0*tmp_38*(ccomplex.complex[double](0, 1)*(307960.0*tmp_45 + tmp_589 + 77106.0) + 101489.15067421827) + tmp_586*(22339528.0*tmp_45 + 12102858.0*tmp_59 - 17578125.0*tmp_77 - 190050.0) + tmp_591*(-tmp_62*(572152.0*tmp_45 - tmp_588 - 27150.0) + 98938.177439503358) + 1616349.4202719486) - tmp_146*tmp_527*tmp_581 + tmp_455*(tmp_497*(ccomplex.complex[double](0, 1)*(243908.0*tmp_45 + 214326.0*tmp_59 - 342633.0) + 416292.4425271835) - tmp_586*(5153960.0*tmp_45 + tmp_508 - 3515625.0*tmp_77 + 639654.0) + tmp_587*(4900.0*tmp_64 + 47412.91632797716 + 52671.0*ccomplex.complex[double](0, 1)) + tmp_590*(ccomplex.complex[double](0, 1)*(311000.0*tmp_45 + tmp_589 + 177018.0) + 215151.97288109699) + tmp_86*(-tmp_221*(tmp_322 + 191498.0) + 745640.0*tmp_64 + 2090808.4507803471) + 5440170.3345153014) + tmp_582*(ccomplex.complex[double](0, 1)*tmp_363 + tmp_581*tmp_9 - 21331.414117874696 + 17557.0*ccomplex.complex[double](0, 1)) + tmp_585*(tmp_253*(ccomplex.complex[double](0, 1)*(-tmp_322 + 77000.0*tmp_45 - 163986.0) + 199239.80609066469) + tmp_583*(1648.0*tmp_45 + tmp_81 - 10860.0) + tmp_584*(440.0*tmp_64 + 9220.5744382860432 + 7602.0*ccomplex.complex[double](0, 1)) - tmp_71*(tmp_414*(140.0*tmp_45 - 3439.0) + 53328.53529468674) + 461814.12007769961)) + self._gr_k_494*tmp_574 + tmp_49*(self._gr_k_489*tmp_444 + self._gr_k_492*e*(tmp_106*(315.0*tmp_116 - 8283.0*tmp_17 - 8212.0*tmp_34 - 1395.0*tmp_38 - 3318.0*tmp_9 + 1386.0*tmp_95 - 525.0) + tmp_146*tmp_576 - tmp_172*(tmp_115 - 746.0*tmp_9 + 407.0) + tmp_176*(-9354.0*tmp_17 - 3838.0*tmp_34 + 1491.0*tmp_38 - 5775.0*tmp_9 + 525.0*tmp_95 - 1225.0) - tmp_449*(tmp_9 - 97.0) + tmp_577*(114.0*tmp_17 + 49.0*tmp_34 - 453.0*tmp_9 - 350.0) + tmp_578*(-9522.0*tmp_17 + 1940.0*tmp_34 + 1141.0*tmp_38 - 12620.0*tmp_9 - 4123.0)) + self._gr_k_493*tmp_574)) + self._gr_k_691*tmp_572*(self._gr_k_668*tmp_170 - tmp_172*(self._gr_k_688*tmp_17 + self._gr_k_689*tmp_9 + self._gr_k_690) + tmp_173*(self._gr_k_684*tmp_9 + self._gr_k_685*tmp_17 + self._gr_k_686*tmp_34 + self._gr_k_687) + tmp_174*(self._gr_k_663*tmp_17 + self._gr_k_664*tmp_34 + self._gr_k_665*tmp_9 + self._gr_k_666*tmp_38 + self._gr_k_667) + tmp_176*(self._gr_k_671*tmp_95 + self._gr_k_672*tmp_38 + self._gr_k_673*tmp_17 + self._gr_k_674*tmp_34 + self._gr_k_675*tmp_9 + self._gr_k_676) + tmp_559*(self._gr_k_677*tmp_38 + self._gr_k_678*tmp_9 + self._gr_k_679*tmp_116 + self._gr_k_680*tmp_34 + self._gr_k_681*tmp_95 + self._gr_k_682*tmp_17 + self._gr_k_683) + tmp_571*(self._gr_k_669*tmp_9 + self._gr_k_670))
        # 52/math.m: hFactEccCorrResumExceptPAv5Flag[5,2]
        self.h52EccCorrResum = self._gr_k_793*tmp_596*(e*tmp_376*(self._gr_k_779*tmp_17 + self._gr_k_780*tmp_9 + self._gr_k_781) - tmp_171*(self._gr_k_785*tmp_9 + self._gr_k_786) + tmp_36*(self._gr_k_773*tmp_38 + self._gr_k_774*tmp_34 + self._gr_k_775*tmp_9 + self._gr_k_776*tmp_17 + self._gr_k_777) + tmp_494*(self._gr_k_787*tmp_38 + self._gr_k_788*tmp_95 + self._gr_k_789*tmp_17 + self._gr_k_790*tmp_34 + self._gr_k_791*tmp_9 + self._gr_k_792) + tmp_595*(self._gr_k_778*tmp_34 + self._gr_k_782*tmp_9 + self._gr_k_783*tmp_17 + self._gr_k_784)) + self._gr_k_794*tmp_453*tmp_525
        # 53/math.m: hFactEccCorrResumExceptPAv5Flag[5,3]
        self.h53EccCorrResum = self._gr_k_1006*tmp_572*(self._gr_k_983*tmp_603 + tmp_172*(self._gr_k_985*tmp_9 + self._gr_k_986*tmp_17 + self._gr_k_987) + tmp_173*(self._gr_k_988*tmp_17 + self._gr_k_989*tmp_34 + self._gr_k_990*tmp_9 + self._gr_k_991) + tmp_174*(self._gr_k_978*tmp_17 + self._gr_k_979*tmp_9 + self._gr_k_980*tmp_34 + self._gr_k_981*tmp_38 + self._gr_k_982) + tmp_176*(self._gr_k_992*tmp_95 + self._gr_k_993*tmp_34 + self._gr_k_994*tmp_9 + self._gr_k_995*tmp_38 + self._gr_k_996*tmp_17 + self._gr_k_997) + tmp_382*(self._gr_k_1004*tmp_9 + self._gr_k_1005) + tmp_559*(self._gr_k_1000*tmp_38 + self._gr_k_1001*tmp_9 + self._gr_k_1002*tmp_17 + self._gr_k_1003 + self._gr_k_984*tmp_95 + self._gr_k_998*tmp_34 + self._gr_k_999*tmp_116)) + self._gr_k_168*tmp_379*tmp_602 + self._gr_k_556*tmp_465*(self._gr_k_549*(self._gr_k_517*tmp_67*(2991440.0*tmp_3 + 522597.0*tmp_37 - 4068592.0) + self._gr_k_518*tmp_432*(4918480.0*tmp_3 + 1015061.0*tmp_37 - 8093936.0) + self._gr_k_519*tmp_429 + self._gr_k_524*tmp_57*(110392.0*tmp_3 + 477981.0*tmp_37 - 845448.0) + self._gr_k_525*tmp_109 + self._gr_k_526*tmp_107 + self._gr_k_527*tmp_428 + self._gr_k_528*tmp_391*(13792.0*tmp_3 + 130727.0*tmp_37 - 231928.0) + self._gr_k_529*tmp_431*(608781.0*tmp_3 - 790576.0) + self._gr_k_530*tmp_74*(29027.0*tmp_3 - 25472.0) + self._gr_k_531*tmp_430*(182441.0*tmp_3 - 211446.0) + self._gr_k_532*tmp_87*(11445.0*tmp_3 + 4898.0) + tmp_575*(self._gr_k_516*tmp_109 + self._gr_k_520*tmp_150 + self._gr_k_521*tmp_37 + self._gr_k_522*tmp_3 + self._gr_k_523)) + self._gr_k_551*e*(tmp_106*(14.0*tmp_116*(15.0*ccomplex.complex[double](0, 1)*(472.0*tmp_45 + tmp_85 + 3258.0) + 84929.81579714647) - tmp_13*(ccomplex.complex[double](0, 1)*(-8784733181.0*tmp_111 + 14564963048.0*tmp_45 + 6045246351.0*tmp_59 + 227343750.0*tmp_77 - 602730.0) + 732305.2475517808) + tmp_17*(ccomplex.complex[double](0, 1)*(-85751414875.0*tmp_111 + 58513288400.0*tmp_45 + 162028848555.0*tmp_59 - 32770390625.0*tmp_77 + 47690604.0) - 57943157.911691716) + 24.0*tmp_34*(ccomplex.complex[double](0, 1)*(1239276360.0*tmp_45 - 1214565030.0*tmp_59 + 265234375.0*tmp_77 - 13899714.0) + 16887882.636784221) + tmp_38*(-ccomplex.complex[double](0, 1)*(11900085680.0*tmp_45 - 4602574305.0*tmp_59 - 1708984375.0*tmp_77 - 56083212.0) - 68294501.501747601) + tmp_586*(-15190250635.0*tmp_111 + 47308040912.0*tmp_45 + tmp_545 - 5396127255.0*tmp_59 + 358380.0) - tmp_592*(-tmp_62*(629720.0*tmp_45 - 387099.0*tmp_59 + 72762.0) + 201008.52275463574) - 3047973.1925128174) + 192.0*tmp_129*(tmp_17*(-ccomplex.complex[double](0, 1)*(1384180.0*tmp_45 - 1377810.0*tmp_59 + 1279851.0) + 1554994.11574734) + tmp_9*(tmp_203*(4437020.0*tmp_45 - 2143260.0*tmp_59 - 449423.0) + 1092080.4382408839) - ccomplex.complex[double](0, 1)*(7977620.0*tmp_45 - 2296350.0*tmp_59 - 1093750.0*tmp_77 - 1548093.0) - 1880902.9376307451) + tmp_174*(7.0*tmp_38*(-ccomplex.complex[double](0, 1)*(305000.0*tmp_45 - 295245.0*tmp_59 + 248694.0) + 280088.69303079802) - tmp_497*(-ccomplex.complex[double](0, 1)*(58811396.0*tmp_45 - 25443558.0*tmp_59 - 10718750.0*tmp_77 + 1785927.0) + 2169866.6299079343) - tmp_583*(14000231.0*tmp_111 + 13437128.0*tmp_45 - tmp_542 - 16461549.0*tmp_59 - 502818.0) - tmp_587*(-ccomplex.complex[double](0, 1)*(9474724.0*tmp_45 - 5786802.0*tmp_59 + 1147359.0) + 1394018.908177399) - tmp_86*(ccomplex.complex[double](0, 1)*(-tmp_319 + 336721560.0*tmp_45 - 115889130.0*tmp_59 - 29231862.0) + 35516144.771804115) - 21381993.759597492) + tmp_571*(tmp_103*(-ccomplex.complex[double](0, 1)*(61236.0*tmp_45 - 61236.0*tmp_59 + 66427.0) + 80707.515270721788) + ccomplex.complex[double](0, 1)*(592900.0*tmp_45 - 306180.0*tmp_59 + 25159.0) - 30567.696519428688) + tmp_585*(-tmp_541*(-ccomplex.complex[double](0, 1)*(203672.0*tmp_45 - 120771.0*tmp_59 + 26426.0) + 32107.076919687687) + tmp_584*(tmp_539 - ccomplex.complex[double](0, 1)*(47840.0*tmp_45 + tmp_481 + 39096.0)) + tmp_586*(1413080.0*tmp_45 + tmp_540 + tmp_562 + 116202.0) + tmp_9*(-ccomplex.complex[double](0, 1)*(-tmp_312 + 20442240.0*tmp_45 - 6812505.0*tmp_59 - 2093808.0) - 2543936.067170871) - 988282.21696627716) - 559872.0*tmp_605*(tmp_579 - ccomplex.complex[double](0, 1)*(tmp_580 - 140.0*tmp_59)) - tmp_608*(tmp_317*(ccomplex.complex[double](0, 1)*(109528760.0*tmp_45 - 766641915.0*tmp_59 - tmp_607 - 107031250.0*tmp_77 - 2920254.0) + 3548051.9111112407) + tmp_321*(-ccomplex.complex[double](0, 1)*(2246860840.0*tmp_45 - 1332342270.0*tmp_59 - 128515625.0*tmp_77 + 21786246.0) + 26469865.893939234) + tmp_586*(tmp_338 + 720441032.0*tmp_45 + 244403811.0*tmp_59 + tmp_607 - 537570.0) + tmp_590*(-ccomplex.complex[double](0, 1)*(22434200.0*tmp_45 - 13920255.0*tmp_59 + 2392458.0) + 2928859.7070519604) + tmp_591*(-tmp_62*(22328.0*tmp_45 - tmp_533 + 16290.0) + 40523.403638654743) + tmp_94*(ccomplex.complex[double](0, 1)*(234424232.0*tmp_45 - 89006526.0*tmp_59 - 32265625.0*tmp_77 - 4329882.0) + 5260722.5621422524) + 4571959.7887692261)) + self._gr_k_554*tmp_604 + self._gr_k_555*tmp_395*(self._gr_k_533*tmp_440 + self._gr_k_540*tmp_443*(4350.0*tmp_17 + 1745.0*tmp_34 + 1390.0*tmp_38 + 1932.0*tmp_9 + 375.0) + self._gr_k_545*tmp_17*(134.0*tmp_9 - 123.0) + self._gr_k_546*tmp_134*(655.0*tmp_17 - 290.0*tmp_9 - 229.0) + self._gr_k_547*tmp_433*(49.0*tmp_17 - 49.0*tmp_9 + 1024.0) + self._gr_k_548*tmp_557 - tmp_3*tmp_510*(self._gr_k_541*tmp_9 + self._gr_k_542*tmp_17 + self._gr_k_543*tmp_34 + self._gr_k_544) + 5.0*tmp_37*(self._gr_k_534*tmp_34 + self._gr_k_535*tmp_17 + self._gr_k_536*tmp_9 + self._gr_k_537*tmp_95 + self._gr_k_538*tmp_38 + self._gr_k_539)) + tmp_49*(self._gr_k_550*tmp_444 + self._gr_k_552*tmp_604 + self._gr_k_553*e*(-tmp_172*(7071.0*tmp_17 + 4966.0*tmp_9 - 8553.0) + tmp_176*(40122.0*tmp_17 + 39870.0*tmp_34 + 15421.0*tmp_38 + 18823.0*tmp_9 + 315.0*tmp_95 + 3465.0) + tmp_449*(139.0 - 1835.0*tmp_9) - 192.0*tmp_454*(-1095.0*tmp_17 - tmp_606 - 1928.0*tmp_9 - 749.0) + tmp_559*(105.0*tmp_116 + 7319.0*tmp_17 - 51196.0*tmp_34 + 8607.0*tmp_38 + 2590.0*tmp_9 + 2814.0*tmp_95 + 385.0) + tmp_578*(161502.0*tmp_17 + 42260.0*tmp_34 - 1603.0*tmp_38 + 65780.0*tmp_9 + 16205.0) + 46656.0*tmp_605))) + self._gr_k_977*tmp_567*tmp_74*(7.0*tmp_3 + 248.0)
        # 54/math.m: hFactEccCorrResumExceptPAv5Flag[5,4]
        self.h54EccCorrResum = self._gr_k_594*tmp_453*tmp_561 + self._gr_k_772*tmp_596*(-tmp_171*(self._gr_k_770*tmp_9 + self._gr_k_771) + 8.0*tmp_35*tmp_7*(self._gr_k_752*tmp_38 + self._gr_k_753*tmp_9 + self._gr_k_754*tmp_17 + self._gr_k_755*tmp_34 + self._gr_k_756) + 3.0*tmp_37*(self._gr_k_758*tmp_34 + self._gr_k_759*tmp_38 + self._gr_k_760*tmp_9 + self._gr_k_761*tmp_95 + self._gr_k_762*tmp_17 + self._gr_k_763) - tmp_493*(self._gr_k_767*tmp_9 + self._gr_k_768*tmp_17 + self._gr_k_769) - tmp_595*(self._gr_k_757*tmp_34 + self._gr_k_764*tmp_17 + self._gr_k_765*tmp_9 + self._gr_k_766))
        # 55/math.m: hFactEccCorrResumExceptPAv5Flag[5,5]
        self.h55EccCorrResum = self._gr_k_166*tmp_379*tmp_611 + self._gr_k_515*tmp_465*(self._gr_k_502*(e*tmp_582*(tmp_9*(-ccomplex.complex[double](0, 1)*(3047660.0*tmp_45 + tmp_617 + 1716061.0) + 2084980.7964079381) + ccomplex.complex[double](0, 1)*(2187500.0*tmp_45 + 3265920.0*tmp_59 + tmp_617 - 1394243.0) + 1693978.1747421524) + 24.0*tmp_106*(2.0*tmp_139*(-ccomplex.complex[double](0, 1)*(-27642220795.0*tmp_111 + 84210550200.0*tmp_45 + 60178837005.0*tmp_59 - 44133906250.0*tmp_77 + 32387778.0) + 39350521.43736353) + 21.0*tmp_270*(ccomplex.complex[double](0, 1)*(1038312440.0*tmp_45 + 253538910.0*tmp_59 - 612421875.0*tmp_77 - 1973262.0) + 2397475.0176605148) + 35.0*tmp_275*(-ccomplex.complex[double](0, 1)*(10597192.0*tmp_45 - tmp_564 - 2103894.0*tmp_59 + 313854.0) + 381326.5162927291) + 21.0*tmp_28*(-ccomplex.complex[double](0, 1)*(-18966195290.0*tmp_111 + 61332819160.0*tmp_45 - 2342736270.0*tmp_59 - 1888984375.0*tmp_77 + 1045818.0) + 1270648.5646709278) + 2.0*tmp_56*(ccomplex.complex[double](0, 1)*(-173722278135.0*tmp_111 + 372928667160.0*tmp_45 + 154340515665.0*tmp_59 - 55749531250.0*tmp_77 - 26260566.0) + 31906077.821711012) + 35.0*tmp_7*(ccomplex.complex[double](0, 1)*(-3032285326.0*tmp_111 + 16019926312.0*tmp_45 - 6046420770.0*tmp_59 + 894384375.0*tmp_77 - 107514.0) + 130627.4225362636)) + tmp_109*(tmp_115*(-ccomplex.complex[double](0, 1)*(-3016979779345.0*tmp_111 + 9356497942064.0*tmp_45 + 46573514379.0*tmp_59 - 413930890625.0*tmp_77 + 8294868.0) + 10078103.569009913) + tmp_185*(ccomplex.complex[double](0, 1)*(-2433927806205.0*tmp_111 + 5413040115560.0*tmp_45 + 2404065578355.0*tmp_59 - 1028625718750.0*tmp_77 - 25050762.0) + 30436189.450949419) + tmp_324*(-ccomplex.complex[double](0, 1)*(-517938545845.0*tmp_111 + 1944975001200.0*tmp_45 + 1219139333685.0*tmp_59 - 1045356484375.0*tmp_77 + 63003204.0) + 76547669.606250471) - tmp_586*(-837308521245.0*tmp_111 + 6745322156496.0*tmp_45 - 3881450556594.0*tmp_59 + 1181889765625.0*tmp_77 - 684146626572.32269) + tmp_592*(ccomplex.complex[double](0, 1)*(4237126328.0*tmp_45 + 708706098.0*tmp_59 - 2294921875.0*tmp_77 - 1182654.0) + 1436901.6478988996) + tmp_593*(ccomplex.complex[double](0, 1)*(-362202446830.0*tmp_111 + 1782618815032.0*tmp_45 - 600932631942.0*tmp_59 + 80399765625.0*tmp_77 - 504990.0) + 613553.04524608662) + tmp_594*(tmp_405*(tmp_488 + 6554439.0*tmp_59 - 645084.0) - 157155664.0*tmp_64 + 3855623.2566756422) + 7481388.7452587336) + 7200000.0*tmp_34*(tmp_579 - ccomplex.complex[double](0, 1)*(tmp_580 - 140.0*tmp_77)) + tmp_410*(tmp_462*(-ccomplex.complex[double](0, 1)*(tmp_367 + 75012840.0*tmp_45 + 67604544.0*tmp_59 - 31390625.0*tmp_77 + 1126182.0) + 1368289.2643444985) + tmp_584*(-ccomplex.complex[double](0, 1)*(10615480.0*tmp_45 + tmp_544 - 1213785.0*tmp_59 + 1258674.0) + 1529264.4719144396) + tmp_586*(-86472015.0*tmp_111 + 181998720.0*tmp_45 - tmp_534 + 56687040.0*tmp_59 - 668976.0) + tmp_71*(ccomplex.complex[double](0, 1)*(189254240.0*tmp_45 + 100044315.0*tmp_59 - 140546875.0*tmp_77 - 6946056.0) + 8439323.1771913334) + 5689549.9593572591) + tmp_418*(42.0*tmp_17*(-ccomplex.complex[double](0, 1)*(1731734920.0*tmp_45 + tmp_566 + 1404808515.0*tmp_59 - 846468750.0*tmp_77 + 4736046.0) + 5754203.9361681371) + tmp_185*(ccomplex.complex[double](0, 1)*(2324066500.0*tmp_45 + 829288530.0*tmp_59 - 1523593750.0*tmp_77 - 20776809.0) + 25243419.537904314) - tmp_583*(-1072252986.0*tmp_111 + 3674110248.0*tmp_45 - 322464402.0*tmp_59 - 66953125.0*tmp_77 + 657030.0) + tmp_590*(-ccomplex.complex[double](0, 1)*(133603400.0*tmp_45 + tmp_565 - 21848130.0*tmp_59 + 7681278.0) + 9332603.6323130552) + tmp_82*(ccomplex.complex[double](0, 1)*(-10895473890.0*tmp_111 + 22905409380.0*tmp_45 + 8465264640.0*tmp_59 - 2435781250.0*tmp_77 - 13630929.0) + 16561314.080443562) + 27939754.264700826) + 576.0*tmp_67*(tmp_266*(-ccomplex.complex[double](0, 1)*(6848884.0*tmp_45 - 377622.0*tmp_59 - 3718750.0*tmp_77 + 1737419.0) + 2110930.3517265898) + tmp_301*(ccomplex.complex[double](0, 1)*(15856260.0*tmp_45 + 13063680.0*tmp_59 - 13562500.0*tmp_77 - 2320601.0) + 2819485.1588172317) - ccomplex.complex[double](0, 1)*(-57648010.0*tmp_111 + 82804260.0*tmp_45 + 78382080.0*tmp_59 - 25156250.0*tmp_77 + 5717247.0) + 6946344.1004258559)) + self._gr_k_512*tmp_7*(self._gr_k_503*tmp_129*(1773.0*tmp_9 + 914.0) + self._gr_k_506*tmp_454*(3245.0*tmp_17 + 2994.0*tmp_9 + 825.0) + self._gr_k_507*tmp_496*(3282.0*tmp_17 + 2860.0*tmp_34 + 1676.0*tmp_9 + 323.0) + self._gr_k_508*tmp_56 + self._gr_k_509*tmp_146*tmp_95 + self._gr_k_510*tmp_127*tmp_603 + self._gr_k_511*tmp_132*(4122.0*tmp_17 + 6020.0*tmp_34 + 5925.0*tmp_38 + 1500.0*tmp_9 + 225.0) + tmp_106*(self._gr_k_496*tmp_17 + self._gr_k_497*tmp_9 + self._gr_k_498*tmp_38 + self._gr_k_499*tmp_34 + self._gr_k_500*tmp_95 + self._gr_k_501)) + self._gr_k_513*tmp_12*(2817232.0*tmp_17 + tmp_36*(360936.0*tmp_17 + 158185.0*tmp_34 + 338337.0*tmp_9 + 115650.0) + tmp_37*(580950.0*tmp_17 + 427476.0*tmp_34 + 161105.0*tmp_38 + 398580.0*tmp_9 + 106785.0) + tmp_493*(167516.0*tmp_9 + 147157.0) + tmp_495*(483523.0*tmp_17 + 801866.0*tmp_9 + 384895.0)) + self._gr_k_514*tmp_613 + tmp_49*(self._gr_k_504*tmp_616 + self._gr_k_505*tmp_613)) + self._gr_k_662*tmp_572*(self._gr_k_634*tmp_109*tmp_275 - tmp_172*(self._gr_k_638*tmp_17 + self._gr_k_660*tmp_9 + self._gr_k_661) - tmp_173*(self._gr_k_640*tmp_17 + self._gr_k_641*tmp_34 + self._gr_k_642*tmp_9 + self._gr_k_643) - tmp_174*(self._gr_k_636*tmp_38 + self._gr_k_644*tmp_34 + self._gr_k_645*tmp_9 + self._gr_k_646*tmp_17 + self._gr_k_647) - tmp_176*(self._gr_k_635*tmp_95 + self._gr_k_639*tmp_9 + self._gr_k_651*tmp_34 + self._gr_k_652*tmp_17 + self._gr_k_653*tmp_38 + self._gr_k_654) - tmp_571*(self._gr_k_648*tmp_9 + self._gr_k_649) - tmp_612*(self._gr_k_637*tmp_116 + self._gr_k_650*tmp_17 + self._gr_k_655*tmp_34 + self._gr_k_656*tmp_95 + self._gr_k_657*tmp_38 + self._gr_k_658*tmp_9 + self._gr_k_659)) + self._gr_k_976*tmp_21*tmp_609
        # 61/math.m: hFactEccCorrResumExceptPAv5Flag[6,1]
        self.h61EccCorrResum = self._gr_k_1159*tmp_43*(1568.0*tmp_134 - tmp_171 + tmp_494*(-tmp_569 - tmp_601 - tmp_606 - 756.0*tmp_9 - 175.0) - tmp_495*(tmp_197 - tmp_568 + 399.0) + tmp_615*(15.0*tmp_34 - 113.0*tmp_9 - 50.0))
        # 62/math.m: hFactEccCorrResumExceptPAv5Flag[6,2]
        self.h62EccCorrResum = self._gr_k_1167*tmp_558*tmp_618*(self._gr_k_1111*tmp_37 + self._gr_k_1112*tmp_3 + self._gr_k_1113) + self._gr_k_596*tmp_179*(20.0*tmp_132*(2322.0*tmp_17 - 752.0*tmp_34 + 63.0*tmp_38 + 1008.0*tmp_9 + 175.0) + 11520.0*tmp_170 + tmp_172*(tmp_619 - 857.0) + tmp_174*(tmp_584 - tmp_621 + 3645.0*tmp_9 + 1267.0) + tmp_559*(3554.0*tmp_17 + 273.0*tmp_38 + tmp_622 + 973.0*tmp_9 + 105.0*tmp_95 + 125.0) + 1024.0*tmp_56 + tmp_620*(25.0*tmp_17 - 738.0*tmp_9 + 105.0)) + self._gr_k_831*tmp_624*(self._gr_k_795*tmp_471 + tmp_268*(self._gr_k_802*tmp_9 + self._gr_k_803*tmp_95 + self._gr_k_804*tmp_116 + self._gr_k_805*tmp_148 + self._gr_k_806*tmp_38 + self._gr_k_807*tmp_34 + self._gr_k_808*tmp_17 + self._gr_k_809) + tmp_387*(self._gr_k_801*tmp_95 + self._gr_k_817*tmp_38 + self._gr_k_818*tmp_17 + self._gr_k_819*tmp_34 + self._gr_k_820*tmp_9 + self._gr_k_821) + tmp_388*(self._gr_k_810*tmp_95 + self._gr_k_811*tmp_116 + self._gr_k_812*tmp_9 + self._gr_k_813*tmp_34 + self._gr_k_814*tmp_17 + self._gr_k_815*tmp_38 + self._gr_k_816) - tmp_450*(self._gr_k_822*tmp_9 + self._gr_k_823*tmp_17 + self._gr_k_824) - tmp_451*(self._gr_k_825*tmp_34 + self._gr_k_828*tmp_9 + self._gr_k_829*tmp_17 + self._gr_k_830) - tmp_576*(self._gr_k_826*tmp_9 + self._gr_k_827) + tmp_623*(self._gr_k_796*tmp_38 + self._gr_k_797*tmp_17 + self._gr_k_798*tmp_9 + self._gr_k_799*tmp_34 + self._gr_k_800))
        # 63/math.m: hFactEccCorrResumExceptPAv5Flag[6,3]
        self.h63EccCorrResum = self._gr_k_1158*tmp_43*(-11664.0*tmp_17 + 56.0*tmp_35*tmp_7*(521.0*tmp_17 + 542.0*tmp_9 + 165.0) + 3.0*tmp_37*(3294.0*tmp_17 + 105.0*tmp_38 + tmp_600 + 1820.0*tmp_9 + 385.0) - tmp_495*(-tmp_599 - 4554.0*tmp_9 - 2905.0) - tmp_614*(tmp_597 - 62.0))
        # 64/math.m: hFactEccCorrResumExceptPAv5Flag[6,4]
        self.h64EccCorrResum = self._gr_k_1164*tmp_624*(self._gr_k_1073*tmp_629 + tmp_387*(self._gr_k_1074*tmp_95 + self._gr_k_1086*tmp_17 + self._gr_k_1087*tmp_34 + self._gr_k_1088*tmp_9 + self._gr_k_1089*tmp_38 + self._gr_k_1090) - tmp_388*(self._gr_k_1091*tmp_116 + self._gr_k_1096*tmp_95 + self._gr_k_1098*tmp_17 + self._gr_k_1100*tmp_9 + self._gr_k_1101*tmp_38 + self._gr_k_1102*tmp_34 + self._gr_k_1103) + tmp_450*(self._gr_k_1076*tmp_9 + self._gr_k_1077*tmp_17 + self._gr_k_1078) + tmp_451*(self._gr_k_1081*tmp_17 + self._gr_k_1092*tmp_9 + self._gr_k_1093*tmp_34 + self._gr_k_1097) + tmp_576*(self._gr_k_1075*tmp_9 + self._gr_k_1079) + tmp_623*(self._gr_k_1080*tmp_17 + self._gr_k_1082*tmp_9 + self._gr_k_1083*tmp_34 + self._gr_k_1084*tmp_38 + self._gr_k_1085) - tmp_630*(self._gr_k_1094*tmp_116 + self._gr_k_1095*tmp_148 + self._gr_k_1099*tmp_9 + self._gr_k_1104*tmp_95 + self._gr_k_1105*tmp_17 + self._gr_k_1106*tmp_38 + self._gr_k_1107*tmp_34 + self._gr_k_1108)) + self._gr_k_1168*tmp_546*tmp_618*(self._gr_k_1109*tmp_3 + self._gr_k_1110) + self._gr_k_597*tmp_179*(tmp_172*(tmp_625 - 759.0) + tmp_174*(-26217.0*tmp_17 - tmp_627 - 20337.0*tmp_9 - 4835.0) + tmp_176*(-32742.0*tmp_17 - 42336.0*tmp_34 + 8195.0*tmp_38 - 13216.0*tmp_9 - 2205.0) + 65536.0*tmp_56 - 11520.0*tmp_603 + tmp_612*(-1210.0*tmp_17 - 2442.0*tmp_34 - tmp_628 - 365.0*tmp_9 + 231.0*tmp_95 - 49.0) + tmp_620*(-tmp_626 - 2306.0*tmp_9 - 1407.0))
        # 65/math.m: hFactEccCorrResumExceptPAv5Flag[6,5]
        self.h65EccCorrResum = self._gr_k_155*tmp_42*tmp_616
        # 66/math.m: hFactEccCorrResumExceptPAv5Flag[6,6]
        self.h66EccCorrResum = self._gr_k_1147*tmp_441*tmp_618 + self._gr_k_1156*tmp_624*(self._gr_k_1155*tmp_634 + 12.0*tmp_107*(self._gr_k_1148*tmp_9 + self._gr_k_1149*tmp_95 + self._gr_k_1150*tmp_34 + self._gr_k_1151*tmp_17 + self._gr_k_1152*tmp_116 + self._gr_k_1153*tmp_38 + self._gr_k_1154) + tmp_387*(self._gr_k_1135*tmp_95 + self._gr_k_1136*tmp_9 + self._gr_k_1137*tmp_34 + self._gr_k_1138*tmp_17 + self._gr_k_1139*tmp_38 + self._gr_k_1140) + tmp_450*(self._gr_k_1116*tmp_17 + self._gr_k_1117*tmp_9 + self._gr_k_1118) + tmp_576*(self._gr_k_1119*tmp_9 + self._gr_k_1120) + tmp_623*(self._gr_k_1115*tmp_38 + self._gr_k_1141*tmp_17 + self._gr_k_1142*tmp_34 + self._gr_k_1143*tmp_9 + self._gr_k_1144) + tmp_635*(self._gr_k_1123*tmp_9 + self._gr_k_1124*tmp_34 + self._gr_k_1125*tmp_17 + self._gr_k_1126) + tmp_636*(self._gr_k_1127*tmp_148 + self._gr_k_1128*tmp_17 + self._gr_k_1129*tmp_34 + self._gr_k_1130*tmp_95 + self._gr_k_1131*tmp_116 + self._gr_k_1132*tmp_9 + self._gr_k_1133*tmp_38 + self._gr_k_1134)) + self._gr_k_595*tmp_179*(3840.0*tmp_109*tmp_275 + tmp_172*(46437.0*tmp_9 + 24337.0) + tmp_173*(tmp_631 + 54814.0*tmp_9 + 15673.0) + tmp_174*(94231.0*tmp_17 + 76265.0*tmp_34 + 50371.0*tmp_9 + 10325.0) + tmp_176*(54046.0*tmp_17 + 73808.0*tmp_34 + tmp_632 + 21168.0*tmp_9 + 3465.0) + 248832.0*tmp_56 + tmp_612*(1430.0*tmp_17 + 2574.0*tmp_34 + 3003.0*tmp_38 + tmp_633 + 455.0*tmp_9 + 63.0))
        # 71/math.m: hFactEccCorrResumExceptPAv5Flag[7,1]
        self.h71EccCorrResum = self._gr_k_1121*tmp_395*tmp_637*(self._gr_k_944*tmp_37 + self._gr_k_945*tmp_3 + self._gr_k_946) + self._gr_k_158*tmp_123*(64.0*e*tmp_56*(tmp_9 - 321.0) - 9.0*tmp_109*(525.0*tmp_116 + 19683.0*tmp_17 + 66932.0*tmp_34 + 19683.0*tmp_38 + 4662.0*tmp_9 + 4662.0*tmp_95 + 525.0) + 32.0*tmp_28*tmp_35*(-795.0*tmp_17 + 931.0*tmp_34 + 17433.0*tmp_9 - 1505.0) + 64.0*tmp_34 - 322560.0*tmp_381 - tmp_387*(-147210.0*tmp_17 - 14076.0*tmp_34 + 12313.0*tmp_38 + 100612.0*tmp_9 + 31353.0) - tmp_635*(9.0*tmp_17 + 842.0*tmp_9 - 4135.0) - tmp_638*(21202.0*tmp_17 - 8506.0*tmp_34 + 4837.0*tmp_38 + 8015.0*tmp_9 + 875.0*tmp_95 + 1225.0))
        # 72/math.m: hFactEccCorrResumExceptPAv5Flag[7,2]
        self.h72EccCorrResum = self._gr_k_1161*tmp_456*(120.0*tmp_132*(-774.0*tmp_17 + 21.0*tmp_38 - 672.0*tmp_9 - 175.0) + tmp_174*(tmp_584 + tmp_621 - 10935.0*tmp_9 - 6335.0) + 576.0*tmp_454*(tmp_619 - 35.0) + tmp_559*(-10662.0*tmp_17 + 273.0*tmp_38 - tmp_622 - 4865.0*tmp_9 + 315.0*tmp_95 - 875.0) - 2048.0*tmp_56 - tmp_639*(41.0*tmp_9 - 857.0))
        # 73/math.m: hFactEccCorrResumExceptPAv5Flag[7,3]
        self.h73EccCorrResum = self._gr_k_1169*tmp_637*tmp_74*(self._gr_k_1145*tmp_3 + self._gr_k_1146) + self._gr_k_157*tmp_123*(139968.0*tmp_34 + tmp_356*(-41087.0*tmp_17 + 1687.0*tmp_34 + 5309.0*tmp_9 + 5355.0) + tmp_387*(561582.0*tmp_17 - 647020.0*tmp_34 + 5005.0*tmp_38 + 407572.0*tmp_9 + 86765.0) + tmp_450*(5729.0*tmp_9 - 14753.0) + 322560.0*tmp_471 + tmp_630*(231.0*tmp_116 + 15273.0*tmp_17 + 37532.0*tmp_34 + 85833.0*tmp_38 + 3906.0*tmp_9 - 4158.0*tmp_95 + 455.0) + tmp_635*(7645.0*tmp_17 - 63342.0*tmp_9 - 5011.0) + tmp_638*(tmp_147 + 45398.0*tmp_17 + 70626.0*tmp_34 - 23737.0*tmp_38 + 15645.0*tmp_9 + 2275.0))
        # 74/math.m: hFactEccCorrResumExceptPAv5Flag[7,4]
        self.h74EccCorrResum = self._gr_k_1162*tmp_456*(15.0*tmp_106*(6050.0*tmp_17 + 7326.0*tmp_34 + tmp_628 + 2555.0*tmp_9 + 231.0*tmp_95 + 441.0) - tmp_174*(-78651.0*tmp_17 - tmp_627 - 101685.0*tmp_9 - 33845.0) + 96.0*tmp_37*tmp_7*(5457.0*tmp_17 + 3528.0*tmp_34 + 3304.0*tmp_9 + 735.0) - 96.0*tmp_454*(-tmp_626 - 4612.0*tmp_9 - 4221.0) - 262144.0*tmp_56 - tmp_639*(tmp_625 - 1265.0))
        # 75/math.m: hFactEccCorrResumExceptPAv5Flag[7,5]
        self.h75EccCorrResum = self._gr_k_1114*tmp_41*tmp_609 + self._gr_k_159*tmp_123*(5000000.0*tmp_34 + tmp_387*(-3905790.0*tmp_17 - 3314036.0*tmp_34 + 1636635.0*tmp_38 - 1856244.0*tmp_9 - 337925.0) + tmp_450*(251361.0*tmp_9 + 29279.0) + tmp_623*(-300159.0*tmp_17 + 493703.0*tmp_34 - 402891.0*tmp_9 - 113533.0) - 322560.0*tmp_629 + tmp_635*(452603.0*tmp_17 - 12610.0*tmp_9 - 77493.0) + tmp_636*(-11115.0*tmp_17 - 24596.0*tmp_34 - 37323.0*tmp_38 - tmp_641 - 2982.0*tmp_9 - 54054.0*tmp_95 - 357.0) + tmp_638*(-138434.0*tmp_17 - 229686.0*tmp_34 - 224301.0*tmp_38 - tmp_640 - 46039.0*tmp_9 - 6545.0))
        # 76/math.m: hFactEccCorrResumExceptPAv5Flag[7,6]
        self.h76EccCorrResum = self._gr_k_557*tmp_456*(tmp_172*(232185.0*tmp_9 + 170359.0) + tmp_174*(471155.0*tmp_17 + 228795.0*tmp_34 + 352597.0*tmp_9 + 92925.0) + 1492992.0*tmp_56 + tmp_577*(tmp_631 + 82221.0*tmp_9 + 31346.0) + tmp_608*(162138.0*tmp_17 + 147616.0*tmp_34 + tmp_632 + 84672.0*tmp_9 + 17325.0) + tmp_612*(10010.0*tmp_17 + 12870.0*tmp_34 + 9009.0*tmp_38 + tmp_633 + 4095.0*tmp_9 + 693.0))
        # 77/math.m: hFactEccCorrResumExceptPAv5Flag[7,7]
        self.h77EccCorrResum = self._gr_k_156*tmp_123*(7529536.0*tmp_34 + tmp_387*(4822314.0*tmp_17 + 6050364.0*tmp_34 + 3701495.0*tmp_38 + 1967420.0*tmp_9 + 329175.0) + tmp_450*(403639.0*tmp_9 + 225033.0) + tmp_623*(1353129.0*tmp_17 + 957775.0*tmp_34 + 768213.0*tmp_9 + 162547.0) + 46080.0*tmp_634 + tmp_635*(788343.0*tmp_17 + 818326.0*tmp_9 + 244615.0) + tmp_636*(15015.0*tmp_116 + 6825.0*tmp_17 + 14300.0*tmp_34 + 19305.0*tmp_38 + 1890.0*tmp_9 + 18018.0*tmp_95 + 231.0) + tmp_638*(109162.0*tmp_17 + 186366.0*tmp_34 + 191625.0*tmp_38 + 35651.0*tmp_9 + 118335.0*tmp_95 + 5005.0))
        # 81/math.m: hFactEccCorrResumExceptPAv5Flag[8,1]
        self.h81EccCorrResum = 1.0
        # 82/math.m: hFactEccCorrResumExceptPAv5Flag[8,2]
        self.h82EccCorrResum = self._gr_k_560*tmp_370*(8192.0*tmp_139 - 1290240.0*tmp_271 + tmp_642*(253.0*tmp_9 - 6817.0) + tmp_643*(157.0*tmp_17 - 18912.0*tmp_9 + 28227.0) + tmp_644*(-79629.0*tmp_17 + 55.0*tmp_34 + 409041.0*tmp_9 + 45205.0) + tmp_645*(73950.0*tmp_17 - 4567.0*tmp_34 + 385.0*tmp_38 - 2117.0*tmp_9 - 5635.0) - tmp_646*(1071142.0*tmp_17 - 1717518.0*tmp_34 - 6105.0*tmp_38 + 714405.0*tmp_9 + 28105.0*tmp_95 + 137235.0) - tmp_647*(1925.0*tmp_116 + 94007.0*tmp_17 + 169184.0*tmp_34 - 70423.0*tmp_38 + 28420.0*tmp_9 + 10780.0*tmp_95 + 3675.0) - tmp_648*(2079.0*tmp_116 + 231.0*tmp_148 + 10629.0*tmp_17 + 30445.0*tmp_34 + 81895.0*tmp_38 + 2373.0*tmp_9 + 4455.0*tmp_95 + 245.0))
        # 83/math.m: hFactEccCorrResumExceptPAv5Flag[8,3]
        self.h83EccCorrResum = 1.0
        # 84/math.m: hFactEccCorrResumExceptPAv5Flag[8,4]
        self.h84EccCorrResum = self._gr_k_559*tmp_370*(524288.0*tmp_139 + tmp_29*tmp_37*(42510.0*tmp_17 - 712384.0*tmp_34 + 31885.0*tmp_38 + 204616.0*tmp_9 + 58205.0) + 322560.0*tmp_555 + tmp_642*(26207.0*tmp_9 - 32768.0) + tmp_643*(34413.0*tmp_17 - 127773.0*tmp_9 - 27152.0) + tmp_644*(-825651.0*tmp_17 + 92950.0*tmp_34 - 183156.0*tmp_9 + 29845.0) + tmp_646*(1374912.0*tmp_17 + 1206842.0*tmp_34 - 1394250.0*tmp_38 + 589095.0*tmp_9 + 15015.0*tmp_95 + 96530.0) - tmp_647*(-52802.0*tmp_17 - 102734.0*tmp_34 - 121017.0*tmp_38 - tmp_640 - 15295.0*tmp_9 - 1925.0) - tmp_648*(-4275.0*tmp_17 - 11180.0*tmp_34 - 20735.0*tmp_38 - tmp_641 - 994.0*tmp_9 - 38610.0*tmp_95 - 105.0))
        # 85/math.m: hFactEccCorrResumExceptPAv5Flag[8,5]
        self.h85EccCorrResum = 1.0
        # 86/math.m: hFactEccCorrResumExceptPAv5Flag[8,6]
        self.h86EccCorrResum = self._gr_k_561*tmp_370*(53747712.0*tmp_139 - 1290240.0*tmp_150*tmp_276 + tmp_311*(384007.0*tmp_17 + 5742835.0*tmp_34 - 2326043.0*tmp_9 - 851375.0) + tmp_642*(3085723.0*tmp_9 + 725953.0) + tmp_643*(4878567.0*tmp_17 + 1410368.0*tmp_9 - 278823.0) + tmp_645*(-1578570.0*tmp_17 - 706217.0*tmp_34 + 1148665.0*tmp_38 - 900467.0*tmp_9 - 179795.0) + tmp_646*(-29256282.0*tmp_17 - 41050302.0*tmp_34 - 25807145.0*tmp_38 - 10631075.0*tmp_9 + 11876865.0*tmp_95 - 1594845.0) + tmp_647*(240065.0*tmp_116 - 711157.0*tmp_17 - 1429664.0*tmp_34 - 1785867.0*tmp_38 - 201740.0*tmp_9 - 1403220.0*tmp_95 - 25025.0) + tmp_649*(-39039.0*tmp_116 + 2145.0*tmp_148 - 6405.0*tmp_17 - 15925.0*tmp_34 - 26455.0*tmp_38 - 1533.0*tmp_9 - 32175.0*tmp_95 - 165.0))
        # 87/math.m: hFactEccCorrResumExceptPAv5Flag[8,7]
        self.h87EccCorrResum = 1.0
        # 88/math.m: hFactEccCorrResumExceptPAv5Flag[8,8]
        self.h88EccCorrResum = self._gr_k_558*tmp_370*(268435456.0*tmp_139 + 645120.0*tmp_150*ccomplex.exp(15*tmp_0) + tmp_37*tmp_492*(49516950.0*tmp_17 + 58521916.0*tmp_34 + 31316705.0*tmp_38 + 20842316.0*tmp_9 + 3548545.0) + tmp_642*(16260029.0*tmp_9 + 9492289.0) + tmp_643*(27447851.0*tmp_17 + 30330654.0*tmp_9 + 9380151.0) + tmp_644*(162524283.0*tmp_17 + 105285005.0*tmp_34 + 96469623.0*tmp_9 + 20925065.0) + tmp_646*(123478534.0*tmp_17 + 202639854.0*tmp_34 + 191695035.0*tmp_38 + 41170815.0*tmp_9 + 94465315.0*tmp_95 + 5845455.0) + tmp_647*(2441775.0*tmp_116 + 2178649.0*tmp_17 + 4456908.0*tmp_34 + 5721909.0*tmp_38 + 610610.0*tmp_9 + 4722690.0*tmp_95 + 75075.0) + tmp_649*(51051.0*tmp_116 + 36465.0*tmp_148 + 16065.0*tmp_17 + 38675.0*tmp_34 + 60775.0*tmp_38 + 3927.0*tmp_9 + 65637.0*tmp_95 + 429.0))


    cpdef ccomplex.complex[double] get(self, int l, int m):
        if l == 1:
            if m == 1:
                raise RuntimeError("Unsupported mode 1, 1")
            raise RuntimeError(f"Unsupported mode 1, {m}")
        elif l == 2:
            if m == 1:
                return self.h21EccCorrResum
            elif m == 2:
                return self.h22EccCorrResum
            raise RuntimeError(f"Unsupported mode 2, {m}")
        elif l == 3:
            if m == 1:
                return self.h31EccCorrResum
            elif m == 2:
                return self.h32EccCorrResum
            elif m == 3:
                return self.h33EccCorrResum
            raise RuntimeError(f"Unsupported mode 3, {m}")
        elif l == 4:
            if m == 1:
                return self.h41EccCorrResum
            elif m == 2:
                return self.h42EccCorrResum
            elif m == 3:
                return self.h43EccCorrResum
            elif m == 4:
                return self.h44EccCorrResum
            raise RuntimeError(f"Unsupported mode 4, {m}")
        elif l == 5:
            if m == 1:
                return self.h51EccCorrResum
            elif m == 2:
                return self.h52EccCorrResum
            elif m == 3:
                return self.h53EccCorrResum
            elif m == 4:
                return self.h54EccCorrResum
            elif m == 5:
                return self.h55EccCorrResum
            raise RuntimeError(f"Unsupported mode 5, {m}")
        elif l == 6:
            if m == 1:
                return self.h61EccCorrResum
            elif m == 2:
                return self.h62EccCorrResum
            elif m == 3:
                return self.h63EccCorrResum
            elif m == 4:
                return self.h64EccCorrResum
            elif m == 5:
                return self.h65EccCorrResum
            elif m == 6:
                return self.h66EccCorrResum
            raise RuntimeError(f"Unsupported mode 6, {m}")
        elif l == 7:
            if m == 1:
                return self.h71EccCorrResum
            elif m == 2:
                return self.h72EccCorrResum
            elif m == 3:
                return self.h73EccCorrResum
            elif m == 4:
                return self.h74EccCorrResum
            elif m == 5:
                return self.h75EccCorrResum
            elif m == 6:
                return self.h76EccCorrResum
            elif m == 7:
                return self.h77EccCorrResum
            raise RuntimeError(f"Unsupported mode 7, {m}")
        elif l == 8:
            if m == 1:
                return ccomplex.complex[double](self.h81EccCorrResum, 0)
            elif m == 2:
                return self.h82EccCorrResum
            elif m == 3:
                return ccomplex.complex[double](self.h83EccCorrResum, 0)
            elif m == 4:
                return self.h84EccCorrResum
            elif m == 5:
                return ccomplex.complex[double](self.h85EccCorrResum, 0)
            elif m == 6:
                return self.h86EccCorrResum
            elif m == 7:
                return ccomplex.complex[double](self.h87EccCorrResum, 0)
            elif m == 8:
                return self.h88EccCorrResum
            raise RuntimeError(f"Unsupported mode 8, {m}")
        raise RuntimeError(f"Unsupported mode {l}, {m}")

    def initialize(self, *, flagMemory, flagPA, flagPN1, flagPN12, flagPN2, flagPN3, flagPN32, flagPN52, flagTail, nu):
        ret = self._initialize(flagMemory, flagPA, flagPN1, flagPN12, flagPN2, flagPN3, flagPN32, flagPN52, flagTail, nu)
        self._initialized = True
        return ret

    def compute(self, *, e, x, z):
        if not self._initialized:
            raise RuntimeError("Instance has not been initialized yet")
        return self._compute(e, x, z)