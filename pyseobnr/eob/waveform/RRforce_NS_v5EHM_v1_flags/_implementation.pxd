# cython: language_level=3
# distutils: language = c++

cimport libc.math as cmath
from libcpp cimport bool
cimport libcpp.complex as ccomplex


cdef class BaseForcesCalculation:
    cdef void _initialize(self)
    cdef void _compute(self)
    cpdef double get(self, str radial_or_azimuthal)


cdef class RRforce_ecc_corr_NS_v5EHM_v1_flags(BaseForcesCalculation):

    cdef:
        bool _initialized

    cdef:
        double _gr_k_0
        double _gr_k_1
        double _gr_k_2
        double _gr_k_3
        double _gr_k_4
        double _gr_k_5
        double _gr_k_6
        double _gr_k_7
        double _gr_k_8
        double _gr_k_9
        double _gr_k_10
        double _gr_k_11
        double _gr_k_12
        double _gr_k_13
        double _gr_k_14
        double _gr_k_15
        double _gr_k_16
        double _gr_k_17
        double _gr_k_18
        double _gr_k_19
        double _gr_k_20
        double _gr_k_21
        double _gr_k_22
        double _gr_k_23
        double _gr_k_24
        double _gr_k_25
        double _gr_k_26
        double _gr_k_27
        double _gr_k_28
        double _gr_k_29
        double _gr_k_30
        double _gr_k_31
        double _gr_k_32
        double _gr_k_33
        double _gr_k_34
        double _gr_k_35
        double _gr_k_36
        double _gr_k_37
        double _gr_k_38
        double _gr_k_39
        double _gr_k_40
        double _gr_k_41
        double _gr_k_42
        double _gr_k_43
        double _gr_k_44
        double _gr_k_45
        double _gr_k_46
        double _gr_k_47
        double _gr_k_48
        double _gr_k_49
        double _gr_k_50
        double _gr_k_51
        double _gr_k_52
        double _gr_k_53
        double _gr_k_54
        double _gr_k_55
        double _gr_k_56
        double _gr_k_57
        double _gr_k_58
        double _gr_k_59
        double _gr_k_60
        double _gr_k_61
        double _gr_k_62
        double _gr_k_63
        double _gr_k_64
        double _gr_k_65
        double _gr_k_66
        double _gr_k_67
        double _gr_k_68
        double _gr_k_69
        double _gr_k_70
        double _gr_k_71
        double _gr_k_72
        double _gr_k_73
        double _gr_k_74
        double _gr_k_75
        double _gr_k_76
        double _gr_k_77
        double _gr_k_78
        double _gr_k_79
        double _gr_k_80
        double _gr_k_81
        double _gr_k_82
        double _gr_k_83
        double _gr_k_84
        double _gr_k_85
        double _gr_k_86
        double _gr_k_87
        double _gr_k_88
        double _gr_k_89
        double _gr_k_90
        double _gr_k_91
        double _gr_k_92
        double _gr_k_93
        double _gr_k_94
        double _gr_k_95
        double _gr_k_96
        double _gr_k_97
        double _gr_k_98
        double _gr_k_99
        double _gr_k_100
        double _gr_k_101
        double _gr_k_102
        double _gr_k_103
        double _gr_k_104
        double _gr_k_105
        double _gr_k_106
        double _gr_k_107
        double _gr_k_108
        double _gr_k_109
        double _gr_k_110
        double _gr_k_111
        double _gr_k_112
        double _gr_k_113
        double _gr_k_114
        double _gr_k_115
        double _gr_k_116
        double _gr_k_117
        double _gr_k_118
        double _gr_k_119
        double _gr_k_120
        double _gr_k_121
        double _gr_k_122
        double _gr_k_123
        double _gr_k_124
        double _gr_k_125
        double _gr_k_126
        double _gr_k_127
        double _gr_k_128
        double _gr_k_129
        double _gr_k_130
        double _gr_k_131
        double _gr_k_132
        double _gr_k_133
        double _gr_k_134
        double _gr_k_135
        double _gr_k_136
        double _gr_k_137
        double _gr_k_138
        double _gr_k_139
        double _gr_k_140
        double _gr_k_141
        double _gr_k_142
        double _gr_k_143
        double _gr_k_144
        double _gr_k_145
        double _gr_k_146
        double _gr_k_147
        double _gr_k_148
        double _gr_k_149
        double _gr_k_150
        double _gr_k_151
        double _gr_k_152
        double _gr_k_153
        double _gr_k_154
        double _gr_k_155
        double _gr_k_156
        double _gr_k_157
        double _gr_k_158
        double _gr_k_159
        double _gr_k_160
        double _gr_k_161
        double _gr_k_162
        double _gr_k_163
        double _gr_k_164
        double _gr_k_165
        double _gr_k_166
        double _gr_k_167
        double _gr_k_168
        double _gr_k_169
        double _gr_k_170
        double _gr_k_171
        double _gr_k_172
        double _gr_k_173
        double _gr_k_174
        double _gr_k_175
        double _gr_k_176
        double _gr_k_177
        double _gr_k_178
        double _gr_k_179
        double _gr_k_180
        double _gr_k_181
        double _gr_k_182
        double _gr_k_183
        double _gr_k_184
        double _gr_k_185
        double _gr_k_186
        double _gr_k_187
        double _gr_k_188
        double _gr_k_189
        double _gr_k_190
        double _gr_k_191
        double _gr_k_192
        double _gr_k_193
        double _gr_k_194
        double _gr_k_195
        double _gr_k_196
        double _gr_k_197
        double _gr_k_198
        double _gr_k_199
        double _gr_k_200
        double _gr_k_201
        double _gr_k_202
        double _gr_k_203
        double _gr_k_204
        double _gr_k_205
        double _gr_k_206
        double _gr_k_207
        double _gr_k_208
        double _gr_k_209
        double _gr_k_210
        double _gr_k_211
        double _gr_k_212
        double _gr_k_213
        double _gr_k_214
        double _gr_k_215
        double _gr_k_216
        double _gr_k_217
        double _gr_k_218
        double _gr_k_219
        double _gr_k_220
        double _gr_k_221
        double _gr_k_222
        double _gr_k_223
        double _gr_k_224
        double _gr_k_225
        double _gr_k_226
        double _gr_k_227
        double _gr_k_228
        double _gr_k_229
        double _gr_k_230
        double _gr_k_231
        double _gr_k_232
        double _gr_k_233
        double _gr_k_234
        double _gr_k_235
        double _gr_k_236
        double _gr_k_237
        double _gr_k_238
        double _gr_k_239
        double _gr_k_240
        double _gr_k_241
        double _gr_k_242
        double _gr_k_243
        double _gr_k_244
        double _gr_k_245
        double _gr_k_246
        double _gr_k_247
        double _gr_k_248
        double _gr_k_249
        double _gr_k_250

    cdef:
        readonly double FphiCorrMultParser
        readonly double FrCorrMultParser

    cdef void _initialize(self
        , int flagPN1=*
        , int flagPN2=*
        , int flagPN3=*
        , int flagPN32=*
        , int flagPN52=*
        , double nu=*
    )

    cdef void _compute(self
        , double e=*
        , double x=*
        , double z=*
    )

    cpdef void compute(self
        , double e
        , double x
        , double z
    )

    cpdef double get(self, str radial_or_azimuthal)
