# cython: language_level=3
# distutils: language = c++

cimport libc.math as cmath
from libcpp cimport bool
cimport libcpp.complex as ccomplex


cdef class BaseCoupledExpressionsCalculation:
    cdef void _initialize(self)
    cdef void _compute(self)
    cpdef double get(self, str radial_or_azimuthal)


cdef class edot_zdot_xdot_flags(BaseCoupledExpressionsCalculation):

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

    cdef:
        readonly double edotExpResumParser
        readonly double xdotExpResumParser
        readonly double zdotParser

    cdef void _initialize(self
        , double chiA=*
        , double chiS=*
        , double delta=*
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

    cpdef double get(self, str radial_or_azimuthal)

    cpdef void compute(self
        , double e
        , double x
        , double z
    )
