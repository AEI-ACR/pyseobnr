# cython: language_level=3
# distutils: language = c++

from abc import abstractmethod

cimport libc.math as cmath
from libcpp cimport bool
cimport libcpp.complex as ccomplex

from ._implementation cimport edot_zdot_xdot_flags

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


cdef class edot_zdot_xdot_flags(BaseCoupledExpressionsCalculation):
    """
    Secular evolution equations for the parameters (e, z, x).
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
            double tmp_init_0 = chiA*delta
            double tmp_init_1 = 120.0*tmp_init_0
            double tmp_init_2 = cmath.pow(nu, 2)
            double tmp_init_3 = 120.0*chiS*(55.0*nu + 10.0*tmp_init_2 - 192.0) - tmp_init_1*(11.0*nu + 192.0)
            double tmp_init_4 = 27.0*nu
            double tmp_init_5 = 15.0*tmp_init_0
            double tmp_init_6 = 18.0*tmp_init_2
            double tmp_init_7 = 377.0*nu
            double tmp_init_8 = 480.0*tmp_init_0
            double tmp_init_9 = 60.0*tmp_init_0
            double tmp_init_10 = tmp_init_0*(nu - 3.0)
            double tmp_init_11 = 8.0*nu
            double tmp_init_12 = -tmp_init_11 + tmp_init_2 + 6.0
            double tmp_init_13 = chiS*tmp_init_12
            double tmp_init_14 = 7680.0*tmp_init_10 - 3840.0*tmp_init_13
            double tmp_init_15 = 1920.0*tmp_init_0
            double tmp_init_16 = 8.0*chiS
            double tmp_init_17 = 48.0*chiS
            double tmp_init_18 = 128.0*chiS
            double tmp_init_19 = nu - 2.0
            double tmp_init_20 = 768.0*chiS
            double tmp_init_21 = 384.0*tmp_init_2
            double tmp_init_22 = 144.0*nu
            double tmp_init_23 = cmath.pow(chiA, 2)
            double tmp_init_24 = 85.0*nu
            double tmp_init_25 = 4.0*tmp_init_2
            double tmp_init_26 = cmath.pow(chiS, 2)
            double tmp_init_27 = chiS*tmp_init_0
            double tmp_init_28 = 4.0*nu
            double tmp_init_29 = 9.8696044010893586
            double tmp_init_30 = -123.0*tmp_init_29
            double tmp_init_31 = 16.0*nu
            double tmp_init_32 = 9.0*nu
            double tmp_init_33 = 192.0*tmp_init_23
            double tmp_init_34 = 6.0*tmp_init_2
            double tmp_init_35 = 192.0*tmp_init_26
            double tmp_init_36 = 64.0*tmp_init_23
            double tmp_init_37 = 64.0*tmp_init_26
            double tmp_init_38 = tmp_init_32 - 11.0
            double tmp_init_39 = tmp_init_23*(-46.0*nu + tmp_init_34 + 11.0)
            double tmp_init_40 = -tmp_init_31
            double tmp_init_41 = tmp_init_26*(tmp_init_25 + tmp_init_40 + 11.0)
            double tmp_init_42 = 2268.0*tmp_init_23
            double tmp_init_43 = 36.0*nu
            double tmp_init_44 = 2268.0*tmp_init_26
            double tmp_init_45 = 0.69314718055994531
            double tmp_init_46 = cmath.pow(nu, 3)
            double tmp_init_47 = 3.1415926535897932*chiS
            double tmp_init_48 = 1247400.0*tmp_init_26
            double tmp_init_49 = 3.1415926535897932*delta
            double tmp_init_50 = chiS*delta
            double tmp_init_51 = 2494800.0*tmp_init_50
            double tmp_init_52 = tmp_init_27*tmp_init_38
            double tmp_init_53 = 12.0*chiS
            double tmp_init_54 = -513965390400.0*tmp_init_49
            double tmp_init_55 = 1455300.0*tmp_init_29
            double tmp_init_56 = 1.6094379124341004
            double tmp_init_57 = 1.0986122886681097
            double tmp_init_58 = 3880800.0*tmp_init_47
            double tmp_init_59 = -2337616325721600.0*tmp_init_45 + 1570602880000.0*tmp_init_46 + 579304687500000.0*tmp_init_56 + 638683384600800.0*tmp_init_57 + tmp_init_58*(42287.0*nu - 132438.0) + 282528472943.56691
            double tmp_init_60 = 1663200.0*tmp_init_23
            double tmp_init_61 = 1663200.0*tmp_init_26
            double tmp_init_62 = 3326400.0*tmp_init_50
            double tmp_init_63 = -8736588907200.0*tmp_init_49
            double tmp_init_64 = -2697716740561920.0*tmp_init_45 + 7164771768000.0*tmp_init_46 - 7761600.0*tmp_init_47*(95561.0*nu + 1125617.0) + 868957031250000.0*tmp_init_56 + 510129379766160.0*tmp_init_57 + 7048255400646.843
            double tmp_init_65 = tmp_init_0*tmp_init_16 + tmp_init_23*(4.0 - tmp_init_31) + 4.0*tmp_init_26
            double tmp_init_66 = 1809.5573684677209
            double tmp_init_67 = 4435200.0*tmp_init_23
            double tmp_init_68 = 4435200.0*tmp_init_26
            double tmp_init_69 = 8870400.0*tmp_init_50
            double tmp_init_70 = 7306397683200.0*tmp_init_49
            double tmp_init_71 = 4435200.0*tmp_init_50
            double tmp_init_72 = 2217600.0*tmp_init_23
            double tmp_init_73 = 2217600.0*tmp_init_26
            double tmp_init_74 = -254015272304640.0*tmp_init_45 - 5965749020000.0*tmp_init_46 + 186278400.0*tmp_init_47*(74954.0*nu - 39223.0) + 91540726421760.0*tmp_init_57 - 14631654901296.161
            double tmp_init_75 = 31322368000.0*tmp_init_46
            double tmp_init_76 = chiA*(9161916825600.0*tmp_init_49 + tmp_init_69*(3985373.0*nu - 2069461.0)) + 24375719121600.0*nu + 1983096561600.0*tmp_init_2 - 46569600.0*tmp_init_29*(17917.0*nu + 49216.0) + 6569517588480.0*tmp_init_45 + 798806624000.0*tmp_init_46 - 1490227200.0*tmp_init_47*(4937.0*nu - 6148.0) + 7472712360960.0*tmp_init_57 - tmp_init_67*(-9635809.0*nu + 4395216.0*tmp_init_2 + 2069461.0) - tmp_init_68*(-6612781.0*nu + 2014516.0*tmp_init_2 + 2069461.0) + 4044479733679.6361

        # computations
        self._gr_k_0 = tmp_init_3
        self._gr_k_1 = tmp_init_3
        self._gr_k_2 = 15.0*chiS*(-73.0*nu + tmp_init_6 + 32.0) + tmp_init_5*(32.0 - tmp_init_4)
        self._gr_k_3 = 30.0*chiS*(1421.0*nu + 310.0*tmp_init_2 - 5216.0) - 30.0*tmp_init_0*(tmp_init_7 + 5216.0)
        self._gr_k_4 = 15.0*chiS*(239.0*nu + 130.0*tmp_init_2 - 1376.0) - tmp_init_5*(179.0*nu + 1376.0)
        self._gr_k_5 = 120.0*chiS*(-113.0*nu + 26.0*tmp_init_2 + 32.0) + tmp_init_1*(32.0 - 43.0*nu)
        self._gr_k_6 = 120.0*chiS*(-139.0*nu + 158.0*tmp_init_2 - 1248.0) - tmp_init_1*(289.0*nu + 1248.0)
        self._gr_k_7 = 480.0*chiS*(21.0*nu + 38.0*tmp_init_2 - 416.0) - tmp_init_8*(67.0*nu + 416.0)
        self._gr_k_8 = 60.0*chiS*(-1275.0*nu + 254.0*tmp_init_2 + 256.0) + tmp_init_9*(256.0 - 481.0*nu)
        self._gr_k_9 = 60.0*chiS*(-3133.0*nu + 1074.0*tmp_init_2 - 5632.0) - tmp_init_9*(2247.0*nu + 5632.0)
        self._gr_k_10 = tmp_init_14
        self._gr_k_11 = tmp_init_14
        self._gr_k_12 = 240.0*chiS*(-1099.0*nu + 166.0*tmp_init_2 + 304.0) + 240.0*tmp_init_0*(304.0 - tmp_init_7)
        self._gr_k_13 = 1920.0*chiS*(-136.0*nu + 25.0*tmp_init_2 - 38.0) - tmp_init_15*(59.0*nu + 38.0)
        self._gr_k_14 = 30720.0*tmp_init_10 - 15360.0*tmp_init_13
        self._gr_k_15 = 480.0*chiS*(-1007.0*nu + 114.0*tmp_init_2 + 400.0) - tmp_init_8*(309.0*nu - 400.0)
        self._gr_k_16 = 15360.0*tmp_init_10 - 7680.0*tmp_init_13
        self._gr_k_17 = 128.0*nu
        self._gr_k_18 = 1920.0*chiS*(-89.0*nu + 8.0*tmp_init_2 + 44.0) + tmp_init_15*(44.0 - 25.0*nu)
        self._gr_k_19 = 0.00026041666666666667*flagPN52
        self._gr_k_20 = 6.0*chiS*(-1067672.0*nu + 272552.0*tmp_init_2 - 3441339.0) - 5813548.6213944483*nu - 6.0*tmp_init_0*(769244.0*nu + 3441339.0) - 27100776.238596404
        self._gr_k_21 = -353402155.83203087*nu - 8.0*tmp_init_0*(4790765.0*nu + 7388082.0) - tmp_init_16*(16825619.0*nu + 2435258.0*tmp_init_2 + 7388082.0) - 778173050.07290616
        self._gr_k_22 = 3252480.0*chiS*tmp_init_12 - 6504960.0*tmp_init_10
        self._gr_k_23 = 48.0*chiA*delta*(1745023.0*nu + 4476734.0) - 1003363520.4644293*nu - tmp_init_17*(3800571.0*nu + 3015082.0*tmp_init_2 - 4476734.0) - 949295718.15287038
        self._gr_k_24 = 4919040.0*chiS*tmp_init_12 - 9838080.0*tmp_init_10
        self._gr_k_25 = 16343040.0*tmp_init_10 - 8171520.0*tmp_init_13
        self._gr_k_26 = 38792.386086526767 - 4310.2651207251963*nu
        self._gr_k_27 = 128.0*chiA*delta*(952868.0*nu + 57681.0) - 348573829.5163581*nu - tmp_init_18*(-985610.0*nu + 670516.0*tmp_init_2 - 57681.0) - 159145141.88463065
        self._gr_k_28 = 4.0792084377610693e-7*flagPN52
        self._gr_k_29 = flagPN32*(-chiS*tmp_init_19 + 2.0*tmp_init_0)
        self._gr_k_30 = -144.0*chiS*(tmp_init_11 - 83.0) + 11952.0*tmp_init_0 + 31437.917684473061
        self._gr_k_31 = chiS*(65152.0*nu + 41632.0) + 41632.0*tmp_init_0 + 593384.02041004015
        self._gr_k_32 = -200576.0*tmp_init_0 + tmp_init_18*(1670.0*nu - 1567.0) + 829380.46054770541
        self._gr_k_33 = -86784.0*tmp_init_0 + tmp_init_20*(76.0*nu - 113.0) + 115811.67158193414
        self._gr_k_34 = 0.00010850694444444444*flagPN32
        self._gr_k_35 = 29904.0*tmp_init_0 - tmp_init_17*(92.0*nu - 623.0) + 76079.949291984023
        self._gr_k_36 = 64.0*chiS*(1993.0*nu - 62.0) - 3968.0*tmp_init_0 + 900103.99436531884
        self._gr_k_37 = -259712.0*tmp_init_0 + tmp_init_18*(1664.0*nu - 2029.0) + 594138.0026469017
        self._gr_k_38 = 3.4265350877192982e-5*flagPN32
        self._gr_k_39 = -3840.0*nu + tmp_init_21 - 2016.0
        self._gr_k_40 = tmp_init_22*(3.0*nu - 4.0)
        self._gr_k_41 = -3584.0*chiS*tmp_init_10 - 6880.0*nu + 160.0*tmp_init_2 + 256.0*tmp_init_23*(-tmp_init_24 + tmp_init_25 + 21.0) + 256.0*tmp_init_26*(-13.0*nu + 3.0*tmp_init_2 + 21.0) - 26688.0
        self._gr_k_42 = 864.0*nu*tmp_init_27 - 384.0*nu + 1440.0*tmp_init_2 + tmp_init_22*tmp_init_23*(1.0 - tmp_init_28) - tmp_init_22*tmp_init_26*(nu - 5.0) + 48.0
        self._gr_k_43 = 32.0*chiA*chiS*delta*(576.0 - 79.0*nu) - 20224.0*nu - 800.0*tmp_init_2 - 16.0*tmp_init_23*(2293.0*nu + 44.0*tmp_init_2 - 576.0) + 16.0*tmp_init_26*(-169.0*nu + 45.0*tmp_init_2 + 576.0) - 3792.0
        self._gr_k_44 = 384.0*chiA*chiS*delta*(tmp_init_31 - 1.0) + nu*(tmp_init_30 + 12224.0) - tmp_init_21 - tmp_init_33*(-tmp_init_32 + tmp_init_6 + 1.0) - tmp_init_35*(tmp_init_34 - tmp_init_4 + 1.0) + 1056.0
        self._gr_k_45 = 128.0*chiA*chiS*delta*(61.0*nu + 237.0) + 8.0*nu*(tmp_init_30 + 2260.0) - 5632.0*tmp_init_2 - tmp_init_36*(911.0*nu + 100.0*tmp_init_2 - 237.0) - tmp_init_37*(24.0*tmp_init_2 - tmp_init_24 - 237.0) + 21696.0
        self._gr_k_46 = 640.0*chiA*chiS*delta*(29.0*nu - 9.0) + 8.0*nu*(tmp_init_30 + 5212.0) - 5248.0*tmp_init_2 - tmp_init_36*(-223.0*nu + 124.0*tmp_init_2 + 45.0) - tmp_init_37*(-247.0*nu + 63.0*tmp_init_2 + 45.0) + 14208.0
        self._gr_k_47 = 4320.0*nu - 17280.0
        self._gr_k_48 = 5760.0*nu - 14400.0
        self._gr_k_49 = 6528.0*nu + 960.0*tmp_init_2 - 17280.0
        self._gr_k_50 = 2.0*nu*(-tmp_init_30 - 8000.0) - tmp_init_0*tmp_init_20*tmp_init_38 + 1920.0*tmp_init_2 + 384.0*tmp_init_39 + 384.0*tmp_init_41 - 8640.0
        self._gr_k_51 = 2016.0*nu - 3456.0
        self._gr_k_52 = 1152.0*chiA*chiS*delta*(19.0*nu - 13.0) + nu*(62176.0 - 1722.0*tmp_init_29) - 4608.0*tmp_init_2 - tmp_init_33*(-171.0*nu + 32.0*tmp_init_2 + 39.0) - tmp_init_35*(-99.0*nu + 28.0*tmp_init_2 + 39.0) + 3456.0
        self._gr_k_53 = 0.0026041666666666667*flagPN3
        self._gr_k_54 = 37227942.0*nu + 7753872.0*tmp_init_2 - 902664.0*tmp_init_27 + tmp_init_42*(832.0*nu - 199.0) - tmp_init_44*(tmp_init_43 + 199.0) + 35430930.0
        self._gr_k_55 = 139648536.0*nu + 31022796.0*tmp_init_2 + 1512.0*tmp_init_23*(8704.0*nu - 1969.0) - 1512.0*tmp_init_26*(828.0*nu + 1969.0) - 5954256.0*tmp_init_27 + 10762968.0
        self._gr_k_56 = 10463040.0*chiA*chiS*delta + 89627040.0*nu + 21300720.0*tmp_init_2 - 6048.0*tmp_init_23*(3232.0*nu - 865.0) - 6048.0*tmp_init_26*(228.0*nu - 865.0) - 42858976.0
        self._gr_k_57 = 955584.0*nu - 2388960.0
        self._gr_k_58 = 2649024.0*nu - 6622560.0
        self._gr_k_59 = 7560000.0 - 3024000.0*nu
        self._gr_k_60 = 1451520.0 - 580608.0*nu
        self._gr_k_61 = 871992.0*nu + 174048.0*tmp_init_2 + 2204109.0
        self._gr_k_62 = 5878656.0*chiA*chiS*delta + 4514976.0*nu + 1903104.0*tmp_init_2 - 36288.0*tmp_init_23*(320.0*nu - 81.0) - 36288.0*tmp_init_26*(tmp_init_28 - 81.0) - 360224.0
        self._gr_k_63 = 1.7223324514991182e-6*flagPN2
        self._gr_k_64 = 27599862.0*nu + 5073936.0*tmp_init_2 - 485352.0*tmp_init_27 + tmp_init_42*(448.0*nu - 107.0) - tmp_init_44*(20.0*nu + 107.0) + 22189718.0
        self._gr_k_65 = 49648356.0*nu + 11850804.0*tmp_init_2 + 1008.0*tmp_init_23*(4064.0*nu - 881.0) - 1008.0*tmp_init_26*(540.0*nu + 881.0) - 1776096.0*tmp_init_27 - 11389620.0
        self._gr_k_66 = 1018824.0*nu + 180768.0*tmp_init_2 + 3169323.0
        self._gr_k_67 = 3049200.0 - 1219680.0*nu
        self._gr_k_68 = 7660800.0 - 3064320.0*nu
        self._gr_k_69 = 5423040.0*chiA*chiS*delta + 12363120.0*nu + 3612672.0*tmp_init_2 - 10080.0*tmp_init_23*(1040.0*nu - 269.0) - 10080.0*tmp_init_26*(tmp_init_43 - 269.0) - 12243152.0
        self._gr_k_70 = 1.6316833751044277e-6*flagPN2
        self._gr_k_71 = 27.0*chiS*(41528.0*nu - 8008.0*tmp_init_2 + 125467.0) + 220065.42379131143*nu + 27.0*tmp_init_0*(27916.0*nu + 125467.0) - 3326842.9475940226
        self._gr_k_72 = chiA*(2281910400.0*tmp_init_49 + tmp_init_51*(141694.0*nu - 503031.0)) + 12690748224000.0*nu - 1140955200.0*tmp_init_19*tmp_init_47 + 4925307340800.0*tmp_init_2 - 1247400.0*tmp_init_23*(-2069104.0*nu + 112000.0*tmp_init_2 + 503031.0) + 1296672300.0*tmp_init_29*(41.0*nu - 64.0) + 507406394880.0*tmp_init_45 + 713101312000.0*tmp_init_46 - tmp_init_48*(-226408.0*nu + 57792.0*tmp_init_2 + 503031.0) + 5158720703419.9745
        self._gr_k_73 = 277200.0*nu*(140343.0*tmp_init_29 - 18306272.0) + 2656888819200.0*tmp_init_2 + 60726758400.0*tmp_init_39 + 60726758400.0*tmp_init_41 - 121453516800.0*tmp_init_52 - 6188907529728.0
        self._gr_k_74 = 2072.0*nu + 6931.0
        self._gr_k_75 = 51660.0*nu + 99106.0
        self._gr_k_76 = 89264.0*nu + 123288.0
        self._gr_k_77 = 14784.0*nu + 11888.0
        self._gr_k_78 = 93889298.753177973*nu + 12.0*tmp_init_0*(1062285.0*nu + 2448332.0) + tmp_init_53*(3133943.0*nu + 337722.0*tmp_init_2 + 2448332.0) + 216862494.68275295
        self._gr_k_79 = chiA*(831600.0*tmp_init_50*(4003818.0*nu - 6397597.0) + tmp_init_54) + 47873641384800.0*nu + 5953045190400.0*tmp_init_2 - 415800.0*tmp_init_23*(-26767760.0*nu + 3295488.0*tmp_init_2 + 6397597.0) - 415800.0*tmp_init_26*(-6830264.0*nu + 1593536.0*tmp_init_2 + 6397597.0) + tmp_init_55*(9061.0*nu - 110016.0) + tmp_init_59 + 18009590842332.0
        self._gr_k_80 = chiA*(tmp_init_51*(521486.0*nu - 1138719.0) + tmp_init_54) + 36614796372000.0*nu + 12469436179200.0*tmp_init_2 - 3742200.0*tmp_init_23*(-1588880.0*nu + 185472.0*tmp_init_2 + 379573.0) + 13097700.0*tmp_init_29*(6519.0*nu - 12224.0) - tmp_init_48*(-831208.0*nu + 169792.0*tmp_init_2 + 1138719.0) + tmp_init_59 + 2264761392444.0
        self._gr_k_81 = chiA*(-5275885507200.0*tmp_init_49 + tmp_init_62*(897617.0*nu - 2460015.0)) + 52159288242600.0*nu + 36261405318600.0*tmp_init_2 - 5302835325849600.0*tmp_init_45 + 4973853192000.0*tmp_init_46 + tmp_init_55*(783633.0*nu - 1793024.0) + 1448261718750000.0*tmp_init_56 + 1294530655405680.0*tmp_init_57 - tmp_init_58*(214925.0*nu + 1359484.0) - tmp_init_60*(-10429817.0*nu + 589232.0*tmp_init_2 + 2460015.0) - tmp_init_61*(-1205477.0*nu + 797748.0*tmp_init_2 + 2460015.0) - 76307112306811.317
        self._gr_k_82 = 5821200.0*nu*(6027.0*tmp_init_29 - 960472.0) + 1317919680000.0*tmp_init_2 + 54765849600.0*tmp_init_39 + 54765849600.0*tmp_init_41 - 109531699200.0*tmp_init_52 + 21690396471744.0
        self._gr_k_83 = 19768.0*nu + 94887.0
        self._gr_k_84 = 257124.0*nu + 464376.0
        self._gr_k_85 = 196448.0*nu + 164376.0
        self._gr_k_86 = 1344.0*chiS*(79817.0*nu + 36774.0*tmp_init_2 - 40462.0) + 423118281.41660296*nu - 672.0*tmp_init_0*(13833.0*nu + 80924.0) + 525883977.42490295
        self._gr_k_87 = chiA*(tmp_init_62*(2182831.0*nu - 3131923.0) + tmp_init_63) + 78553258707000.0*nu + 62213110759800.0*tmp_init_2 + tmp_init_55*(790193.0*nu - 2744576.0) - tmp_init_60*(-13837483.0*nu + 1693552.0*tmp_init_2 + 3131923.0) - tmp_init_61*(-3055871.0*nu + 1466108.0*tmp_init_2 + 3131923.0) + tmp_init_64 - 206275000274856.0
        self._gr_k_88 = chiA*(tmp_init_62*(1260511.0*nu - 2004643.0) + tmp_init_63) + 64875068643000.0*nu + 60944647995000.0*tmp_init_2 + tmp_init_55*(865223.0*nu - 2744576.0) - tmp_init_60*(-9123403.0*nu + 1078672.0*tmp_init_2 + 2004643.0) - tmp_init_61*(-1416191.0*nu + 1056188.0*tmp_init_2 + 2004643.0) + tmp_init_64 - 158355480094632.0
        self._gr_k_89 = 650496.0*chiS*tmp_init_12 - 1300992.0*tmp_init_10
        self._gr_k_90 = 1634304.0*chiS*tmp_init_12 - 3268608.0*tmp_init_10
        self._gr_k_91 = chiA*(-461691014400.0*tmp_init_49 + 2217600.0*tmp_init_50*(6220865.0*nu + 8736761.0)) + 9249237382000.0*nu + 57743681965200.0*tmp_init_2 - 1108800.0*tmp_init_23*(31182647.0*nu + 8270976.0*tmp_init_2 - 8736761.0) - 1108800.0*tmp_init_26*(-8677333.0*nu + 5503540.0*tmp_init_2 - 8736761.0) + 2910600.0*tmp_init_29*(591261.0*nu - 2647552.0) + 523497612718080.0*tmp_init_45 + 7311455228000.0*tmp_init_46 - 93139200.0*tmp_init_47*(101614.0*nu + 4957.0) - 252204042182400.0*tmp_init_57 - 214690401187925.21
        self._gr_k_92 = -6.0*nu - 3.0
        self._gr_k_93 = tmp_init_40 + tmp_init_65 - 80.0
        self._gr_k_94 = 96.0*chiS*(-125759.0*nu + 632758.0*tmp_init_2 - 642396.0) - 96.0*tmp_init_0*(696941.0*nu + 642396.0) + tmp_init_66*(171104.0*nu + 115991.0)
        self._gr_k_95 = chiA*(10941248102400.0*tmp_init_49 + tmp_init_69*(3657185.0*nu + 593801.0)) - 310182488000.0*nu + 20365567992000.0*tmp_init_2 - 58212000.0*tmp_init_29*(9717.0*nu + 69632.0) + 2332588769280.0*tmp_init_45 + 2406622064000.0*tmp_init_46 - 745113600.0*tmp_init_47*(13789.0*nu - 14684.0) + 22418137082880.0*tmp_init_57 - tmp_init_67*(851591.0*nu + 4835712.0*tmp_init_2 - 593801.0) - tmp_init_68*(-5790757.0*nu + 1987972.0*tmp_init_2 - 593801.0) - 94657177254764.919
        self._gr_k_96 = chiA*(tmp_init_70 + tmp_init_71*(4433569.0*nu + 7701538.0)) - 38365525968000.0*nu + 44062562728800.0*tmp_init_2 + 46569600.0*tmp_init_29*(28413.0*nu - 178048.0) - tmp_init_72*(28354633.0*nu + 6528312.0*tmp_init_2 - 7701538.0) - tmp_init_73*(-6415619.0*nu + 3646412.0*tmp_init_2 - 7701538.0) - tmp_init_74 - 175188432361920.0
        self._gr_k_97 = chiA*(tmp_init_70 + tmp_init_71*(3284449.0*nu + 9106018.0)) - 62018474179200.0*nu + 50052530959200.0*tmp_init_2 + 186278400.0*tmp_init_29*(8077.0*nu - 44512.0) - tmp_init_72*(34227913.0*nu + 5762232.0*tmp_init_2 - 9106018.0) - tmp_init_73*(-4372739.0*nu + 3135692.0*tmp_init_2 - 9106018.0) - tmp_init_74 - 194179275208896.0
        self._gr_k_98 = 266750668800.0*chiA*chiS*delta*tmp_init_38 - 554400.0*nu*(154119.0*tmp_init_29 - 19288240.0) - 3788530099200.0*tmp_init_2 - 133375334400.0*tmp_init_39 - 133375334400.0*tmp_init_41 + 27439417184256.0
        self._gr_k_99 = -32.0*nu + tmp_init_65 + 4.0
        self._gr_k_100 = 12.0*nu - 30.0
        self._gr_k_101 = tmp_init_28 - 24.0
        self._gr_k_102 = tmp_init_28 - 60.0
        self._gr_k_103 = -40.0*nu + tmp_init_0*tmp_init_53 + tmp_init_23*(6.0 - 24.0*nu) + 6.0*tmp_init_26 + 48.0
        self._gr_k_104 = 234956937600.0*nu + 166905446400.0*tmp_init_2 - 1425179448000.0
        self._gr_k_105 = 13305600.0*nu*(861.0*tmp_init_29 - 19748.0) - 353183846400.0*tmp_init_2 + 17882726400.0*tmp_init_39 + 17882726400.0*tmp_init_41 - 35765452800.0*tmp_init_52 + 3540081904128.0
        self._gr_k_106 = 175882291200.0*nu + 70945459200.0*tmp_init_2 + 10210816000.0*tmp_init_46 + 545305318425.0
        self._gr_k_107 = 384.0*chiS*(-91900.0*nu + 26544.0*tmp_init_2 + 31319.0) - 384.0*tmp_init_0*(48678.0*nu - 31319.0) + tmp_init_66*(15876.0*nu + 4159.0)
        self._gr_k_108 = -2.5834986772486772e-6*flagPN52
        self._gr_k_109 = chiA*(2011806720000.0*tmp_init_49 + 159667200.0*tmp_init_50*(68481.0*nu - 55817.0)) + 14110115958400.0*nu + 320791363200.0*tmp_init_2 - 79833600.0*tmp_init_23*(-243029.0*nu + 59136.0*tmp_init_2 + 55817.0) - 79833600.0*tmp_init_26*(-117201.0*nu + 33460.0*tmp_init_2 + 55817.0) - 279417600.0*tmp_init_29*(1845.0*nu + 1024.0) + 1749441576960.0*tmp_init_45 + 116010048000.0*tmp_init_46 - 8941363200.0*tmp_init_47*(148.0*nu - 225.0) - 9350951111594.3237
        self._gr_k_110 = -1.8639961596310803e-11*flagPN3
        self._gr_k_111 = -0.00018601190476190476*flagPN1
        self._gr_k_112 = 12.8*nu
        self._gr_k_113 = -709055424000.0*nu - 514084032000.0*tmp_init_2 + tmp_init_75 + 8213198775075.0
        self._gr_k_114 = 575031441600.0*nu + 227955974400.0*tmp_init_2 + tmp_init_75 + 1844398491075.0
        self._gr_k_115 = tmp_init_76 - 40817973192384.0
        self._gr_k_116 = tmp_init_76 - 41077048481472.0
        self._gr_k_117 = -5.886303661992885e-12*flagPN3
        self._gr_k_118 = -1.9580200501253133e-5*flagPN1
        self._gr_k_119 = -0.25*flagPN2
        self._gr_k_120 = -20.266666666666667*nu
        self._gr_k_121 = -3.0*flagPN1


    cdef void _compute(self
            , double e=-1
            , double x=-1
            , double z=-1
        ):
        """
        values being computed:
        - edotExpResumParser
        - xdotExpResumParser
        - zdotParser
        """

        # internal computations intermediate variables declaration/initialisation
        cdef:
            double tmp_0 = cmath.pow(e, 2)
            double tmp_1 = cmath.fabs(tmp_0 - 1)
            double tmp_2 = cmath.pow(tmp_1, -2.5)
            double tmp_3 = cmath.pow(e, 4)
            double tmp_4 = cmath.pow(tmp_1, -1)
            double tmp_5 = tmp_4*x
            double tmp_6 = cmath.pow(e, 6)
            double tmp_7 = x
            double tmp_8 = cmath.pow(tmp_1, 1.5)
            double tmp_9 = cmath.pow(tmp_7, 1.5)/tmp_8
            double tmp_10 = cmath.pow(x, 2)/cmath.pow(tmp_1, 2)
            double tmp_11 = cmath.pow(e, 8)
            double tmp_12 = cmath.sqrt(tmp_1)
            double tmp_13 = tmp_2*cmath.pow(tmp_7, 2.5)
            double tmp_14 = tmp_12 + 1.0
            double tmp_15 = cmath.sqrt(tmp_7)
            double tmp_16 = cmath.pow(tmp_1, 3)
            double tmp_17 = cmath.pow(x, 3)/tmp_16
            double tmp_18 = cmath.cos(z)
            double tmp_19 = e*tmp_18
            double tmp_20 = tmp_19 + 1.0
            double tmp_21 = cmath.pow(tmp_20, 2)
            double tmp_22 = tmp_19 + 2.0
            double tmp_23 = cmath.pow(e, 3)
            double tmp_24 = cmath.cos(2*z)
            double tmp_25 = cmath.pow(e, 5)
            double tmp_26 = cmath.cos(3*z)
            double tmp_27 = cmath.cos(4*z)

        # edot/math.m: --- edotFullExpResumSimplified ---#
        self.edotExpResumParser = self._gr_k_120*e*tmp_2*cmath.pow(x, 4)*(self._gr_k_117*tmp_17*(self._gr_k_113*tmp_11 + self._gr_k_116 + self._gr_k_79*tmp_6 + self._gr_k_87*tmp_3 + self._gr_k_97*tmp_0 + tmp_12*(self._gr_k_114*tmp_11 + self._gr_k_115 + self._gr_k_80*tmp_6 + self._gr_k_88*tmp_3 + self._gr_k_96*tmp_0) + 284739840.0*tmp_14*(-0.5*cmath.log(cmath.pow(tmp_14, 2)) + cmath.log(2*tmp_1*tmp_15))*(89024.0*tmp_0 + 42884.0*tmp_3 + 1719.0*tmp_6 + 24608.0))/tmp_14 + self._gr_k_118*tmp_5*(self._gr_k_83*tmp_3 + self._gr_k_84*tmp_0 + self._gr_k_85) + self._gr_k_28*tmp_13*(self._gr_k_20*tmp_6 + self._gr_k_21*tmp_3 + self._gr_k_23*tmp_0 + self._gr_k_26*tmp_11 + self._gr_k_27 + tmp_12*(self._gr_k_22*tmp_3 + self._gr_k_24*tmp_0 + self._gr_k_25)) + self._gr_k_38*tmp_9*(self._gr_k_35*tmp_3 + self._gr_k_36*tmp_0 + self._gr_k_37 - 307.87608005179974*tmp_6) + self._gr_k_70*tmp_10*(self._gr_k_64*tmp_3 + self._gr_k_65*tmp_0 + self._gr_k_66*tmp_6 + self._gr_k_69 + tmp_8*(self._gr_k_67*tmp_0 + self._gr_k_68)) + 0.39802631578947368*tmp_0 + 1.0)
        # xdot/math.m: --- xdotFullExpResumSimplified ---#
        self.xdotExpResumParser = self._gr_k_112*cmath.pow(tmp_1, -3.5)*cmath.pow(x, 5)*(self._gr_k_108*tmp_13*(self._gr_k_107 + self._gr_k_71*tmp_11 + self._gr_k_78*tmp_6 + self._gr_k_86*tmp_3 + self._gr_k_94*tmp_0 + tmp_8*(self._gr_k_89*tmp_3 + self._gr_k_90*tmp_0)) + self._gr_k_110*tmp_17*(self._gr_k_106*cmath.pow(e, 10) + self._gr_k_109 + self._gr_k_72*tmp_11 + self._gr_k_81*tmp_6 + self._gr_k_91*tmp_3 + self._gr_k_95*tmp_0 + tmp_12*(self._gr_k_104*tmp_11 + self._gr_k_105 + self._gr_k_73*tmp_6 + self._gr_k_82*tmp_3 + self._gr_k_98*tmp_0) + (-12391877836800.0*tmp_0 - 253703197440.0*tmp_11 - 23558235402240.0*tmp_3 - 7977271357440.0*tmp_6 - 874720788480.0)*cmath.log((1.0/2.0)*tmp_14*tmp_4/tmp_15)) + self._gr_k_111*tmp_5*(self._gr_k_74*tmp_6 + self._gr_k_75*tmp_3 + self._gr_k_76*tmp_0 + self._gr_k_77) + self._gr_k_34*tmp_9*(self._gr_k_30*tmp_6 + self._gr_k_31*tmp_3 + self._gr_k_32*tmp_0 + self._gr_k_33) + self._gr_k_63*tmp_10*(self._gr_k_54*tmp_6 + self._gr_k_55*tmp_3 + self._gr_k_56*tmp_0 + self._gr_k_61*tmp_11 + self._gr_k_62 + tmp_12*(self._gr_k_57*tmp_6 + self._gr_k_58*tmp_3 + self._gr_k_59*tmp_0 + self._gr_k_60)) + 3.0416666666666667*tmp_0 + 0.38541666666666667*tmp_3 + 1.0)
        # zdot/math.m: --- \[Chi]dotFullSimplified ---#
        self.zdotParser = tmp_21*tmp_9*(self._gr_k_119*tmp_10*(self._gr_k_100*tmp_8 + self._gr_k_101*tmp_3 + self._gr_k_102*tmp_18*tmp_23 + self._gr_k_103 + self._gr_k_99*tmp_19 + tmp_0*(self._gr_k_92*tmp_24 + self._gr_k_93)) + self._gr_k_121*tmp_5*(tmp_0 + tmp_20) + self._gr_k_19*tmp_13*(self._gr_k_15*tmp_19 + self._gr_k_17*e*tmp_16*tmp_22*(-1762.0*tmp_0 - 3265.0*tmp_3 + 5635.0*tmp_6 - 608.0)*cmath.sin(z) + self._gr_k_18 + tmp_0*(self._gr_k_12*tmp_24 + self._gr_k_13) + tmp_23*(self._gr_k_8*tmp_26 + self._gr_k_9*tmp_18) + tmp_25*(self._gr_k_2*cmath.cos(5*z) + self._gr_k_3*tmp_18 + self._gr_k_4*tmp_26) + tmp_3*(self._gr_k_5*tmp_27 + self._gr_k_6*tmp_24 + self._gr_k_7) + tmp_6*(self._gr_k_0*tmp_24 + self._gr_k_1) + tmp_8*(self._gr_k_14*tmp_19 + self._gr_k_16 + tmp_0*(self._gr_k_10*tmp_24 + self._gr_k_11)))/tmp_21 + self._gr_k_29*tmp_9*(tmp_0 + tmp_22) + self._gr_k_53*tmp_17*(self._gr_k_46*tmp_19 + self._gr_k_47*tmp_18*tmp_25 + self._gr_k_51*tmp_6 + self._gr_k_52 + tmp_0*(self._gr_k_44*tmp_24 + self._gr_k_45) + tmp_23*(self._gr_k_42*tmp_26 + self._gr_k_43*tmp_18) + tmp_3*(self._gr_k_39*tmp_24 + self._gr_k_40*tmp_27 + self._gr_k_41) + tmp_8*(self._gr_k_48*tmp_19 + self._gr_k_49*tmp_0 + self._gr_k_50)) + 1.0)


    cpdef double get(self, str expression_name):
        if expression_name == "edot":
            return self.edotExpResumParser
        elif expression_name == "xdot":
            return self.xdotExpResumParser
        elif expression_name == "zdot":
            return self.zdotParser
        raise RuntimeError(f"Unsupported expression '{expression_name}'")

    def initialize(self, *, chiA, chiS, delta, flagPN1, flagPN2, flagPN3, flagPN32, flagPN52, nu):
        ret = self._initialize(chiA, chiS, delta, flagPN1, flagPN2, flagPN3, flagPN32, flagPN52, nu)
        self._initialized = True
        return ret

    cpdef void compute(self
        , double e
        , double x
        , double z
    ):
        if not self._initialized:
            raise RuntimeError("Instance has not been initialized yet")
        self._compute(e, x, z)
