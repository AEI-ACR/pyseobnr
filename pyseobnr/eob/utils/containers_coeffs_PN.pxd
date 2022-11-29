# cython: language_level=3


cdef class lNdotCoeffs:

  cdef public double alNdotv11642, alNdotv11641, alNdotv11632, alNdotv11631, alNdotv1162, alNdotv1161, alNdotv1052, alNdotv1051, alNdotv9442, alNdotv9441, alNdotv9432, alNdotv9431, alNdotv942, alNdotv941, alNdotv832, alNdotv831, alNdotv7222, alNdotv7221, alNdotv7212, alNdotv7211, alNdotv612, alNdotv611

cdef class lhatCoeffs:

  cdef public double alhatvLog8, alhatv8618, alhatv8608, alhatv8598, alhatv8588, alhatv8578, alhatv8568, alhatv8558, alhatv8548, alhatv8538, alhatv8528, alhatv8518, alhatv8508, alhatv8498, alhatv8488, alhatv8478, alhatv8468, alhatv8458, alhatv8448, alhatv8438, alhatv8428, alhatv8418, alhatv8408, alhatv8398, alhatv8388, alhatv8378, alhatv8368, alhatv8358, alhatv8348, alhatv8338, alhatv8328, alhatv8318, alhatv8308, alhatv8298, alhatv8288, alhatv8278, alhatv8268, alhatv8258, alhatv8248, alhatv8238, alhatv8228, alhatv8218, alhatv8208, alhatv8198, alhatv8188, alhatv8178, alhatv8168, alhatv8158, alhatv8148, alhatv8138, alhatv8128, alhatv8118, alhatv8108, alhatv898, alhatv888, alhatv878, alhatv868, alhatv858, alhatv848, alhatv838, alhatv828, alhatv818, alhatv7287, alhatv7277, alhatv7267, alhatv7257, alhatv7247, alhatv7237, alhatv7227, alhatv7217, alhatv7207, alhatv7197, alhatv7187, alhatv7177, alhatv7167, alhatv7157, alhatv7147, alhatv7137, alhatv7127, alhatv7117, alhatv7107, alhatv797, alhatv787, alhatv777, alhatv767, alhatv757, alhatv747, alhatv737, alhatv727, alhatv717, alhatv6366, alhatv6356, alhatv6346, alhatv6336, alhatv6326, alhatv6316, alhatv6306, alhatv6296, alhatv6286, alhatv6276, alhatv6266, alhatv6256, alhatv6246, alhatv6236, alhatv6226, alhatv6216, alhatv6206, alhatv6196, alhatv6186, alhatv6176, alhatv6166, alhatv6156, alhatv6146, alhatv6136, alhatv6126, alhatv6116, alhatv6106, alhatv696, alhatv686, alhatv676, alhatv666, alhatv656, alhatv646, alhatv636, alhatv626, alhatv616, alhatv5205, alhatv5195, alhatv5185, alhatv5175, alhatv5165, alhatv5155, alhatv5145, alhatv5135, alhatv5125, alhatv5115, alhatv5105, alhatv595, alhatv585, alhatv575, alhatv565, alhatv555, alhatv545, alhatv535, alhatv525, alhatv515, alhatv4194, alhatv4184, alhatv4174, alhatv4164, alhatv4154, alhatv4144, alhatv4134, alhatv4124, alhatv4114, alhatv4104, alhatv494, alhatv484, alhatv474, alhatv464, alhatv454, alhatv444, alhatv434, alhatv424, alhatv414, alhatv3113, alhatv3103, alhatv393, alhatv383, alhatv373, alhatv363, alhatv353, alhatv343, alhatv333, alhatv323, alhatv313, alhatv222, alhatv212, alhatv01

cdef class omegadotCoeffs:

  cdef public double aomegadotv8448, aomegadotv8438, aomegadotv8428, aomegadotv8418, aomegadotv8408, aomegadotv8398, aomegadotv8388, aomegadotv8378, aomegadotv8368, aomegadotv8358, aomegadotv8348, aomegadotv8338, aomegadotv8328, aomegadotv8318, aomegadotv8308, aomegadotv8298, aomegadotv8288, aomegadotv8278, aomegadotv8268, aomegadotv8258, aomegadotv8248, aomegadotv8238, aomegadotv8228, aomegadotv8218, aomegadotv8208, aomegadotv8198, aomegadotv8188, aomegadotv8178, aomegadotv8168, aomegadotv8158, aomegadotv8148, aomegadotv8138, aomegadotv8128, aomegadotv8118, aomegadotv8108, aomegadotv898, aomegadotv888, aomegadotv878, aomegadotv868, aomegadotv858, aomegadotv848, aomegadotv838, aomegadotv828, aomegadotv818, aomegadotv7317, aomegadotv7307, aomegadotv7297, aomegadotv7287, aomegadotv7277, aomegadotv7267, aomegadotv7257, aomegadotv7247, aomegadotv7237, aomegadotv7227, aomegadotv7217, aomegadotv7207, aomegadotv7197, aomegadotv7187, aomegadotv7177, aomegadotv7167, aomegadotv7157, aomegadotv7147, aomegadotv7137, aomegadotv7127, aomegadotv7117, aomegadotv7107, aomegadotv797, aomegadotv787, aomegadotv777, aomegadotv767, aomegadotv757, aomegadotv747, aomegadotv737, aomegadotv727, aomegadotv717, aomegadotvLog6, aomegadotv6376, aomegadotv6366, aomegadotv6356, aomegadotv6346, aomegadotv6336, aomegadotv6326, aomegadotv6316, aomegadotv6306, aomegadotv6296, aomegadotv6286, aomegadotv6276, aomegadotv6266, aomegadotv6256, aomegadotv6246, aomegadotv6236, aomegadotv6226, aomegadotv6216, aomegadotv6206, aomegadotv6196, aomegadotv6186, aomegadotv6176, aomegadotv6166, aomegadotv6156, aomegadotv6146, aomegadotv6136, aomegadotv6126, aomegadotv6116, aomegadotv6106, aomegadotv696, aomegadotv686, aomegadotv676, aomegadotv666, aomegadotv656, aomegadotv646, aomegadotv636, aomegadotv626, aomegadotv616, aomegadotv5125, aomegadotv5115, aomegadotv5105, aomegadotv595, aomegadotv585, aomegadotv575, aomegadotv565, aomegadotv555, aomegadotv545, aomegadotv535, aomegadotv525, aomegadotv515, aomegadotv4174, aomegadotv4164, aomegadotv4154, aomegadotv4144, aomegadotv4134, aomegadotv4124, aomegadotv4114, aomegadotv4104, aomegadotv494, aomegadotv484, aomegadotv474, aomegadotv464, aomegadotv454, aomegadotv444, aomegadotv434, aomegadotv424, aomegadotv414, aomegadotv363, aomegadotv353, aomegadotv343, aomegadotv333, aomegadotv323, aomegadotv313, aomegadotv222, aomegadotv212, aomegadotv0

cdef class s1dotCoeffs:

  cdef public double asdotv110622, asdotv110621, asdotv11061, asdotv1950, asdotv18422, asdotv18421, asdotv1841, asdotv1730, asdotv16222, asdotv16221, asdotv1621, asdotv1510

cdef class s2dotCoeffs:

  cdef public double asdotv210622, asdotv210621, asdotv21061, asdotv2950, asdotv28422, asdotv28421, asdotv2841, asdotv2730, asdotv26222, asdotv26221, asdotv2621, asdotv2510


cdef class PNCoeffs:
    cdef public s1dotCoeffs s1dot_coeffs
    cdef public s2dotCoeffs s2dot_coeffs
    cdef public omegadotCoeffs omegadot_coeffs
    cdef public lNdotCoeffs lNdot_coeffs
    cdef public lhatCoeffs lhat_coeffs


