# cython: language_level=3

cdef class vpowers:
    cdef public double v,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,logv


cdef class spinVars:
    #cdef public array SxS12,lNxS1,lNxS2,lN,S1,S2
    cdef public double SxS12x
    cdef public double SxS12y
    cdef public double SxS12z

    cdef public double lNxS1x
    cdef public double lNxS1y
    cdef public double lNxS1z

    cdef public double lNxS2x
    cdef public double lNxS2y
    cdef public double lNxS2z

    cdef public double lNx
    cdef public double lNy
    cdef public double lNz
    cdef public double S1x
    cdef public double S1y
    cdef public double S1z
    cdef public double S2x
    cdef public double S2y
    cdef public double S2z
    cdef public double lNSxS12
    cdef public double lNS1
    cdef public double lNS2
    cdef public double SS12
    cdef public double Ssq1
    cdef public double Ssq2
    cdef public double lNSsq1
    cdef public double lNSsq2

#cpdef public void my_cross_cy(double ax,double ay,double az,double bx,double by,double bz, double cx, double cy, double cz)
cpdef public double my_dot_cy(double ax,double ay,double az,double bx,double by,double bz)
cpdef public double my_norm_cy(double ax,double ay,double az,double bx,double by,double bz)
cpdef public list my_cross_cy(double ax,double ay,double az,double bx,double by,double bz)
