# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
cimport cython 
from libc.math cimport log, sqrt

cdef class vpowers:
    @cython.cdivision(True)
    @cython.embedsignature(True)
    def __cinit__(self, double omega):

        self.v1 = omega**(1./3)
        self.v2 = self.v1*self.v1
        self.v3 = self.v2*self.v1
        self.v4 = self.v3*self.v1
        self.v5 = self.v4*self.v1
        self.v6 = self.v5*self.v1
        self.v7 = self.v6*self.v1
        self.v8 = self.v7*self.v1
        self.v9 = self.v8*self.v1
        self.v10 = self.v9*self.v1
        self.v11 = self.v10*self.v1
        self.logv = log(self.v1)


cdef class spinVars:
    @cython.cdivision(True)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, double lNx, double lNy, double lNz,
                        double S1x, double S1y, double S1z,
                        double S2x, double S2y, double S2z):

        self.lNx = lNx
        self.lNy = lNy
        self.lNz = lNz

        self.S1x = S1x
        self.S1y = S1y
        self.S1z = S1z

        self.S2x = S2x
        self.S2y = S2y
        self.S2z = S2z

        #my_cross_cy(self.lNx,self.lNy,self.lNz,self.S1x,self.S1y,self.S1z,self.lNxS1x,self.lNxS1y,self.lNxS1z)
        #my_cross_cy(self.lNx,self.lNy,self.lNz,self.S2x,self.S2y,self.S2z,self.lNxS2x,self.lNxS2y,self.lNxS2z)
        #my_cross_cy(self.S1x,self.S1y,self.S1z,self.S2x,self.S2y,self.S2z,self.SxS12x,self.SxS12y,self.SxS12z)

        self.lNxS1x,self.lNxS1y,self.lNxS1z = my_cross_cy(self.lNx,self.lNy,self.lNz,self.S1x,self.S1y,self.S1z)
        self.lNxS2x,self.lNxS2y,self.lNxS2z = my_cross_cy(self.lNx,self.lNy,self.lNz,self.S2x,self.S2y,self.S2z)
        self.SxS12x,self.SxS12y,self.SxS12z = my_cross_cy(self.S1x,self.S1y,self.S1z,self.S2x,self.S2y,self.S2z)


        self.lNSxS12 = my_dot_cy(self.lNx,self.lNy,self.lNz,self.SxS12x,self.SxS12y,self.SxS12z)
        self.Ssq2 = my_dot_cy(self.S2x,self.S2y,self.S2z,self.S2x,self.S2y,self.S2z)
        self.Ssq1 = my_dot_cy(self.S1x,self.S1y,self.S1z,self.S1x,self.S1y,self.S1z)
        self.SS12 = my_dot_cy(self.S1x,self.S1y,self.S1z,self.S2x,self.S2y,self.S2z)

        self.lNS1 = my_dot_cy(self.S1x,self.S1y,self.S1z, self.lNx,self.lNy,self.lNz)
        self.lNS2 = my_dot_cy(self.S2x,self.S2y,self.S2z, self.lNx,self.lNy,self.lNz)

        self.lNSsq1 = self.lNS1*self.lNS1
        self.lNSsq2 = self.lNS2*self.lNS2


## Faster functions for 3D operations

#@cython.wraparound(False)
#@cython.boundscheck(False)
#@cython.cdivision(True)
#@cython.nonecheck(False)
#@cython.initializedcheck(False)
#cpdef public void my_cross_cy(double ax,double ay,double az,double bx,double by,double bz, double cx, double cy, double cz):
#
#    cx = ay * bz - az * by
#    cy = az * bx - ax * bz
#    cz = ax * by - ay * bx

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef public double my_norm_cy(double ax,double ay,double az,double bx,double by,double bz):
    return sqrt(ax*bx +  ay*by +  az*bz)



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef public double my_dot_cy(double ax,double ay,double az,double bx,double by,double bz):


    return ax*bx +  ay*by +  az*bz

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef public list my_cross_cy(double ax,double ay,double az,double bx,double by,double bz):

    cdef double cx,cy,cz

    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx

    return [cx,cy,cz]
