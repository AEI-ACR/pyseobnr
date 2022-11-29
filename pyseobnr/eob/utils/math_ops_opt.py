import numpy as np
from numba import jit
from numba import float64
from math import cos,sin

@jit(float64[:](float64[:], float64[:]), cache=True, nopython=True)
def my_cross(a, b):
    result = np.empty(3, dtype=np.float64)
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]
    return result


@jit(float64(float64[:], float64[:]), cache=True, nopython=True)
def my_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]



@jit(
   float64(float64[:]),
   cache=True,
   nopython=True,
)
def my_norm(a):

   norm = np.sqrt(a[0]*a[0] +  a[1]*a[1] +  a[2]*a[2])

   return  norm



@jit(
   float64(float64[:]),
   cache=True,
   nopython=True,
)
def my_norm(a):

   norm = np.sqrt(a[0]*a[0] +  a[1]*a[1] +  a[2]*a[2])

   return  norm

@jit(float64(float64,float64,float64,float64,float64,float64),
     cache=True,
     nopython=True)
def my_projection(a1, phi1, theta1, a2, phi2, theta2):

    cosphi_1 = cos(phi1)
    sinphi_1 = sin(phi1)
    costheta_1 = cos(theta1)
    sintheta_1 = sin(theta1)

    cosphi_2 = cos(phi2)
    sinphi_2 = sin(phi2)
    costheta_2 = cos(theta2)
    sintheta_2 = sin(theta2)

    result = a1*a2*( (cosphi_1*cosphi_2 + sinphi_1*sinphi_2)*sintheta_1*sintheta_2 + costheta_1*costheta_2)
    return result
