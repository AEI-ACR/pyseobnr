"""
Contains hand-written vector operations wrapped in numba for speed.
For small operations, i.e. involving just 2 vectors these are faster
than NumPy since they do not incur overheads.
"""

from math import sqrt

import numpy as np
from numba import float64, jit


@jit(float64[:](float64[:], float64[:]), cache=True, nopython=True)
def my_cross(a, b):
    """
    Function to compute the cross product between two 3D arrays

    Args:
        a (np.ndarray): Array a
        b (np.ndarray): Array b

    Returns:
        (np.ndarray): Cross product axb
    """
    result = np.empty(3, dtype=np.float64)
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]
    return result


@jit(float64(float64[:], float64[:]), cache=True, nopython=True)
def my_dot(a, b):
    """
    Function to compute the dot product between two 3D arrays

    Args:
        a (np.ndarray): Array a
        b (np.ndarray): Array b

    Returns:
        (float): Dot product a.b
    """

    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@jit(
    float64(float64[:]),
    cache=True,
    nopython=True,
)
def my_norm(a):
    """
    Function to compute the norm of a 3D array

    Args:
        a (np.ndarray): Array a

    Returns:
        (float): L2-norm of a
    """
    norm = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

    return norm
