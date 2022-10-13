#------------------------------------------------------------------------------
# Piecewise power basis polynomials
#------------------------------------------------------------------------------
#cython: language_level=3
import numpy as np

cimport cython

cimport libc.stdlib
cimport libc.math
from cython.parallel import prange
from scipy.linalg.cython_lapack cimport dgeev

ctypedef double complex double_complex

ctypedef fused double_or_complex:
    double
    double complex

cdef extern from "numpy/npy_math.h":
    double nan "NPY_NAN"

DEF MAX_DIMS = 64


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def evaluate(double_or_complex[:,:,::1] c,
             const double[::1] x,
             const double[::1] xp,
             int dx,
             bint extrapolate,
             double_or_complex[:,::1] out,
             int[::1] indices):
    """
    Evaluate a piecewise polynomial.
    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
        There are `n` polynomials in each interval.
        Coefficient of highest order-term comes first.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials.
    xp : ndarray, shape (r,)
        Points to evaluate the piecewise polynomial at.
    dx : int
        Order of derivative to evaluate.  The derivative is evaluated
        piecewise and may have discontinuities.
    extrapolate : bint
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    out : ndarray, shape (r, n)
        Value of each polynomial at each of the input points.
        This argument is modified in-place.
    """
    cdef int ip, jp
    cdef int interval,stored_index
    cdef double xval



    interval = 0
    cdef int i
    # Evaluate.
    for ip in range(len(xp)):
        xval = xp[ip]

        # Find correct interval
        stored_index = indices[ip]
        if stored_index >=0:
            i = stored_index
        else:
            i = find_interval_ascending(&x[0], x.shape[0], xval, interval,
                                extrapolate)
            indices[ip] = i

        if i < 0:
            # xval was nan etc
            for jp in range(c.shape[2]):
                out[ip, jp] = nan
            continue
        else:
            interval = i

        # Evaluate the local polynomial(s)

        for jp in range(c.shape[2]):
            out[ip, jp] = evaluate_poly1(xval - x[interval], c, interval,
                                         jp, dx)







@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int find_interval_ascending(const double *x,
                                 size_t nx,
                                 double xval,
                                 int prev_interval=0,
                                 bint extrapolate=1) nogil:
    """
    Find an interval such that x[interval] <= xval < x[interval+1]. Assuming
    that x is sorted in the ascending order.
    If xval < x[0], then interval = 0, if xval > x[-1] then interval = n - 2.
    Parameters
    ----------
    x : array of double, shape (m,)
        Piecewise polynomial breakpoints sorted in ascending order.
    xval : double
        Point to find.
    prev_interval : int, optional
        Interval where a previous point was found.
    extrapolate : bint, optional
        Whether to return the last of the first interval if the
        point is out-of-bounds.
    Returns
    -------
    interval : int
        Suitable interval or -1 if nan.
    """
    cdef int interval, high, low, mid
    cdef double a, b

    a = x[0]
    b = x[nx-1]

    interval = prev_interval
    if interval < 0 or interval >= nx:
        interval = 0

    if not (a <= xval <= b):
        # Out-of-bounds (or nan)
        if xval < a and extrapolate:
            # below
            interval = 0
        elif xval > b and extrapolate:
            # above
            interval = nx - 2
        else:
            # nan or no extrapolation
            interval = -1
    elif xval == b:
        # Make the interval closed from the right
        interval = nx - 2
    else:
        # Find the interval the coordinate is in
        # (binary search with locality)
        if xval >= x[interval]:
            low = interval
            high = nx - 2
        else:
            low = 0
            high = interval

        if xval < x[low+1]:
            high = low

        while low < high:
            mid = (high + low)//2
            if xval < x[mid]:
                # mid < high
                high = mid
            elif xval >= x[mid + 1]:
                low = mid + 1
            else:
                # x[mid] <= xval < x[mid+1]
                low = mid
                break

        interval = low

    return interval


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.profile(True)
cdef double_or_complex evaluate_poly1(double s, double_or_complex[:,:,::1] c, int ci, int cj, int dx) nogil:
    """
    Evaluate polynomial, derivative, or antiderivative in a single interval.
    Antiderivatives are evaluated assuming zero integration constants.
    Parameters
    ----------
    s : double
        Polynomial x-value
    c : double[:,:,:]
        Polynomial coefficients. c[:,ci,cj] will be used
    ci, cj : int
        Which of the coefs to use
    dx : int
        Order of derivative (> 0) or antiderivative (< 0) to evaluate.
    """
    cdef int kp, k
    cdef double_or_complex res, z

    res = 0.0
    z = 1.0


    for kp in range(c.shape[0]):

        res = res + c[c.shape[0] - kp - 1, ci, cj] * z

        # compute x**max(k-dx,0)
        if kp < c.shape[0] - 1 and kp >= dx:
            z *= s

    return res