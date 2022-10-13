import itertools
import warnings

import numpy as np
from numpy import (
    array,
    transpose,
    searchsorted,
    atleast_1d,
    atleast_2d,
    ravel,
    poly1d,
    asarray,
    intp,
)

from . import _ppoly
from scipy._lib._util import prod
from scipy.special import comb
import scipy.special as spec
from scipy.interpolate import splev


class _PPolyBase:
    """Base class for piecewise polynomials."""

    __slots__ = ("c", "x", "extrapolate", "axis", "indices")

    def __init__(self, c, x, extrapolate=None, axis=0, indices=None):
        self.c = np.asarray(c)
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        if indices is not None:
            self.indices = indices
        else:
            self.indices = None
        if extrapolate is None:
            extrapolate = True
        elif extrapolate != "periodic":
            extrapolate = bool(extrapolate)
        self.extrapolate = extrapolate

        if self.c.ndim < 2:
            raise ValueError("Coefficients array must be at least " "2-dimensional.")

        if not (0 <= axis < self.c.ndim - 1):
            raise ValueError(
                "axis=%s must be between 0 and %s" % (axis, self.c.ndim - 1)
            )

        self.axis = axis
        if axis != 0:
            # roll the interpolation axis to be the first one in self.c
            # More specifically, the target shape for self.c is (k, m, ...),
            # and axis !=0 means that we have c.shape (..., k, m, ...)
            #                                               ^
            #                                              axis
            # So we roll two of them.
            self.c = np.rollaxis(self.c, axis + 1)
            self.c = np.rollaxis(self.c, axis + 1)

        if self.x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        if self.x.size < 2:
            raise ValueError("at least 2 breakpoints are needed")
        if self.c.ndim < 2:
            raise ValueError("c must have at least 2 dimensions")
        if self.c.shape[0] == 0:
            raise ValueError("polynomial must be at least of order 0")
        if self.c.shape[1] != self.x.size - 1:
            raise ValueError("number of coefficients != len(x)-1")
        dx = np.diff(self.x)
        if not (np.all(dx >= 0) or np.all(dx <= 0)):
            raise ValueError("`x` must be strictly increasing or decreasing.")

        dtype = self._get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dtype)

    def _get_dtype(self, dtype):
        if np.issubdtype(dtype, np.complexfloating) or np.issubdtype(
            self.c.dtype, np.complexfloating
        ):
            return np.complex_
        else:
            return np.float_

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0, indices=None):
        """
        Construct the piecewise polynomial without making checks.
        Takes the same parameters as the constructor. Input arguments
        ``c`` and ``x`` must be arrays of the correct shape and type. The
        ``c`` array can only be of dtypes float and complex, and ``x``
        array must have dtype float.
        """
        self = object.__new__(cls)
        self.c = c
        self.x = x
        self.axis = axis
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        self.indices = indices
        return self

    def _ensure_c_contiguous(self):
        """
        c and x may be modified by the user. The Cython code expects
        that they are C contiguous.
        """
        if not self.x.flags.c_contiguous:
            self.x = self.x.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def extend(self, c, x, right=None):
        """
        Add additional breakpoints and coefficients to the polynomial.
        Parameters
        ----------
        c : ndarray, size (k, m, ...)
            Additional coefficients for polynomials in intervals. Note that
            the first additional interval will be formed using one of the
            ``self.x`` end points.
        x : ndarray, size (m,)
            Additional breakpoints. Must be sorted in the same order as
            ``self.x`` and either to the right or to the left of the current
            breakpoints.
        right
            Deprecated argument. Has no effect.
            .. deprecated:: 0.19
        """
        if right is not None:
            warnings.warn("`right` is deprecated and will be removed.")

        c = np.asarray(c)
        x = np.asarray(x)

        if c.ndim < 2:
            raise ValueError("invalid dimensions for c")
        if x.ndim != 1:
            raise ValueError("invalid dimensions for x")
        if x.shape[0] != c.shape[1]:
            raise ValueError(
                "Shapes of x {} and c {} are incompatible".format(x.shape, c.shape)
            )
        if c.shape[2:] != self.c.shape[2:] or c.ndim != self.c.ndim:
            raise ValueError(
                "Shapes of c {} and self.c {} are incompatible".format(
                    c.shape, self.c.shape
                )
            )

        if c.size == 0:
            return

        dx = np.diff(x)
        if not (np.all(dx >= 0) or np.all(dx <= 0)):
            raise ValueError("`x` is not sorted.")

        if self.x[-1] >= self.x[0]:
            if not x[-1] >= x[0]:
                raise ValueError("`x` is in the different order " "than `self.x`.")

            if x[0] >= self.x[-1]:
                action = "append"
            elif x[-1] <= self.x[0]:
                action = "prepend"
            else:
                raise ValueError(
                    "`x` is neither on the left or on the right " "from `self.x`."
                )
        else:
            if not x[-1] <= x[0]:
                raise ValueError("`x` is in the different order " "than `self.x`.")

            if x[0] <= self.x[-1]:
                action = "append"
            elif x[-1] >= self.x[0]:
                action = "prepend"
            else:
                raise ValueError(
                    "`x` is neither on the left or on the right " "from `self.x`."
                )

        dtype = self._get_dtype(c.dtype)

        k2 = max(c.shape[0], self.c.shape[0])
        c2 = np.zeros(
            (k2, self.c.shape[1] + c.shape[1]) + self.c.shape[2:], dtype=dtype
        )

        if action == "append":
            c2[k2 - self.c.shape[0] :, : self.c.shape[1]] = self.c
            c2[k2 - c.shape[0] :, self.c.shape[1] :] = c
            self.x = np.r_[self.x, x]
        elif action == "prepend":
            c2[k2 - self.c.shape[0] :, : c.shape[1]] = c
            c2[k2 - c.shape[0] :, c.shape[1] :] = self.c
            self.x = np.r_[x, self.x]

        self.c = c2

    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate the piecewise polynomial or its derivative.
        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.
        nu : int, optional
            Order of derivative to evaluate. Must be non-negative.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.
        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.
        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = np.asarray(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = np.ascontiguousarray(x.ravel(), dtype=np.float_)

        # With periodic extrapolation we map x to the segment
        # [self.x[0], self.x[-1]].
        if extrapolate == "periodic":
            x = self.x[0] + (x - self.x[0]) % (self.x[-1] - self.x[0])
            extrapolate = False

        out = np.empty((len(x), prod(self.c.shape[2:])), dtype=self.c.dtype)
        if self.indices is None:
            indices = np.ones(len(x), dtype=np.intc) * -1
        else:
            indices = self.indices
        self._ensure_c_contiguous()
        self._evaluate(x, nu, extrapolate, out, indices)
        out = out.reshape(x_shape + self.c.shape[2:])
        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            l = list(range(out.ndim))
            l = l[x_ndim : x_ndim + self.axis] + l[:x_ndim] + l[x_ndim + self.axis :]
            out = out.transpose(l)
        return out, indices


class PPoly(_PPolyBase):
    """
    Piecewise polynomial in terms of coefficients and breakpoints
    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
    local power basis::
        S = sum(c[m, i] * (xp - x[i])**(k-m) for m in range(k+1))
    where ``k`` is the degree of the polynomial.
    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals.
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool or 'periodic', optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.
    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-D array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.
    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    solve
    roots
    extend
    from_spline
    from_bernstein_basis
    construct_fast
    See also
    --------
    BPoly : piecewise polynomials in the Bernstein basis
    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable. Precision problems can start to appear for orders
    larger than 20-30.
    """

    def _evaluate(self, x, nu, extrapolate, out, indices):
        # print(x,nu,extrapolate,indices)
        _ppoly.evaluate(
            self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
            self.x,
            x,
            nu,
            bool(extrapolate),
            out,
            indices,
        )

    def derivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the derivative.
        Parameters
        ----------
        nu : int, optional
            Order of derivative to evaluate. Default is 1, i.e., compute the
            first derivative. If negative, the antiderivative is returned.
        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k - n representing the derivative
            of this polynomial.
        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if nu < 0:
            return self.antiderivative(-nu)

        # reduce order
        if nu == 0:
            c2 = self.c.copy()
        else:
            c2 = self.c[:-nu, :].copy()

        if c2.shape[0] == 0:
            # derivative of order 0 is zero
            c2 = np.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        # multiply by the correct rising factorials
        factor = spec.poch(np.arange(c2.shape[0], 0, -1), nu)
        c2 *= factor[(slice(None),) + (None,) * (c2.ndim - 1)]

        # construct a compatible polynomial
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the antiderivative.
        Antiderivative is also the indefinite integral of the function,
        and derivative is its inverse operation.
        Parameters
        ----------
        nu : int, optional
            Order of antiderivative to evaluate. Default is 1, i.e., compute
            the first integral. If negative, the derivative is returned.
        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k + n representing
            the antiderivative of this polynomial.
        Notes
        -----
        The antiderivative returned by this function is continuous and
        continuously differentiable to order n-1, up to floating point
        rounding error.
        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.
        """
        if nu <= 0:
            return self.derivative(-nu)

        c = np.zeros(
            (self.c.shape[0] + nu, self.c.shape[1]) + self.c.shape[2:],
            dtype=self.c.dtype,
        )
        c[:-nu] = self.c

        # divide by the correct rising factorials
        factor = spec.poch(np.arange(self.c.shape[0], 0, -1), nu)
        c[:-nu] /= factor[(slice(None),) + (None,) * (c.ndim - 1)]

        # fix continuity of added degrees of freedom
        self._ensure_c_contiguous()
        _ppoly.fix_continuity(c.reshape(c.shape[0], c.shape[1], -1), self.x, nu - 1)

        if self.extrapolate == "periodic":
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        # construct a compatible polynomial
        return self.construct_fast(c, self.x, extrapolate, self.axis)

    def integrate(self, a, b, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.
        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.
        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over [a, b]
        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        # Swap integration bounds if needed
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1

        range_int = np.empty((prod(self.c.shape[2:]),), dtype=self.c.dtype)
        self._ensure_c_contiguous()

        # Compute the integral.
        if extrapolate == "periodic":
            # Split the integral into the part over period (can be several
            # of them) and the remaining part.

            xs, xe = self.x[0], self.x[-1]
            period = xe - xs
            interval = b - a
            n_periods, left = divmod(interval, period)

            if n_periods > 0:
                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x,
                    xs,
                    xe,
                    False,
                    out=range_int,
                )
                range_int *= n_periods
            else:
                range_int.fill(0)

            # Map a to [xs, xe], b is always a + left.
            a = xs + (a - xs) % period
            b = a + left

            # If b <= xe then we need to integrate over [a, b], otherwise
            # over [a, xe] and from xs to what is remained.
            remainder_int = np.empty_like(range_int)
            if b <= xe:
                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x,
                    a,
                    b,
                    False,
                    out=remainder_int,
                )
                range_int += remainder_int
            else:
                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x,
                    a,
                    xe,
                    False,
                    out=remainder_int,
                )
                range_int += remainder_int

                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x,
                    xs,
                    xs + left + a - xe,
                    False,
                    out=remainder_int,
                )
                range_int += remainder_int
        else:
            _ppoly.integrate(
                self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                self.x,
                a,
                b,
                bool(extrapolate),
                out=range_int,
            )

        # Return
        range_int *= sign
        return range_int.reshape(self.c.shape[2:])

    def solve(self, y=0.0, discontinuity=True, extrapolate=None):
        """
        Find real solutions of the the equation ``pp(x) == y``.
        Parameters
        ----------
        y : float, optional
            Right-hand side. Default is zero.
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to return roots from the polynomial
            extrapolated based on first and last intervals, 'periodic' works
            the same as False. If None (default), use `self.extrapolate`.
        Returns
        -------
        roots : ndarray
            Roots of the polynomial(s).
            If the PPoly object describes multiple polynomials, the
            return value is an object array whose each element is an
            ndarray containing the roots.
        Notes
        -----
        This routine works only on real-valued polynomials.
        If the piecewise polynomial contains sections that are
        identically zero, the root list will contain the start point
        of the corresponding interval, followed by a ``nan`` value.
        If the polynomial is discontinuous across a breakpoint, and
        there is a sign change across the breakpoint, this is reported
        if the `discont` parameter is True.
        Examples
        --------
        Finding roots of ``[x**2 - 1, (x - 1)**2]`` defined on intervals
        ``[-2, 1], [1, 2]``:
        >>> from scipy.interpolate import PPoly
        >>> pp = PPoly(np.array([[1, -4, 3], [1, 0, 0]]).T, [-2, 1, 2])
        >>> pp.solve()
        array([-1.,  1.])
        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        self._ensure_c_contiguous()

        if np.issubdtype(self.c.dtype, np.complexfloating):
            raise ValueError("Root finding is only for " "real-valued polynomials")

        y = float(y)
        r = _ppoly.real_roots(
            self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
            self.x,
            y,
            bool(discontinuity),
            bool(extrapolate),
        )
        if self.c.ndim == 2:
            return r[0]
        else:
            r2 = np.empty(prod(self.c.shape[2:]), dtype=object)
            # this for-loop is equivalent to ``r2[...] = r``, but that's broken
            # in NumPy 1.6.0
            for ii, root in enumerate(r):
                r2[ii] = root

            return r2.reshape(self.c.shape[2:])

    def roots(self, discontinuity=True, extrapolate=None):
        """
        Find real roots of the the piecewise polynomial.
        Parameters
        ----------
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to return roots from the polynomial
            extrapolated based on first and last intervals, 'periodic' works
            the same as False. If None (default), use `self.extrapolate`.
        Returns
        -------
        roots : ndarray
            Roots of the polynomial(s).
            If the PPoly object describes multiple polynomials, the
            return value is an object array whose each element is an
            ndarray containing the roots.
        See Also
        --------
        PPoly.solve
        """
        return self.solve(0, discontinuity, extrapolate)

    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        """
        Construct a piecewise polynomial from a spline
        Parameters
        ----------
        tck
            A spline, as returned by `splrep` or a BSpline object.
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        if isinstance(tck, BSpline):
            t, c, k = tck.tck
            if extrapolate is None:
                extrapolate = tck.extrapolate
        else:
            t, c, k = tck

        cvals = np.empty((k + 1, len(t) - 1), dtype=c.dtype)
        for m in range(k, -1, -1):
            y = splev(t[:-1], tck, der=m)
            cvals[k - m, :] = y / spec.gamma(m + 1)

        return cls.construct_fast(cvals, t, extrapolate)

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        """
        Construct a piecewise polynomial in the power basis
        from a polynomial in Bernstein basis.
        Parameters
        ----------
        bp : BPoly
            A Bernstein basis polynomial, as created by BPoly
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        if not isinstance(bp, BPoly):
            raise TypeError(
                ".from_bernstein_basis only accepts BPoly instances. "
                "Got %s instead." % type(bp)
            )

        dx = np.diff(bp.x)
        k = bp.c.shape[0] - 1  # polynomial order

        rest = (None,) * (bp.c.ndim - 2)

        c = np.zeros_like(bp.c)
        for a in range(k + 1):
            factor = (-1) ** a * comb(k, a) * bp.c[a]
            for s in range(a, k + 1):
                val = comb(k - a, s - a) * (-1) ** s
                c[k - s] += factor * val / dx[(slice(None),) + rest] ** s

        if extrapolate is None:
            extrapolate = bp.extrapolate

        return cls.construct_fast(c, bp.x, extrapolate, bp.axis)
