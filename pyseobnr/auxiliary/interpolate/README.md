# Rationale

This directory contains a slightly-modified version of Scipy's
[`CubicSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html) class. The main new feature is that the construction of the
interpolant can take in (and return) a set of lookup indices to be
used to figure out which local polynomial is to be used when
evaluating the spline at a given point. The scenario where this is
useful is when you need to interpolate a set of *functions* (all
defined on the same grid) on a common grid. This is precisely what
happens when we interpolate waveform modes to an equally spaced grid,
when finishing the model construction.