import numpy as np
from scipy.interpolate import CubicSpline

class VectorSpline:
    """This is just a cubic spline for every
    vector component separately. Added to make
    it clear what is happening
    """
    def __init__(self, t:np.ndarray, v:np.ndarray):
        self.spline = CubicSpline(t, v)

    def __call__(self, t):
        return self.spline(t)
