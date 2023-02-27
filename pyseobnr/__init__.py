from . import eob
from .eob.dynamics import *
from .eob.hamiltonian import *
from .eob.fits import *
from . import generate_waveform


try:
    from ._version import version as __version__
except ModuleNotFoundError:
    __version__ = ""
