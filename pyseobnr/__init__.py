from .generate_waveform import GenerateWaveform  # noqa

try:
    from ._version import version as __version__
except ModuleNotFoundError:
    __version__ = ""
