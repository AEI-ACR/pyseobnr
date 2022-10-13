from abc import ABC


class Model(ABC):
    """An abstract base class representing a model.
    Should serve as basis for concrete implementation
    """

    def __init__(self) -> None:
        pass

    def __call__(self):
        """Evaluate the waveform model"""
        pass
