from abc import ABC


class Model(ABC):
    """An abstract base class representing a model.
    Should serve as basis for concrete implementation
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self):
        """Evaluate the waveform model"""
        pass
