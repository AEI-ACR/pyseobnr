from typing import Any

from pycbc.types import FrequencySeries, TimeSeries

from pyseobnr.generate_waveform import GenerateWaveform


class PySEOBNRv5PyCBCPlugin:
    approximant = None

    @staticmethod
    def _cleanup_parameters(p: dict[str, Any]) -> dict[str, Any]:
        # f_ref is 0 as default value in pyCBC

        for param in ("f_ref", "f_final"):
            if param in p and ((p[param] == 0) or (p[param] is None)):
                p.pop(param)

        return p

    @classmethod
    def gen_td(cls, **p):
        assert cls.approximant is not None
        # Some waveform input parameters from PyCBC have the same naming
        # conventions as PySEOBNR, thus they can be directly inherited.
        # We only update the settings used uniquely by PySEOBNR
        p = p.copy()
        p.update(
            {
                "approximant": cls.approximant,
                "phi_ref": p.pop("coa_phase"),  # reference phase needed by SEOBNRv5
                "f22_start": p.pop("f_lower"),  # starting frequency
                "deltaT": p.pop("delta_t"),
            }
        )
        p = cls._cleanup_parameters(p)

        waveform = GenerateWaveform(p)
        hp, hc = waveform.generate_td_polarizations()

        # Build the PyCBC TimeSeries format
        hp = TimeSeries(hp.data.data[:], delta_t=hp.deltaT, epoch=hp.epoch)
        hc = TimeSeries(hc.data.data[:], delta_t=hc.deltaT, epoch=hc.epoch)

        return hp, hc

    @classmethod
    def gen_fd(cls, **p):
        assert cls.approximant is not None

        # Some waveform input parameters from PyCBC have the same naming
        # conventions as PySEOBNR, thus they can be directly inherited.
        # We only update the settings used uniquely by PySEOBNR
        p = p.copy()
        p.update(
            {
                "approximant": cls.approximant,
                "phi_ref": p.pop("coa_phase"),
                "f22_start": p.pop("f_lower"),
                "deltaF": p.pop("delta_f"),
            }
        )

        p = cls._cleanup_parameters(p)

        if "f_final" in p:
            p["f_max"] = p.pop("f_final")

        waveform = GenerateWaveform(p)
        hp, hc = waveform.generate_fd_polarizations()

        # Build the PyCBC TimeSeries format
        hp = FrequencySeries(hp.data.data[:], delta_f=hp.deltaF, epoch=hp.epoch)
        hc = FrequencySeries(hc.data.data[:], delta_f=hc.deltaF, epoch=hp.epoch)

        return hp, hc


class PySEOBNRv5PyCBCPlugin_v5HM(PySEOBNRv5PyCBCPlugin):
    approximant = "SEOBNRv5HM"


class PySEOBNRv5PyCBCPlugin_v5PHM(PySEOBNRv5PyCBCPlugin):
    approximant = "SEOBNRv5PHM"
