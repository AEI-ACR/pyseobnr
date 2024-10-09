---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# pyCBC example

+++

This notebook demonstrate a simple usage of `SEOBNRv5HM` through `pyCBC`.

```{code-cell} ipython3
import warnings

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
```

```{code-cell} ipython3
import lal
import matplotlib.pyplot as pp
import numpy as np
import pycbc
import pycbc.waveform
from pycbc.types import TimeSeries
```

```{code-cell} ipython3
m1 = 50.0
m2 = 30.0

distance = 1.0
inclination = np.pi / 3.0

phiRef = 0.0
s1x, s1y, s1z = 0.0, 0.0, 0.5
s2x, s2y, s2z = 0.0, 0.0, 0.1

dt = 1 / 2048.0
f_max = 1024.0
f_min = 0.0157 / ((m1 + m2) * np.pi * lal.MTSUN_SI)

params_dict = {
    "mass1": m1,
    "mass2": m2,
    "spin1x": s1x,
    "spin1y": s1y,
    "spin1z": s1z,
    "spin2x": s2x,
    "spin2y": s2y,
    "spin2z": s2z,
    "delta_t": dt,  # pyCBC parameter
    # pyCBC parameter:
    # if f_ref not specified, pyCBC will set it to "0"
    # and it will be as if it was not set and defaults to pyseobnr default
    "f_ref": 20,
    "f_lower": f_min,  # pyCBC parameter
    "coa_phase": phiRef,  # pyCBC parameter
    "distance": distance,
    "inclination": inclination,
    # pyCBC parameter: same as f_ref. Only used in get_fd_waveform
    # "f_final": f_max,
    "postadiabatic": False,  # pyseobnr specific parameter
}
```

```{code-cell} ipython3
# Let's plot what our new waveform looks like
hp, hc = pycbc.waveform.get_td_waveform(approximant="SEOBNRv5HM", **params_dict)
```

```{code-cell} ipython3
pp.plot(hp.sample_times, hp)
pp.xlabel("Time (s)");
```

```{code-cell} ipython3
hf = hp.to_frequencyseries()
pp.plot(hf.sample_frequencies, hf.real())
pp.xlabel("Frequency (Hz)")
pp.xscale("log")
pp.xlim(20, 100);
```

Comparison to `pyseobnr` interface

```{code-cell} ipython3
from pyseobnr.generate_waveform import GenerateWaveform
```

```{code-cell} ipython3
params_dict_pyseobnr = {
    "mass1": m1,
    "mass2": m2,
    "spin1x": s1x,
    "spin1y": s1y,
    "spin1z": s1z,
    "spin2x": s2x,
    "spin2y": s2y,
    "spin2z": s2z,
    "deltaT": dt,  # different from pyCBC
    "f_ref": 20,
    "f22_start": f_min,  # different from pyCBC
    "phi_ref": phiRef,  # different from pyCBC
    "distance": distance,
    "inclination": inclination,
    "f_max": f_max,
    "postadiabatic": False,
}

hp_py, hc_py = GenerateWaveform(params_dict_pyseobnr).generate_td_polarizations()
```

```{code-cell} ipython3
t = hp_py.deltaT * np.arange(hp_py.data.length) + hp_py.epoch
pp.plot(t, hp_py.data.data[:])
pp.xlabel("Time (s)");
```

```{code-cell} ipython3
"SEOBNRv5HM" in pycbc.waveform.td_approximants()
```

```{code-cell} ipython3
"SEOBNRv5HM" in pycbc.waveform.fd_approximants()
```

```{code-cell} ipython3
"SEOBNRv5PHM" in pycbc.waveform.td_approximants()
```

```{code-cell} ipython3
"SEOBNRv5PHM" in pycbc.waveform.fd_approximants()
```

```{code-cell} ipython3

```
