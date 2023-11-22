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

## pSEOB example

The notebook contains an example on how to use the parametrized model pSEOBNRv5HM \
to generate a waveform with a GR deviation in the quasi-normal-mode damping time.

```{code-cell} ipython3
import warnings
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

# silence warnings coming from LAL
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
```

```{code-cell} ipython3
# Import the library
from pyseobnr.generate_waveform import GenerateWaveform
```

```{code-cell} ipython3
# Start with the usual parameter definitions
# Masses in solar masses
m1 = 50.0
m2 = 30.0
s1x, s1y, s1z = 0.0, 0.0, 0.5
s2x, s2y, s2z = 0.0, 0.0, 0.8

deltaT = 1.0 / 4096.0
deltaF = 1.0 / 16
f_min = 20.0
f_max = 2048.0

distance = 1000.0  # Mpc
inclination = np.pi / 3.0
phiRef = 0.0
approximant = "SEOBNRv5HM"
ModeArray = [(2, 2), (3, 3)]

dev = 0.3

# Fractional deviations for each mode
deviation_dict = {
    "2,2": dev,
    "2,1": 0.0,
    "3,3": dev,
    "3,2": 0.0,
    "4,4": 0.0,
    "4,3": 0.0,
    "5,5": 0.0,
}
```

```{code-cell} ipython3
# Combine parameters in a dictionary
params_GR = {
    "mass1": m1,
    "mass2": m2,
    "spin1x": s1x,
    "spin1y": s1y,
    "spin1z": s1z,
    "spin2x": s2x,
    "spin2y": s2y,
    "spin2z": s2z,
    "distance": distance,
    "inclination": inclination,
    "phi_ref": phiRef,
    "f22_start": f_min,
    "f_ref": f_min,
    "f_max": f_max,
    "deltaT": deltaT,
    "deltaF": deltaF,
    "approximant": approximant,
    "ModeArray": ModeArray,
}

# Add parametrized deviations to the parameter dictionary
params_dtau = deepcopy(params_GR)
params_dtau["dtau_dict"] = deviation_dict
```

```{code-cell} ipython3
# We call the generator with the parameters
wfm_gen = GenerateWaveform(params_GR)
wfm_gen_dtau = GenerateWaveform(params_dtau)
```

```{code-cell} ipython3
# Generate time-domain modes dictionary
times, hlm = wfm_gen.generate_td_modes()
times_dtau, hlm_dtau = wfm_gen_dtau.generate_td_modes()
```

```{code-cell} ipython3
plt.plot(times, np.abs(hlm[(2, 2)]), label="GR")
plt.plot(times_dtau, np.abs(hlm_dtau[(2, 2)]), label=r"$d \tau_{22} = " + f"{dev}$")

plt.xlabel(r"$t$[s]")
plt.ylabel(r"$|h_{22}|$")
plt.xlim(-0.05, 0.05)
plt.grid()
plt.legend();
```

```{code-cell} ipython3
# Generate Fourier-domain polarizations
hp, hc = wfm_gen.generate_fd_polarizations()
hp_dtau, hc_dtau = wfm_gen_dtau.generate_fd_polarizations()

freqs = hp.deltaF * np.arange(hp.data.length)
```

```{code-cell} ipython3
plt.plot(freqs, np.abs(hp.data.data), label="GR")
plt.plot(
    freqs,
    np.abs(hp_dtau.data.data),
    label=r"$\delta \tau_{22} = \delta \tau_{33} = " + f"{dev}$",
)

plt.xlabel("$f [Hz]$")
plt.ylabel(r"$\tilde{h}_+$")
plt.xscale("log")
plt.yscale("log")
plt.xlim(1.0)
plt.grid()
plt.legend();
```
