---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# PySEOBNR introduction

The notebook shows how to get started with `pyseobnr`.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import warnings

# silence warnings coming from LAL
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import numpy as np
from matplotlib import pyplot as plt

# import the library
from pyseobnr.generate_waveform import GenerateWaveform
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# Start with the usual parameter definitions
# Masses in solar masses
m1 = 50.0
m2 = 30.0
s1x, s1y, s1z = 0.0, 0.0, 0.5
s2x, s2y, s2z = 0.0, 0.0, 0.8

deltaT = 1.0 / 2048.0
f_min = 20.0
f_max = 1024.0

distance = 1000.0  # Mpc
inclination = np.pi / 3.0
phiRef = 0.0
approximant = "SEOBNRv5HM"

params_dict = {
    "mass1": m1,
    "mass2": m2,
    "spin1x": s1x,
    "spin1y": s1y,
    "spin1z": s1z,
    "spin2x": s2x,
    "spin2y": s2y,
    "spin2z": s2z,
    "deltaT": deltaT,
    "f22_start": f_min,
    "phi_ref": phiRef,
    "distance": distance,
    "inclination": inclination,
    "f_max": f_max,
    "approximant": approximant,
}
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# We call the generator with the parameters
wfm_gen = GenerateWaveform(params_dict)

# Generate mode dictionary
times, hlm = wfm_gen.generate_td_modes()
```

Plot some modes

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
plt.figure()
plt.plot(times, hlm[(2, 2)].real)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$\Re[h_{22}]$")
plt.grid(True)
plt.show()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
plt.figure()
plt.plot(times, hlm[(3, 3)].imag)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$\Im[h_{33}]$")
plt.grid(True)
plt.show()
```

## Expert mode

```{code-cell} ipython3
from pyseobnr.generate_waveform import generate_modes_opt

q = 5.3
chi_1 = 0.9
chi_2 = 0.3
omega0 = 0.0137  # This is the orbital frequency in geometric units with M=1

_, _, model = generate_modes_opt(q, chi_1, chi_2, omega0, debug=True)
model
```

```{code-cell} ipython3
t, r, phi, pr, pphi, H, Omega, _ = model.dynamics.T
```

```{code-cell} ipython3
import pandas as pd

frame_dynamics = pd.DataFrame(
    data=model.dynamics,
    columns="t, r, phi, pr, pphi, H, Omega, Omega_circular".replace(" ", "").split(","),
)
frame_dynamics
```

## Selecting different convention for reference phase and $t=0$

By default, `pyseobnr` will generate the waveforms with the following conventions
for phase and time:

* at the reference frequency, the orbital phase will be set to 0,
* the origin of the time axis, $t=0$, is set to coincide with the peak of the frame invariant amplitude $A$:

    $$
    A(t) = \sqrt{\sum_{\ell m}|h_{\ell m}(t)|}\newline
    $$

    $${\rm argmax}_t A = 0$$

Other models, in particular the `IMRPhenomT` models, follow a different
convention for setting the reference phase and the time origin:

* at the reference frequency, the phase
of the co-precessing dominant $(2,2)$ mode is set to 0, while the other modes
are rotated consistenly to preserve the correct relative phase with
the dominant mode,
* the time origin is set to coincide with the peak amplitude of the co-precessing $(2,2)$ mode.

One can set this second convention in pyseobnr via the following options:

* `convention_coprecessing_phase22_set_to_0_at_reference_frequency` to `True`
* `convention_t0_set_to_0_at_coprecessing_amplitude22_peak` to `True`

Note that when `convention_coprecessing_phase22_set_to_0_at_reference_frequency` is `True` as well as (for precessing systems) `polarizations_from_coprec` is `True`, the phase rotations will be applied directly to the spin weighted spherical harmonics.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# Start with the usual parameter definitions
# Masses in solar masses
m1 = 50.0
m2 = 10.0
s1x, s1y, s1z = 0.0, 0.0, 0.5
s2x, s2y, s2z = 0.0, 0.0, 0.8

deltaT = 1.0 / 2048.0
f_min = 20.0
f_max = 1024.0

distance = 1000.0  # Mpc
inclination = np.pi / 3.0
phiRef = 0.0
approximant = "SEOBNRv5HM"

params_dict = {
    "mass1": m1,
    "mass2": m2,
    "spin1x": s1x,
    "spin1y": s1y,
    "spin1z": s1z,
    "spin2x": s2x,
    "spin2y": s2y,
    "spin2z": s2z,
    "deltaT": deltaT,
    "f22_start": f_min,
    "phi_ref": phiRef,
    "distance": distance,
    "inclination": inclination,
    "f_max": f_max,
    "approximant": approximant,
}

# We call the generator with the parameters
wfm_gen = GenerateWaveform(params_dict)

# We update with the new convention
wfm_gen_2 = GenerateWaveform(
    params_dict
    | {
        "convention_coprecessing_phase22_set_to_0_at_reference_frequency": True,
        "convention_t0_set_to_0_at_coprecessing_amplitude22_peak": True,
    }
)

# Generate mode dictionary
times_0, hlms_0 = wfm_gen.generate_td_modes()
times_1, hlms_1 = wfm_gen_2.generate_td_modes()


plt.figure()
plt.plot(times_0, hlms_0[(2, 2)].real, label="default pyseobnr convention")
plt.plot(times_1, hlms_1[(2, 2)].real, label="IMRPhenomT convention", ls="dashed")
plt.xlabel("Time (seconds)")
plt.ylabel(r"$\Re[h_{22}]$")
plt.legend()
plt.xlim([-0.05, 0.01])
plt.grid(True)
plt.tight_layout()
```

```{code-cell} ipython3

```
