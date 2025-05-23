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

# SEOBNRv5PHM: spin-precessing model example

The notebook contains an example on how to access spin-precessing waveforms \
and dynamical quantities with the SEOBNRv5PHM model.

```{code-cell} ipython3
import warnings

import matplotlib.pyplot as plt
import numpy as np
```

```{code-cell} ipython3
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")  # silence LAL warnings

from pyseobnr.generate_waveform import generate_modes_opt
```

```{code-cell} ipython3
q = 2.0
m1 = q / (1.0 + q)
m2 = 1 - m1
chi_1 = np.array([0.5, 0.0, 0.5])
chi_2 = np.array([0.5, 0.0, 0.5])
omega0 = 0.01
```

Using `generate_modes_opt` with `debug=True` we also get an `SEOBNRv5PHM` object back which contains more information than just the waveform modes.

```{code-cell} ipython3
_, _, model = generate_modes_opt(
    q,
    chi_1,
    chi_2,
    omega0,
    debug=True,
    approximant="SEOBNRv5PHM",
    settings=dict(polarizations_from_coprec=False),
)
```

```{code-cell} ipython3
model
```

The model object does of course contain the waveform mode information

```{code-cell} ipython3
plt.figure(figsize=(8, 6))
plt.plot(model.t, model.waveform_modes["2,2"].real)
plt.xlabel("Time (M)")
plt.ylabel(r"$\Re[h_{22}]$")
plt.grid(True)
```

One also has access to the dynamics, stored as $(t,r,\phi,p_r,p_{\phi},H,\Omega,\Omega_{{\rm circ}},...)$

```{code-cell} ipython3
t_dyn = model.dynamics[:, 0]
r_dyn = model.dynamics[:, 1]
```

```{code-cell} ipython3
plt.figure(figsize=(8, 6))

plt.plot(t_dyn, r_dyn)
plt.xlabel("Time (M)")
plt.ylabel(r"$r$ (M)")
plt.grid(True)
```

The PN precession dynamics as well as the evolution of the PN orbital frequency are stored in `model.pn_dynamics`.Note that we evolve the _dimensionful_ spins, i.e. $$S_{i}=m_{i}^{2}\vec{\chi}_{i}$$

```{code-cell} ipython3
t_PN = model.pn_dynamics[:, 0]
LNhat_PN = model.pn_dynamics[:, 1:4]
chi1_PN = model.pn_dynamics[:, 4:7] / m1**2  # Mind the normalization!
chi2_PN = model.pn_dynamics[:, 7:10] / m2**2  # Mind the normalization!
omega_PN = model.pn_dynamics[:, 10]
```

```{code-cell} ipython3
plt.figure(figsize=(8, 6))

plt.plot(t_PN, chi1_PN, "-")
plt.xlabel("Time (M)")
plt.grid(True)
```

```{code-cell} ipython3
plt.figure(figsize=(8, 6))

plt.plot(t_PN, LNhat_PN, "-")
plt.xlabel("Time (M)")
plt.grid(True)
```

For development and debugging purposes one can also request the co-precessing frame modes to be stored. This can be done as follows:

```{code-cell} ipython3
_, _, model = generate_modes_opt(
    q,
    chi_1,
    chi_2,
    omega0,
    debug=True,
    approximant="SEOBNRv5PHM",
    settings={"return_coprec": True},
)
```

```{code-cell} ipython3
plt.figure(figsize=(8, 6))

plt.plot(model.t, model.waveform_modes["2,1"].real, label="Inertial")
plt.plot(model.t, model.coprecessing_modes["2,1"].real, ls="-", label="Coprecessing")
plt.xlabel("Time (M)")
plt.ylabel(r"$\Re[h_{21}]$")
plt.grid(True, which="both")
plt.legend(loc=2)
```

The Euler angles describing the P->J frame rotation are also stored as follows:

```{code-cell} ipython3
plt.figure(figsize=(8, 6))

plt.plot(model.t, np.unwrap(model.anglesJ2P[0]))
plt.xlabel("Time (M)")
plt.ylabel(r"$\alpha$")
plt.grid(True, which="both")
```

```{code-cell} ipython3
plt.figure(figsize=(8, 6))

plt.plot(model.t, np.unwrap(model.anglesJ2P[1]))
plt.xlabel("Time (M)")
plt.ylabel(r"$\beta$")
plt.grid(True, which="both")
```

```{code-cell} ipython3
plt.figure(figsize=(8, 6))

plt.plot(model.t, np.unwrap(model.anglesJ2P[2]))
plt.xlabel("Time (M)")
plt.ylabel(r"$\gamma$")
plt.grid(True, which="both")
```

## Activation of the anti-symmetric modes

The model `SEOBNRv5PHM` supports the handling of the anti-symmetric modes that should be activated explicitly using the setting `enable_antisymmetric_modes`. The selection of the anti-symmetric modes can be done through `antisymmetric_modes`.

```{code-cell} ipython3
q = 2.0
m1 = q / (1.0 + q)
m2 = 1 - m1
chi_1 = np.array([0.9, 0.0, 0.1])
chi_2 = np.array([-0.5, 0.0, 0.1])
omega0 = 0.01
```

```{code-cell} ipython3
_, _, model = generate_modes_opt(
    q,
    chi_1,
    chi_2,
    omega0,
    debug=True,
    approximant="SEOBNRv5PHM",
    settings={"return_coprec": True},
)

_, _, model_antisym = generate_modes_opt(
    q,
    chi_1,
    chi_2,
    omega0,
    debug=True,
    approximant="SEOBNRv5PHM",
    settings=dict(
        return_coprec=True,
        enable_antisymmetric_modes=True,
        antisymmetric_modes=[(2, 2), (3, 3), (4, 4)],
    ),
)
```

```{code-cell} ipython3
plt.figure(figsize=(8, 6))

plt.plot(model.t, model.waveform_modes["2,2"].real, label="Inertial")
plt.plot(model.t, model_antisym.waveform_modes["2,2"].real, label="Inertial - antisym")
plt.plot(
    model.t,
    (model.waveform_modes["2,2"].real - model_antisym.waveform_modes["2,2"].real) / 2,
    label="Inertial - only antisym",
)

plt.xlabel("Time (M)")
plt.ylabel(r"$\Re[h_{21}]$")
plt.grid(True, which="both")
plt.legend(loc=2)
plt.xlim(-200, 50)
```

```{code-cell} ipython3

```
