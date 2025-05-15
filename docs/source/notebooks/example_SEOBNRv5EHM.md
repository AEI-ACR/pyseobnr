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

# SEOBNRv5EHM: aligned spin eccentric model example

This notebook gives examples on how to run the SEOBNRv5EHM waveform generator.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
```

## Using the _pyseobnr_ interface

The following demonstrates how to use the internal SEOBNRv5EHM waveform generator. It takes in parameters in geometric units and returns results in geometric units as well. It also gives access to things beyond the waveform modes, such as the dynamics.

```{code-cell} ipython3
import warnings

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")  # silence LAL warnings
from pyseobnr.generate_waveform import GenerateWaveform, generate_modes_opt
```

```{code-cell} ipython3
# Input parameters
q = 5.3
chi_1 = 0.9
chi_2 = 0.3
omega_start = 0.0137  # This is the orbital frequency in geometric units with M=1
eccentricity = 0.4
rel_anomaly = 2.3
```

## Default mode
We get a time array and a dictionary of complex modes. Note that for *aligned* spins, *only positive m modes are returned*

```{code-cell} ipython3
t, modes = generate_modes_opt(
    q,
    chi_1,
    chi_2,
    omega_start,
    eccentricity=eccentricity,
    rel_anomaly=rel_anomaly,
    approximant="SEOBNRv5EHM",
)
```

```{code-cell} ipython3
modes.keys()
```

```{code-cell} ipython3
plt.figure()
plt.plot(t, modes["2,2"].real)
plt.xlabel("Time (M)")
plt.ylabel(r"$\Re[h_{22}]$")
plt.grid(True)
```

```{code-cell} ipython3
plt.figure()
plt.plot(t, np.abs(modes["2,2"]))
plt.xlabel("Time (M)")
plt.ylabel(r"$|h_{22}|$")
plt.axvline(x=0, ls="--", color="k")
plt.xlim(-1000, 100)
plt.grid(True)
```

By default the aligned spin model sets $t=0$ at the last peak of the frame-invariant amplitude, given a certain threshold

```{code-cell} ipython3
for mode in modes.keys():
    plt.semilogy(t, np.abs(modes[mode]), label=mode)
plt.xlabel("Time (M)")
plt.ylabel(r"$\|h_{\ell m}|$")
plt.xlim(-500, 150)
plt.ylim(1e-5, 1)
plt.legend(loc=3)
plt.grid(True)
```

## Customising the default mode

+++

### Mode array

+++

Suppose you want only the (2,2) and (2,1) modes. This can be done by adding a special argument to the settings dictionary, as follows:

```{code-cell} ipython3
settings = dict(
    # "EccIC" determines the prescription for the starting frequency.
    # EccIC=0 means that omega_start corresponds to the *instantaneous* angular frequency, and
    # EccIC=1 means that omega_start corresponds to the *orbit-averaged* angular frequency [default option]
    EccIC=1,
    # Specify which modes are to be returned,
    return_modes=[(2, 2), (2, 1)],
)
# See the pyseobnr/generate_waveform.py file for a description of more settings

t_new, modes_new = generate_modes_opt(
    q,
    chi_1,
    chi_2,
    omega_start,
    eccentricity=eccentricity,
    rel_anomaly=rel_anomaly,
    approximant="SEOBNRv5EHM",
    settings=settings,
)
```

```{code-cell} ipython3
modes_new.keys()
```

```{code-cell} ipython3
plt.figure()
plt.plot(t_new, np.abs(modes_new["2,1"]))
plt.xlabel("Time (M)")
plt.ylabel(r"$|h_{21}|$")
plt.axvline(x=0, ls="--", color="k")
plt.xlim(-1000, 100)
plt.grid(True)
```

### Backwards evolution

+++

One can also evolve the system backwards in time, by specifying a certain amount of time (in geometric units).
This will evolve backwards the full equations of motion of the system.

This option is particularly useful for post-processing tasks, e.g. with `gw-eccentricity`.

This applies to the modes, but also to the BH dynamics (see below Expert mode).

Note that the warning message can be deactivated to avoid filling log files with these messages.

```{code-cell} ipython3
t, modes = generate_modes_opt(
    q,
    chi_1,
    chi_2,
    omega_start,
    eccentricity=eccentricity,
    rel_anomaly=rel_anomaly,
    approximant="SEOBNRv5EHM",
)

settings = dict(
    t_backwards=1000,  # Geometric units
    # warning_bwd_int=False,  # Setting this to False will avoid the warning message
)

t_back, modes_back = generate_modes_opt(
    q,
    chi_1,
    chi_2,
    omega_start,
    eccentricity=eccentricity,
    rel_anomaly=rel_anomaly,
    approximant="SEOBNRv5EHM",
    settings=settings,
)
```

```{code-cell} ipython3
plt.figure()
plt.plot(t, np.abs(modes["2,2"]), label="Original system")
plt.plot(t_back, np.abs(modes_back["2,2"]), "--", label="Backwards integration")
plt.xlabel("Time (M)")
plt.ylabel(r"$|h_{2}|$")
plt.axvline(x=0, ls="--", color="k")
plt.legend()
plt.grid(True)
```

We note that the model has implemented by default a _secular backwards integration_ which is activated whenever the starting separation of the binary is less than $10M$.
This avoids running into issues with the prescription for initial conditions in configurations where the BHs are very close to each other.

In the following, we see an example where this secular backwards integration is activated.

Note that the warning message can be deactivated to avoid filling log files with these messages.

```{code-cell} ipython3
settings = dict(
    # Setting this to False will avoid the warning message
    # warning_secular_bwd_int=False,
)

t, modes = generate_modes_opt(
    q,
    chi_1,
    chi_2,
    omega_start,
    eccentricity=0.5,
    rel_anomaly=0.0,
    approximant="SEOBNRv5EHM",
    settings=settings,
)
```

```{code-cell} ipython3
plt.figure()
plt.plot(t, np.abs(modes["2,2"]))
plt.xlabel("Time (M)")
plt.ylabel(r"$|h_{2}|$")
plt.axvline(x=0, ls="--", color="k")
plt.grid(True)
```

We can deactivate this secular backwards integration with the following flag: (now, the code will raise an error if the starting separation is below $10M$)

```{code-cell} ipython3
try:
    settings = dict(
        secular_bwd_int=False,
    )

    t, modes = generate_modes_opt(
        q,
        chi_1,
        chi_2,
        omega_start,
        eccentricity=0.5,
        rel_anomaly=0.0,
        approximant="SEOBNRv5EHM",
        settings=settings,
    )
except Exception as e:
    print(e)
```

It is also possible to modify the minimum starting separation.
In the following example, we set the minimum starting separation to $7M$, and hence the secular backward integration is not activated

```{code-cell} ipython3
settings = dict(
    secular_bwd_int=True,
    r_start_min=7.0,
)

t, modes = generate_modes_opt(
    q,
    chi_1,
    chi_2,
    omega_start,
    eccentricity=0.5,
    rel_anomaly=0.0,
    approximant="SEOBNRv5EHM",
    settings=settings,
)
```

One can explore combinations of the values of 'secular_bwd_int' and 'r_start_min'.
However, one must be careful while doing this, specially for high eccentricities or high frequencies.

However, there is a lower limit of minimum starting separation of $6M$. Trying to initialize a binary with a separation below $6M$ will very likely introduce some kind of error. Hence, it is no possible to go below $6M$ with `SEOBNRv5EHM`.

+++

## Expert mode
In this mode we also get an `SEOBNRv5EHM` object back which contains more information than just the waveform modes.

```{code-cell} ipython3
t, modes, model = generate_modes_opt(
    q,
    chi_1,
    chi_2,
    omega_start,
    eccentricity=eccentricity,
    rel_anomaly=rel_anomaly,
    approximant="SEOBNRv5EHM",
    debug=True,
)
```

```{code-cell} ipython3
model
```

The model object does of course contain the waveform mode information:

```{code-cell} ipython3
plt.figure()
plt.plot(model.t, model.waveform_modes["2,2"].real)
plt.xlabel("Time (M)")
plt.ylabel(r"$\Re[h_{22}]$")
plt.grid(True)
```

One also has access to the dynamics, stored as $(t, r, \phi, p_r, p_{\phi}, e, \zeta, x, H,\Omega)$, where

* $t$: time
* $r$: relative separation
* $\phi$: azimuthal orbital angle
* $p_r$: tortoise radial momentum
* $p_\phi$: angular momentum
* $e$: eccentricity
* $\zeta$: relativistic anomaly
* $x$: orbit-averaged frequency to the 2/3 power, i.e. $x=< \omega >^{2/3}$ <br>
  This variable presents oscillations since we compute it with a PN formula, the true orbit-averaged frequency does not have any oscillation
* $H$: EOB Hamiltonian
* $\Omega$: instantaneous angular frequency, i.e. $\Omega=\dot\phi$

```{code-cell} ipython3
t, r, phi, pr, pphi, e, z, x, H, Omega = model.dynamics.T
```

```{code-cell} ipython3
plt.figure()
plt.plot(t, r)
plt.xlabel("Time (M)")
plt.ylabel(r"$r$ (M)")
plt.grid(True)
```

For debugging purposes, other information is also provided, for example, one can examine directly the NQC coefficients:

```{code-cell} ipython3
model.nqc_coeffs
```

## A note on conventions
The internal `SEOBNRv5EHM` generator uses the same conventions as in the `SEOBNRv5HM` model.

+++

## Generate modes and polarizations in physical units with LAL conventions

The GenerateWaveform() class accepts a dictionary of parameters (example below) and from it, one can recover the gravitational modes dictionary with the right convention and physical scaling, the time-domain polarizations and the Fourier-domain polarizations

```{code-cell} ipython3
# start with the usual parameter definitions

m1 = 50.0
m2 = 30.0
s1x = 0.0
s1y = 0.0
s1z = 0.0
s2x = 0.0
s2y = 0.0
s2z = 0.0

eccentricity = 0.3
rel_anomaly = np.pi

deltaT = 1.0 / 1024.0
f_min = 20.0
f_max = 512.0

distance = 1000.0
inclination = np.pi / 3.0
approximant = "SEOBNRv5EHM"

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
    "distance": distance,
    "inclination": inclination,
    "f_max": f_max,
    "return_modes": [
        (2, 2),
        (2, 1),
        (3, 2),
        (3, 3),
        (4, 3),
        (4, 4),
    ],  # Specify which modes are to be returned
    "approximant": approximant,
    "eccentricity": eccentricity,
    "rel_anomaly": rel_anomaly,
    "EccIC": 1,  # EccIC = 0 for instantaneous initial orbital frequency, and EccIC = 1 for orbit-averaged initial orbital frequency
}
```

```{code-cell} ipython3
wfm_gen = GenerateWaveform(params_dict)  # We call the generator with the parameters
```

```{code-cell} ipython3
wfm_gen.parameters  # Access the parameters
```

```{code-cell} ipython3
# Generate mode dictionary
times, hlm = wfm_gen.generate_td_modes()

plt.figure()
plt.plot(times, hlm[(2, 2)].real)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$\Re[h_{22}]$")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(times, hlm[(3, 3)].imag)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$\Im[h_{33}]$")
plt.grid(True)
plt.show()
```

```{code-cell} ipython3
# Generate time-domain polarizations - As LAL REAL8TimeSeries
hp, hc = wfm_gen.generate_td_polarizations()

times = hp.deltaT * np.arange(hp.data.length) + hp.epoch

plt.figure()
plt.plot(times, hp.data.data, label=r"$h_+$")
plt.plot(times, hc.data.data, label=r"$h_{\times}$")
plt.xlabel("Time (seconds)")
plt.legend()
plt.grid(True)
plt.show()
```

```{code-cell} ipython3
# Generate Fourier-domain polarizations - As LAL COMPLEX16FrequencySeries

hpf, hcf = wfm_gen.generate_fd_polarizations()

freqs = hpf.deltaF * np.arange(hpf.data.length)

plt.figure()
plt.plot(freqs, np.abs(hpf.data.data), label=r"$\tilde{h}_+$")
plt.plot(freqs, np.abs(hcf.data.data), label=r"$\tilde{h}_{\times}$")
plt.xlabel("f (Hz)")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.show()
```

### Backwards evolution from `GenerateWaveform`

+++

We also show here the backwards evolution with the `GenerateWaveform` class

```{code-cell} ipython3
m1 = 50.0
m2 = 30.0
eccentricity = 0.3
rel_anomaly = np.pi
approximant = "SEOBNRv5EHM"

params_dict = {
    "mass1": m1,
    "mass2": m2,
    "approximant": approximant,
    "eccentricity": eccentricity,
    "rel_anomaly": rel_anomaly,
}
params_dict_back = {
    "mass1": m1,
    "mass2": m2,
    "approximant": approximant,
    "eccentricity": eccentricity,
    "rel_anomaly": rel_anomaly,
    "t_backwards": 1000,
    # 'warning_bwd_int' : False,  # Setting this to False will avoid the warning message
}

wfm_gen = GenerateWaveform(params_dict)  # We call the generator with the parameters
wfm_gen_back = GenerateWaveform(
    params_dict_back
)  # We call the generator with the parameters
```

```{code-cell} ipython3
# Generate mode dictionary
times, hlm = wfm_gen.generate_td_modes()
times_back, hlm_back = wfm_gen_back.generate_td_modes()
```

```{code-cell} ipython3
plt.figure()
plt.plot(times, np.abs(hlm[(2, 2)]), label="Original system")
plt.plot(times_back, np.abs(hlm_back[(2, 2)]), "--", label="Backwards integration")
plt.xlabel("Time (seconds)")
plt.ylabel(r"$\Re[h_{22}]$")
plt.grid(True)
plt.legend()
plt.show()
```

Note that we can also control the secular backwards integration behavior within the GenerateWaveform class.

```{code-cell} ipython3
m1 = 50.0
m2 = 30.0
eccentricity = 0.4
rel_anomaly = 0
approximant = "SEOBNRv5EHM"

params_dict = {
    "mass1": m1,
    "mass2": m2,
    "approximant": approximant,
    "eccentricity": eccentricity,
    "rel_anomaly": rel_anomaly,
    # 'secular_bwd_int' : False,  # Setting this to False will deactivate the secular bwd integration
    # 'warning_secular_bwd_int' : False,  # Setting this to False will avoid the warning message
}

wfm_gen = GenerateWaveform(params_dict)
```

```{code-cell} ipython3
# Generate mode dictionary
times, hlm = wfm_gen.generate_td_modes()
```

```{code-cell} ipython3
plt.figure()
plt.plot(times, np.abs(hlm[(2, 2)]))
plt.xlabel("Time (seconds)")
plt.ylabel(r"$\Re[h_{22}]$")
plt.grid(True)
plt.show()
```

## Using the `gwsignal` interface (new waveform interface)
Unlike the internal generator, the interface can accept a much wider variety of inputs both in SI and so-called 'cosmo' units (where say masses are in solar masses). This interface also returns the modes and polarizations in SI units and follows `LAL` conventions.

```{code-cell} ipython3
import astropy.units as u
from lalsimulation.gwsignal import (
    GenerateFDWaveform,
    GenerateTDModes,
    GenerateTDWaveform,
)
from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator
```

```{code-cell} ipython3
# start with the usual parameter definitions

# Masses
m1 = 30.0
m2 = 30.0

# spings
s1x = 0.0
s1y = 0.0
s1z = 0.5
s2x = 0.0
s2y = 0.0
s2z = -0.2

# Extrinsic parameters
deltaT = 1.0 / 4096.0 / 2.0
f_min = 10.0
f_ref = 10.0
luminosity_distance = 1000.0

iota = np.pi / 3.0
phase = 0.0

# Eccentric parameters
eccentricity = 0.2
longitude_ascending_nodes = 0.0
meanPerAno = 0.5

# Conditioning
condition = 1
deltaF = 1.0 / 8.0  # /20.
# Create dict for gwsignal generator
gwsignal_dict = {
    "mass1": m1 * u.solMass,
    "mass2": m2 * u.solMass,
    "spin1x": s1x * u.dimensionless_unscaled,
    "spin1y": s1y * u.dimensionless_unscaled,
    "spin1z": s1z * u.dimensionless_unscaled,
    "spin2x": s2x * u.dimensionless_unscaled,
    "spin2y": s2y * u.dimensionless_unscaled,
    "spin2z": s2z * u.dimensionless_unscaled,
    #'deltaF' : delta_frequency * u.Hz,
    "deltaT": deltaT * u.s,
    #'deltaF': deltaF*u.Hz,
    "f22_start": f_min * u.Hz,
    "f_max": f_max * u.Hz,
    "f22_ref": f_ref * u.Hz,
    "phi_ref": phase * u.rad,
    "distance": luminosity_distance * u.Mpc,
    "inclination": iota * u.rad,
    "eccentricity": eccentricity * u.dimensionless_unscaled,
    "longAscNodes": longitude_ascending_nodes * u.rad,
    "meanPerAno": meanPerAno * u.rad,
    # 'ModeArray': mode_array,
    "condition": 0,
    "lmax": 4,
    "lmax_nyquist": 4,
    # Some additional specific SEOBNRv5EHM parameters
    "secular_bwd_int": True,
    "warning_secular_bwd_int": False,
    "t_backwards": 0,
    "warning_bwd_int": False,
}

waveform_approximant = "SEOBNRv5EHM"

try:
    wf_gen = gwsignal_get_waveform_generator(waveform_approximant)
except ValueError as e:
    if str(e) != "Approximant not implemented in GWSignal!":
        raise

    wf_gen = None
    print("SEOBNRv5EHM not supported by this version of lal")
```

```{code-cell} ipython3
if wf_gen is not None:
    hpc = GenerateTDWaveform(gwsignal_dict, wf_gen)
    time = np.arange(len(hpc.hp)) * gwsignal_dict["deltaT"]

    plt.plot(time, hpc.hp, label=r"$h_p$")
    plt.plot(time, hpc.hc, label=r"$h_x$")
    plt.xlabel("Time (seconds)")
    plt.ylabel(r"$h_{x,p}$")
    plt.show()
else:
    print("Plot skipped")
```

```{code-cell} ipython3
if wf_gen is not None:
    # Generate TD modes
    hlm = GenerateTDModes(gwsignal_dict, wf_gen)

    l, m = 2, -1
    plt.plot(time, np.real(hlm[(l, m)]), label=f"({l,m})")
    plt.plot(time, np.real(hlm[(l, -m)]), ls="--", label=f"({l,-m})")
    plt.xlabel("Time (seconds)")
    plt.ylabel(r"$\Re[h_{22}]$")
    plt.legend()
    plt.show()
else:
    print("Plot skipped")
```

```{code-cell} ipython3
deltaF = 1.0 / 9
# Create dict for gwsignal generator
gwsignal_dict = {
    "mass1": m1 * u.solMass,
    "mass2": m2 * u.solMass,
    "spin1x": s1x * u.dimensionless_unscaled,
    "spin1y": s1y * u.dimensionless_unscaled,
    "spin1z": s1z * u.dimensionless_unscaled,
    "spin2x": s2x * u.dimensionless_unscaled,
    "spin2y": s2y * u.dimensionless_unscaled,
    "spin2z": s2z * u.dimensionless_unscaled,
    #'deltaF' : delta_frequency * u.Hz,
    #'deltaT': deltaT*u.s,
    "deltaF": deltaF * u.Hz,
    "f22_start": f_min * u.Hz,
    "f_max": f_max * u.Hz,
    "f22_ref": f_ref * u.Hz,
    "phi_ref": phase * u.rad,
    "distance": luminosity_distance * u.Mpc,
    "inclination": iota * u.rad,
    "eccentricity": eccentricity * u.dimensionless_unscaled,
    "longAscNodes": longitude_ascending_nodes * u.rad,
    "meanPerAno": meanPerAno * u.rad,
    # 'ModeArray': mode_array,
    "condition": 1,
    "lmax": 4,
    "lmax_nyquist": 4,
    # Some additional specific SEOBNRv5EHM parameters
    "secular_bwd_int": True,
    "warning_bwd_int": False,
    "warning_secular_bwd_int": False,
    "t_backwards": 0,
}

waveform_approximant = "SEOBNRv5EHM"

try:
    wf_gen = gwsignal_get_waveform_generator(waveform_approximant)
except ValueError as e:
    if str(e) != "Approximant not implemented in GWSignal!":
        raise

    wf_gen = None
    print("SEOBNRv5EHM not supported by this version of lal")
```

```{code-cell} ipython3
if wf_gen is not None:
    hpc = GenerateFDWaveform(gwsignal_dict, wf_gen)
    freqs = np.arange(len(hpc.hp)) * gwsignal_dict["deltaF"]
    plt.plot(freqs, np.abs(hpc.hp), label=r"$h_p$")
    plt.plot(freqs, np.abs(hpc.hc), label=r"$h_x$")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"$h_{p,x}$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)

    plt.show()
else:
    print("Plot skipped")
```

```{code-cell} ipython3

```
