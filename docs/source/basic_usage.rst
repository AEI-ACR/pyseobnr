Using pyseobnr
==============

Standard interface
------------------

``pyseobnr`` provides an interface that closely follows the conventions used in the
`gwsignal`_ waveform interface.

In this interface, one first constructs a :py:class:`GenerateWaveform
<pyseobnr.generate_waveform.GenerateWaveform>` class which serves as a container for
waveform information. This class then allows one to:

- compute the time-domain waveform modes via :py:meth:`generate_td_modes
  <pyseobnr.generate_waveform.GenerateWaveform.generate_td_modes>`
- compute the time-domain polarizations via :py:meth:`generate_td_polarizations
  <pyseobnr.generate_waveform.GenerateWaveform.generate_td_polarizations>`
- compute the frequency-domain polarizations via :py:meth:`generate_fd_polarizations
  <pyseobnr.generate_waveform.GenerateWaveform.generate_fd_polarizations>`

The input parameters to :py:class:`GenerateWaveform
<pyseobnr.generate_waveform.GenerateWaveform>` are expected to be in the so-called `cosmo`
units. The most salient point is that masses are expected in `solar masses` and distance in
Mpc. For details see :py:class:`GenerateWaveform
<pyseobnr.generate_waveform.GenerateWaveform>` docstring.

.. tip::

    See also the relevant documentation in `gwsignal`_.

.. _gwsignal: https://waveforms.docs.ligo.org/reviews/lalsuite/lalsimulation/gwsignal/index.html

Here is a simple example to get the modes:

.. code-block:: python

    import numpy as np
    from pyseobnr.generate_waveform import GenerateWaveform

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

    # We call the generator with the parameters
    wfm_gen = GenerateWaveform(params_dict)
    # Generate mode dictionary
    times, hlm = wfm_gen.generate_td_modes()

To get the polarizations in the frequency domain:


.. code-block:: python

    # Generate Fourier-domain polarizations - As LAL COMPLEX16FrequencySeries

    hpf, hcf = wfm_gen.generate_fd_polarizations()
    freqs = hpf.deltaF*np.arange(hpf.data.length)


The notebook below gives a complete example on how to use ``pyseobnr``:

.. nblinkgallery::
    :name: notebooks-introduction

    notebooks/getting_started.md



EOB internal interface
----------------------

Internally, ``pyseobnr`` computes the waveforms in geometric units and follows slightly
different conventions (that agree with previous models in the ``SEOBNR`` family).
The output is a numpy array of times and a dictionary of modes. Note that for aligned-spin,
the internal EOB generator only outputs modes with :math:`m>0`.

.. code-block:: python

    >>> from pyseobnr.generate_waveform import generate_modes_opt
    >>> q = 5.3
    >>> chi_1 = 0.9
    >>> chi_2 = 0.3
    >>> omega0 = 0.0137 # This is the orbital frequency in geometric units with M=1
    >>> t,modes = generate_modes_opt(q,chi_1,chi_2,omega0)
    >>> modes.keys()
    dict_keys(['2,2', '2,1', '3,3', '3,2', '4,4', '4,3', '5,5'])

Usage through ``pyCBC``
-----------------------

.. versionadded:: 0.2.13

It is possible to use the approximants implemented in ``pyseobnr`` from ``pyCBC``
directly thanks to the ``pyCBC`` plugin infrastructure.

Installing ``pyseobnr`` will automatically create the required ``pyCBC`` plugins,
which will translate the pyCBC parameters into ``pyseobnr`` compatible ones. Note that
installing ``pyseobnr`` will not install ``pyCBC``.

.. nblinkgallery::
    :name: notebooks-graph-reduction

    notebooks/pycbc_example.md


.. seealso::

   pyCBC plugin infrastructure `pycbc plugin`_.

.. _pycbc plugin: https://pycbc.org/pycbc/latest/html/waveform_plugin.html
