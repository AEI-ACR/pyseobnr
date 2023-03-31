Using pyseobnr
==============

Standard interface
------------------

``pyseobnr`` provides an interface that closely follows the conventions used in in the
new waveform interface, `gwsignal <>`_.

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

    See also the relevant documentation in `gwsignal <>`_.

Here is a simple example to get the modes:

.. code-block:: python

    # Start with the usual parameter definitions

    # Masses in solar masses
    >>> m1 = 50.
    m2 = 30.
    s1x,s1y,s1z = 0.,0.,0.5
    s2x,s2y,s2z = 0.,0.,0.8

    deltaT = 1./1024.
    f_min = 20.
    f_max = 512.

    distance = 1000. # Mpc
    inclination = np.pi/3.
    phiRef = 0.
    approximant = "SEOBNRv5HM"

    params_dict = {'mass1' : m1,
                'mass2' : m2,
                'spin1x' : s1x,
                'spin1y' : s1y,
                'spin1z' : s1z,
                'spin2x' : s2x,
                'spin2y' : s2y,
                'spin2z' : s2z,
                'deltaT' : deltaT,
                'f22_start' : f_min,
                'phi_ref' : phiRef,
                'distance' : distance,
                'inclination' : inclination,
                'f_max' : f_max,
                'approximant' : approximant}

    wfm_gen = GenerateWaveform(params_dict) # We call the generator with the parameters
    # Generate mode dictionary
    times, hlm = wfm_gen.generate_td_modes()

    # Plot some modes

    plt.figure()
    plt.plot(times,hlm[(2,2)].real)
    plt.xlabel("Time (seconds)")
    plt.ylabel(r"$\Re[h_{22}]$")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(times,hlm[(3,3)].imag)
    plt.xlabel("Time (seconds)")
    plt.ylabel(r"$\Im[h_{33}]$")
    plt.grid(True)
    plt.show()


To get the polarizations in the frquency domain:


.. code-block:: python


    # Generate Fourier-domain polarizations - As LAL COMPLEX16FrequencySeries

    hpf, hcf = wfm_gen.generate_fd_polarizations()
    freqs = hpf.deltaF*np.arange(hpf.data.length)

EOB internal interface
----------------------

Internally, ``pyseobnr`` computes the waveforms in geometric units and follows sligtly
different convetions (that agree with previous models in the ``SEOBNR`` family).
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