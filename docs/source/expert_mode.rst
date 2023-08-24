Expert mode and debugging
=========================

The expert mode is intended for users who either want to debug the models or for those
seeking results beyond the waveform modes. In this mode, the EOB waveform generator will
return an extra output which is a model class that contains many additional details of
the model.

.. caution::

    Results in expert mode may not follow LAL conventions. Always check the
    documentation about the assumptions of the output.

One can invoke this as follows:

.. code-block:: python

    >>> from pyseobnr.generate_waveform import generate_modes_opt
    >>> q = 5.3
    >>> chi_1 = 0.9
    >>> chi_2 = 0.3
    >>> omega0 = 0.0137 # This is the orbital frequency in geometric units with M=1
    >>> _, _, model = generate_modes_opt(q, chi_1, chi_2, omega0, debug=True)
    >>> model
    <pyseobnr.models.SEOBNRv5HM.SEOBNRv5HM_opt at 0x7f876766c0a0>

The model object contains a lot of information. Broadly speaking these can be split
into:

- Inputs and derived quantities: e.g masses, spins, etc and transformations thereof
- Auxiliary quantities: final state quantities, various fits used in the model
- Intermediate results: e.g. waveform modes in particular frames
- Additional output: e.g. the dynamics

The last category is of most interest to those that want additional information from
the mode. As an example, one can easily access the dynamics with

.. code-block:: python

    t, r, phi, pr, pphi, H, Omega, _ = model.dynamics.T

.. note::

    The dynamics are represented internally in terms of *rescaled* quantities, i.e. the
    momenta are appropriate scaled by :math:`\mu`, while :math:`H` is scaled by :math:`\nu`. The code also
    internally uses the convention that the total mass of the system is 1.
