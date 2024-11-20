.. pyseobnr documentation master file, created by
   sphinx-quickstart on Fri Oct 14 09:49:18 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyseobnr's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/installation
   source/basic_usage
   source/expert_mode
   source/citations

.. the notebooks in the markdown myst format

.. toctree::
   :maxdepth: 1
   :caption: Notebooks

   source/notebooks/getting_started.md
   source/notebooks/example_precession.md
   source/notebooks/pseob_example.md
   source/notebooks/pycbc_example.md

API
---
Throughout the code we refer to several different technical documents, that can be found
`here <https://dcc.ligo.org/LIGO-T2300060/public>`_

.. autosummary::
   :toctree: api
   :template: custom-module-template.rst
   :caption: API:
   :recursive:

   pyseobnr.generate_waveform
   pyseobnr.eob.dynamics
   pyseobnr.eob.waveform
   pyseobnr.eob.fits

   pyseobnr.models
   pyseobnr.auxiliary.mode_mixing.auxiliary_functions_modemixing
