|pipeline status|

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

``pyseobnr`` provides state-of-the-art gravitational waveforms using the
effective-one-body (EOB) approach.

For installation instructions, documentation, examples and more, visit the documentation `here <https://waveforms.docs.ligo.org/software/pyseobnr/>`__.

Installation
============

To install, you will need the following non-python dependencies:

* ``hdf5``
* ``gsl``>=2.7
* ``swig``>=4.0.1
* ``fftw3``
* ``lalsuite``


The easiest way to get these is by using ``conda``. First, create a new ``conda`` environment with

::

   conda create -n pyseobnr python=3.9
   conda activate pyseobnr
   conda install -c conda-forge lalsuite

You can install a released version of ``pyseobnr`` by running

::

   pip install pyseobnr



If installing from source, you can do:

::

   pip install -U pip wheel setuptools
   pip install .

Installing dependencies for checks
----------------------------------

If one wants to run sanity checks in ``pyseobnr/auxiliary/sanity_checks``
additional dependencies must be installed. This can be done simply by
running

::

   pip install .[checks]

You will have to install the ``waveform_tools`` from `here <https://bitbucket.org/sergei_ossokine/waveform_tools>`__ manually though,
with e.g. the following command

::

    pip install git+https://bitbucket.org/sergei_ossokine/waveform_tools

License
=======

``pyseobnr`` is released under the GNU General Public License v3.0 or later,
see `here <https://choosealicense.com/licenses/gpl-3.0/>`__ for a
description of this license, or see the
`LICENSE <https://git.ligo.org/waveforms/software/pyseobnr/-/blob/main/LICENSE>`__ file for
the full text.


References
==========

We request that any academic report, publication, or other academic disclosure of results derived from the use of ``pyseobnr`` acknowledge the use of the software by an appropriate acknowledgment or citation.

The code can be cited by citing the `code repository <https://git.ligo.org/waveforms/software/pyseobnr>`_  and the associated publication: Mihaylov et al., "pySEOBNR: a software package for the next generation of
effective-one-body multipolar waveform models", 2023, `arXiv:2303.18203 <https://inspirehep.net/literature/2647812>`_. A bibtex entry is provided::

    @article{Mihaylov:2023bkc,
        author = {Mihaylov, Deyan P. and Ossokine, Serguei and Buonanno, Alessandra and Estelles, Hector and Pompili, Lorenzo and P\"urrer, Michael and Ramos-Buades, Antoni},
        title = "{pySEOBNR: a software package for the next generation of effective-one-body multipolar waveform models}",
        eprint = "2303.18203",
        archivePrefix = "arXiv",
        primaryClass = "gr-qc",
        doi = "10.1016/j.softx.2025.102080",
        journal = "SoftwareX",
        volume = "30",
        pages = "102080",
        year = "2025"
    }


In addition, if released models are used, the model papers should be cited:

* For ``SEOBNRv5PHM``, Ramos-Buades et al., "SEOBNRv5PHM: Next generation of accurate and efficient multipolar precessing-spin effective-one-body waveforms for binary black holes", 2023, `arXiv:2303.18046 <https://inspirehep.net/literature/2647783>`_. If using the version of the model including equatorial multipole asymmetries, please also cite Estellés et al., "Adding equatorial-asymmetric effects for spin-precessing binaries into the SEOBNRv5PHM waveform model", `arXiv:2506.19911 <https://inspirehep.net/literature/2938778>`_.

* For ``SEOBNRv5HM``, Pompili et al., "Laying the foundation of the effective-one-body waveform models SEOBNRv5: improved accuracy and efficiency for spinning non-precessing binary black holes", 2023, `arXiv:2303.18039 <https://inspirehep.net/literature/2647776>`_.

* For ``SEOBNRv5EHM``, Gamboa et al., "Accurate waveforms for eccentric, aligned-spin binary black holes: The multipolar effective-one-body model SEOBNRv5EHM", 2024, `arXiv:2412.12823 <https://inspirehep.net/literature/2860452>`_.

* For ``pSEOBNRv5PHM``, Pompili et al., "A parametrized spin-precessing inspiral-merger-ringdown waveform model for tests of general relativity", 2025, `arXiv:2504.10130 <https://inspirehep.net/literature/2911709>`_.

If you build on the existing models, please cite:

* Khalil et al., "Theoretical groundwork supporting the precessing-spin two-body dynamics of the effective-one-body waveform models SEOBNRv5", 2023, `arXiv:2303.18143 <https://inspirehep.net/literature/2647760>`_.

* Van de Meent et al., "Enhancing the SEOBNRv5 effective-one-body waveform model with second-order gravitational self-force fluxes", 2023, `arXiv:2303.18026 <https://inspirehep.net/literature/2647748>`_.

* Toubiana et al., "Measuring source properties and quasinormal mode frequencies of heavy massive black-hole binaries with LISA", 2024, `arXiv:2307.15086 <https://inspirehep.net/literature/2683211>`_.

* Gamboa et al., "Third post-Newtonian dynamics for eccentric orbits and aligned spins in the effective-one-body waveform model SEOBNRv5EHM", 2024, `arXiv:2412.12831 <https://inspirehep.net/literature/2860457>`_.


.. |pipeline status| image:: https://git.ligo.org/waveforms/software/pyseobnr/badges/main/pipeline.svg
   :target: https://git.ligo.org/waveforms/software/pyseobnr/commits/main

