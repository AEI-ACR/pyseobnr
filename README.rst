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



If installing from source,  you can do:

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

You will have to install the ``waveform`` tools from `here <https://bitbucket.org/sergei_ossokine/waveform_tools>`__ manually though
with eg. the following command

::

    pip install git+https://bitbucket.org/sergei_ossokine/waveform_tools

License
=======

``pyseobnr`` is released under the GNU General Public License v3.0 or later,
see `here <https://choosealicense.com/licenses/gpl-3.0/>`__ for a
description of this license, or see the
`LICENSE <https://github.com/gwpy/gwpy/blob/main/LICENSE>`__ file for
the full text.


References
==========

We request that any academic report, publication, or other academic disclosure of results derived from the use of ``pyseobnr`` acknowledge the use of the software by an appropriate acknowledgment or citation.

The code can be cited by citing `code repo <https://git.ligo.org/waveforms/software>`_  and the code paper: Mihaylov et al, "pySEOBNR: a software package for the next generation of 
effective-one-body multipolar waveform models", 2023, `arXiv:2303.18203 <https://arxiv.org/abs/2303.18203>`_. A bibtex entry is provided::

  @article{Mihaylov:2023bkc,
    author = {Mihaylov, Deyan P. and Ossokine, Serguei and Buonanno, Alessandra and Estelles, Hector and Pompili, Lorenzo and P\"urrer, Michael and Ramos-Buades, Antoni},
    title = "{pySEOBNR: a software package for the next generation of effective-one-body multipolar waveform models}",
    eprint = "2303.18203",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "3",
    year = "2023"
  }



In addition, if released models are used, the model papers should be cited:

* For SEOBNRv5PHM, Ramos-Buades et al, "SEOBNRv5PHM: Next generation of accurate and efficient multipolar precessing-spin effective-one-body waveforms for binary black holes", 2023, `arXiv:2303.18046 <https://arxiv.org/abs/2303.18046>`_::

    @article{Ramos-Buades:2023ehm,
    author = "Ramos-Buades, Antoni and Buonanno, Alessandra and Estell\'es, H\'ector and Khalil, Mohammed and Mihaylov, Deyan P. and Ossokine, Serguei and Pompili, Lorenzo and Shiferaw, Mahlet",
    title = "{SEOBNRv5PHM: Next generation of accurate and efficient multipolar precessing-spin effective-one-body waveforms for binary black holes}",
    eprint = "2303.18046",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "3",
    year = "2023"
    }
    
* For SEOBNRv5HM, Pompili et al, "Laying the foundation of the effective-one-body waveform models SEOBNRv5: improved accuracy and efficiency for spinning non-precessing binary black holes", 2023, `arXiv:2303.18039 <https://arxiv.org/abs/2303.18039>`_::

    @article{Pompili:2023tna,
    author = "Pompili, Lorenzo and others",
    title = "{Laying the foundation of the effective-one-body waveform models SEOBNRv5: improved accuracy and efficiency for spinning non-precessing binary black holes}",
    eprint = "2303.18039",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "3",
    year = "2023"
    }

If you build on the existing models, please cite:

* Khalil et al, "Theoretical groundwork supporting the precessing-spin two-body dynamics of the effective-one-body waveform models SEOBNRv5", 2023, `arXiv:2303.18143 <https://arxiv.org/abs/2303.18143>`_::

    @article{Khalil:2023kep,
    author = "Khalil, Mohammed and Buonanno, Alessandra and Estell\'es, H\'ector and Mihaylov, Deyan P. and Ossokine, Serguei and Pompili, Lorenzo and Ramos-Buades, Antoni",
    title = "{Theoretical groundwork supporting the precessing-spin two-body dynamics of the effective-one-body waveform models SEOBNRv5}",
    eprint = "2303.18143",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "3",
    year = "2023"
    }


* Van de Meent et al, "Enhancing the SEOBNRv5 effective-one-body waveform model with second-order gravitational self-force fluxes", 2023, `arXiv:2303.18026 <https://arxiv.org/abs/2303.18026>`_::

    @article{vandeMeent:2023ols,
    author = "van de Meent, Maarten and Buonanno, Alessandra and Mihaylov, Deyan P. and Ossokine, Serguei and Pompili, Lorenzo and Warburton, Niels and Pound, Adam and Wardell, Barry and Durkan, Leanne and Miller, Jeremy",
    title = "{Enhancing the SEOBNRv5 effective-one-body waveform model with second-order gravitational self-force fluxes}",
    eprint = "2303.18026",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "3",
    year = "2023"
    }


.. |pipeline status| image:: https://git.ligo.org/waveforms/software/pyseobnr/badges/main/pipeline.svg
   :target: https://git.ligo.org/waveforms/software/pyseobnr/commits/main

