|pipeline status|

pyseobnr provides state-of-the-art gravitational waveforms using the
effective-one-body (EOB) approach.

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
   conda install lalsuite

You can install a released version of ``pyseobnr`` by running

::

   pip install pyseobnr



If installing from source,  you can do:

::

   pip install -U pip wheel setuptools
   pip install .

Installing dependencies for checks
----------------------------------

If one wants to run sanity checks in ``pyseobnr/auxilary/sanity_checks``
additional dependencies must be installed. This can be done simply by
running

::

   pip install .[checks]

License
=======

pyseobnr is released under the GNU General Public License v3.0 or later,
see `here <https://choosealicense.com/licenses/gpl-3.0/>`__ for a
description of this license, or see the
`LICENSE <https://github.com/gwpy/gwpy/blob/main/LICENSE>`__ file for
the full text.



We request that any academic report, publication, or other academic disclosure of results derived from the use of ``pyseobnr`` acknowledge the use of the software by an appropriate acknowledgment or citation.

The code can be cited by citing `code repo <https://git.ligo.org/waveforms/software>`_  and the code paper: Mihaylov et al, "pySEOBNR: a software package for the next generation of effective-one-body multipolar waveform models", 2023

In addition, if released models are used, the model papers should be cited:

* For SEOBNRv5PHM, Ramos-Buades et al, "SEOBNRv5PHM: Next generation of accurate and efficient multipolar precessing-spin effective-one-body waveforms for binary black holes", 2023
* For SEOBNRv5HM, Pompili et al, "Laying the foundation of the effective-one-body waveform models SEOBNRv5: improved accuracy and efficiency for spinning non-precessing binary black holes", 2023

If you build on the existing models, please cite:

* Khalil et al, "Theoretical groundwork supporting the precessing-spin two-body dynamics of the effective-one-body waveform models SEOBNRv5", 2023
* Van de Meent et al, "Enhancing the SEOBNRv5 effective-one-body waveform model with second-order gravitational self-force fluxes", 2023


.. |pipeline status| image:: https://git.ligo.org/serguei.ossokine/pyseobnr/badges/main/pipeline.svg
   :target: https://git.ligo.org/serguei.ossokine/pyseobnr/commits/main
