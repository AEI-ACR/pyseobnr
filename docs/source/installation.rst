Installation
============

Quick guide for installing ``pyseobnr``.

.. _install:

Install pyseobnr from source
----------------------------

To install ``pyseobnr`` you will need a few non-python dependencies, at a minimum:

- python >=3.8, <3.11
- gcc
- gsl
- swig

All of these are easily installable with ``conda`` (or available in standard ``conda``
envs provided by ``IGWN``)

Next clone the ``pyseobnr`` repository and install via ``pip`` . We recommend using a
clean environment (either a ``venv`` or a ``conda`` environment).

.. code-block:: console

    (.venv) $ git clone git@git.ligo.org:serguei.ossokine/pyseobnr.git
    (.venv) $ cd pyseobnr
    (.venv) $ pip install -U pip wheel setuptools numpy
    (.venv) $ pip install .

.. tip::

    If you are actively developing the code, consider using ``pip install -e .`` to
    avoid having to reinstall after ever change to the code.

Installing optional dependencies to run checks
----------------------------------------------

If one wants to run sanity checks in ``pyseobnr/auxilary/sanity_checks``, additional
dependencies must be installed. This can be done simply by running

.. code-block:: console

    (.venv) $ pip install .[checks]

Installing optional dependencies to build documentation
-------------------------------------------------------

To build documentation, install the relevant dependencies with

.. code-block:: console

    (.venv) $ pip install .[docs]

Then the documentation can be built via

.. code-block:: console

    (.venv) $ cd docs
    (.venv) $ make html
