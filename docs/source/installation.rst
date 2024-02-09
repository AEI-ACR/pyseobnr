Installation
============

Quick guide for installing ``pyseobnr``.

.. _install:

Install pyseobnr from source
----------------------------

To install ``pyseobnr`` you will need a few non-python dependencies, at a minimum:

- ``python >=3.9, <3.12``
- ``gcc``
- ``gsl``
- ``swig``

All of these are easily installable with ``conda`` (or available in standard ``conda``
envs provided by ``IGWN``), or your favorite package manager.

Next clone the ``pyseobnr`` repository and install via ``pip`` . We recommend using a
clean environment (either a ``venv`` or a ``conda`` environment).

.. code-block:: console

    (.venv) $ git clone https://git.ligo.org/waveforms/software/pyseobnr.git
    (.venv) $ cd pyseobnr
    (.venv) $ pip install -U pip wheel setuptools numpy
    (.venv) $ pip install .

.. tip::

    If you are actively developing the code, consider using ``pip install -e .`` to
    avoid having to reinstall after every change to the code.

Installing optional dependencies to run checks
----------------------------------------------

If one wants to run sanity checks in ``pyseobnr/auxiliary/sanity_checks``, additional
dependencies must be installed. This can be done simply by running

.. code-block:: console

    (.venv) $ pip install ".[checks]"

The ``waveform_tools`` package located `here <https://bitbucket.org/sergei_ossokine/waveform_tools>`_ has
to be installed separately with the command

.. code-block:: console

    (.venv) $ pip install git+https://bitbucket.org/sergei_ossokine/waveform_tools


Installing optional dependencies to build documentation
-------------------------------------------------------

To build documentation, install the relevant dependencies with

.. code-block:: console

    (.venv) $ pip install ".[docs]"

Then the documentation can be built via

.. code-block:: console

    (.venv) $ cd docs
    (.venv) $ make html

Developments and tests
----------------------
A ``tox`` environment is provided for easier development on variations of python versions.

.. code-block:: console

    (.venv) $ pip install tox
    (.venv) $ tox -l            # lists the environments
    (.venv) $ CI_TEST_DYNAMIC_REGRESSIONS=1 \
              tox -e py311      # runs additional tests
    (.venv) $ tox -e py311      # runs the tests for python 3.11
    (.venv) $ tox -e docs       # builds the documentation




Platform specific instructions
------------------------------

pygsl and pygsl-lite
^^^^^^^^^^^^^^^^^^^^
``gsl`` is a library that gets installed on your operating system (with ``apt``,
``brew`` etc) and the python package ``pygsl`` links to it.

It may happen that a binary version of ``pygsl`` was built with another version of the
``gsl`` system library: in that case your installation will not work.

It is possible to force the installation of this library in order to use your system
installed version by either this command line:

.. code-block:: console

    (.venv) $ GSL_HOME=/path/to/your/GSL/HOME pip install --force-reinstall pygsl-lite

which becomes, if installed with ``brew``:

.. code-block:: console

    (.venv) $ GSL_HOME=$(brew --prefix gsl) pip install --force-reinstall pygsl-lite

or by this command line:

.. code-block:: console

    (.venv) $ pip install \
        --use-pep517 \
        --config-setting="--global-option=build_ext" \
        --config-setting="--build-option=-I$(brew --prefix gsl)/include/" \
        --config-setting="--build-option=-L$(brew --prefix gsl)/lib/" \
        --force-reinstall \
        --no-binary pygsl_lite \
        --no-cache-dir \
        pygsl_lite
