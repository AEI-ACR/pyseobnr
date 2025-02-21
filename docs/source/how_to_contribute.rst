Contributing to ``pyseobnr``
============================

You are using ``pyseobnr``, but something is missing or not working well, or
the documentation needs improvement?

We are more than happy to accept contributions from our users' community,
and this short guide introduces you to contributing to ``pyseobnr`` package.
It follows closely the excellent one from the ``bilby`` package
`contribution guide <https://github.com/bilby-dev/bilby/blob/main/CONTRIBUTING.md>`_.

Some familiarity with ``python`` and ``git`` is assumed.

Before contributing
-------------------

``pyseobnr`` is employed for scientific work, and the code changes are reviewed very
carefully. In that sense, we aim at high quality standards concerning code changes, which
involves:

* following the coding guidelines
* meeting all the quality metrics, especially test coverage and documentation

Besides the developments, everyone participating in any way in the development of
``pyseobnr`` (e.g on issues and merge requests) is expected to treat other people with
respect, and follow the guidelines articulated in the
`Python Software Foundation Code of Conduct`_.

.. _Python Software Foundation Code of Conduct: https://www.python.org/psf/codeofconduct/


Contribution procedure
^^^^^^^^^^^^^^^^^^^^^^
The procedure for contributing is as follow:

* get in touch with the ``pyseobnr`` team, especially if you are not an LVK member. The main code
  of ``pyseobnr`` lives in LVK `GitLab instance`_
  and, unless you are part of the LVK consortium, you will not be able to report bugs or request features,
  nor open PRs.

* a bug report is more than welcome: please provide a minimal code snippet to reproduce it,
  possibly indicating the environment
  under which this bug is happening (pinned versions of packages including ``pyseobnr``,
  operating system, architecture).

* a feature request would need to be accompanied with the use case and will likely open a discussion
  thread. Please make that discussion lively.

* finally, if you are making a merge request (PR) or providing code changes by means of patches,
  please make sure you follow the development guidelines. See below for details.

.. note::

    We have a `read only` `GitHub mirror`_ that can be used to create patches. However, as it is read-only
    we are not accepting PRs directly there. See
    `this guide <https://git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project#_public_project>`_
    on how to send patches by email.

.. _GitLab instance: https://git.ligo.org/waveforms/software/pyseobnr/
.. _GitHub mirror: https://github.com/AEI-ACR/pyseobnr

Merge requests
^^^^^^^^^^^^^^
Whenever possible (if you have access to the `GitLab instance`_, are part of the LVK consortium, etc), code contributions
should go through a `merge request <https://docs.gitlab.com/ee/user/project/merge_requests/>`_.
Opening a merge request can be done at different stage of your contribution:

* early development to suggest a design, idea, direction of development
* mature code ready to be merged

In all cases, the merge request will be the place to validate your contribution, and will run the tests and generate the documentation
for various platforms: it is mandatory that the code that we merge is fully `green`.

Development guidelines
----------------------

This is based on a few python conventions and is generally maintained to ensure the
code base remains consistent and readable. Here we list some things to keep in mind

Code style
^^^^^^^^^^

For code contributions, please ensure your code follows the ``pyseobnr`` style:

* we follow the `PEP8`_ conventions for style, which covers most of the aspects of the coding style,
* name of the variables should be clear enough and not mislead their content. Variable reuse for different
  contexts is preferably avoided,
* comments are here to help reading and understanding the code and its intent, and they complement names
  and documentation,

* functions and classes scope should be limited: we discourage the writing of very long functions that do many different things

* constructions for ensuring correctness are encouraged:

  * we make extensive use of ``python``'s ``typing`` library, which makes the developments much easier, faster,
    and less error-prone as linting tools and IDEs can spot mistakes while typing,
  * use of ``TypedDict``, ``dataclass``, ``NamedTuple`` is preferred over ``tuple``, especially on returned objects.
  * ``@override``, runtime checked inheritance,
  * use of ``assert`` for checking correctness on the use of the function parameters,
    if the expression being evaluated in the assertion does not involve a noticeable overhead,
  * functions should not change their return type (including number of tuple elements) depending on the parameters

* we delegate code formatting to ``black``, ``isort`` and ``cython-lint``, and ``flake8`` for linting.
  Those tools as well as additional checks are automatically run on the code you want to commit
  if the ``pre-commit`` tool is installed (see below),

* we try to be as minimalistic as possible with respect to the dependencies, and separate the packages needed for
  actually building, testing or generating the documentation,

* we use ``tox`` for testing on various environment, and make extensive use of CIs.

.. _PEP8: https://www.python.org/dev/peps/pep-0008/

Automated code checking
***********************

In order to automate checking of the code quality, we use `pre-commit`_. For more details, see the documentation,
here we will give a quick-start guide:

.. _pre-commit: https://pre-commit.com/

1. Install and configure:

   .. code-block :: console

      # install the pre-commit package
      pip install pre-commit
      cd pyseobnr
      pre-commit install


2. Now, when you run `$ git commit`, there will be a pre-commit check.
   This is going to search for issues in your code: spelling, formatting, etc.
   In some cases, it will automatically fix the code, in other cases, it will
   print a warning. If it automatically fixed the code, you'll need to add the
   changes to the index (`$ git add FILE.py`) and run `$ git commit` again. If
   it didn't automatically fix the code, but still failed, it will have printed
   a message as to why the commit failed. Read the message, fix the issues,
   then recommit.

3. The pre-commit checks are done to avoid pushing and then failing. But, you
   can skip them by running `$ git commit --no-verify`, but note that the C.I.
   still does the check so you won't be able to merge until the issues are
   resolved.

Besides the automated code checking with ``pre-commit``, you can always run the linting task in ``tox``:

.. code-block:: console

  tox -e linting

Testing
^^^^^^^
There are many strategies for testing, and unit test should accompany code changes, new functions or classes,
bug fixes (test that reproduces the bug), etc. On the other hand, the testing code needs also to be maintained.

* test is code, the same coding guidelines as for the rest of the package apply
* we use ``pytest`` for writing tests, but we also access tests written using the `python unit testing framework`_
* we make extensive use of the mocking_ facility of the python unit testing framework: this allows us to test the logic of
  input/outputs, flow of the program, error handling, etc... without actually always executing CPU intensive code
* we organize the tests in files that are scoped with more or less one purpose
* we can sometimes check for exact numerical values, but it is important to test on various platforms to get a sense
  of the tolerated numerical deviations from those reference values. Some arrays can be stored in a format compatible
  with ``pandas``
* ``plugins`` are also covered with our tests, which involves the installation of the system using the plugin
* some tests are autogenerated by external tools: if this is the case, they should of course not be modified manually.

.. code-block:: console

  tox -e py311

.. _python unit testing framework: https://docs.python.org/3/library/unittest.html
.. _mocking: https://docs.python.org/3/library/unittest.mock.html

Documentation
^^^^^^^^^^^^^
We place a particular care to the documentation of ``pyseobnr``. The science behind is particularly complicated and precise,
the package documentation deserves precision as well.

We use Sphinx as our primary tool for documentation, which makes it easy to extract documentation
from ``python`` source code directly through ``docstrings``. Please make sure:

* the functions and classes are properly documented. In particular parameters and returned objects, as well as
  settings or options are clearly indicated.

  .. note::

    If the parameters are properly typed in the function or class, there is no
    need to repeat their type in the documentation.

* we like math and references, which makes the navigation and reading more appealing. Do not hesitate
  to use the facilities Sphinx provide for `referencing citations <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#citations>`_
  (see the ``citations.rst`` file)
  and writing `mathematical expressions <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-math>`_ .

* documentation contains jupyter notebooks that are automatically rendered while building the documentation.
  The notebooks are stored in a ``jupyter-lab`` compatible format called ``jupytext``, where only the code
  of the cells (without input) is actually stored.

* the documentation generation supports the `numpy and Google style <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_ markup:
  be consistent when you choose one or the other (vanilla ``rst`` formatting used by Sphinx).

* make sure the rendered documentation does what you want, see below to generate the documentation locally

You can run:

.. code-block:: console

  tox -e docs

to generate the documentation. This command will install all the dependencies for generating the documentation. See
:doc:`installation` for further instructions on the requirements for building the documentation.

.. seealso::

    The `google docstring guide <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_ can give
    nice hints on how to document effectively.

git branches and commits
^^^^^^^^^^^^^^^^^^^^^^^^
We welcome contributions as few commits as possible, as well as sound commit messages.
Short branches and commit history help inspecting the changes retrospectively.
