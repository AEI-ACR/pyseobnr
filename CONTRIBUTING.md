# Contributing to `pyseobnr`

This is a short guide to contributing to `pyseobnr` package.  It follows very closely the one of the [`bilby` package](https://git.ligo.org/lscsoft/bilby).

Some familiarity with `python` and `git` is assumed.

- [Contributing to `pyseobnr`](#contributing-to-pyseobnr)
  - [Code style](#code-style)
  - [Automated code checking](#automated-code-checking)
  - [Merge requests](#merge-requests)

Everyone participating in any way in the development of `pyseobnr` (e.g on issues and merge requests) is expected to treat other people with respect, and follow the guidelines articulated in the [Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).

## Code style

For code contributions, please ensure your code fits with the`pyseobnr` style. This is based on a few python conventions and is generally maintained to ensure the code base remains consistent and readable. Here we list some things to keep in mind

1. We follow the [standard python PEP8](https://www.python.org/dev/peps/pep-0008/) conventions for style.

2. New classes/functions/methods should have a docstring and following the [google docstring guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings), for example (from the guide)

   ```python
   def fetch_smalltable_rows(
       table_handle: smalltable.Table,
       keys: Sequence[bytes | str],
       require_all_keys: bool = False,
   ) -> Mapping[bytes, tuple[str, ...]]:
       """Fetches rows from a Smalltable.

       Retrieves rows pertaining to the given keys from the Table instance
       represented by table_handle.  String keys will be UTF-8 encoded.

       Args:
         table_handle:
           An open smalltable.Table instance.
         keys:
           A sequence of strings representing the key of each table row to
           fetch.  String keys will be UTF-8 encoded.
         require_all_keys:
           If True only rows with values set for all keys will be returned.

       Returns:
         A dict mapping keys to the corresponding table row data
         fetched. Each row is represented as a tuple of strings. For
         example:

         {b'Serak': ('Rigel VII', 'Preparer'),
          b'Zim': ('Irk', 'Invader'),
          b'Lrrr': ('Omicron Persei 8', 'Emperor')}

         Returned keys are always bytes.  If a key from the keys argument is
         missing from the dictionary, then that row was not found in the
         table (and require_all_keys must have been False).

       Raises:
         IOError: An error occurred accessing the smalltable.
       """
   ```

3. Add  inline comments only when necessary. Ideally, the code should make it obvious what is going on, only in subtle cases use comments.  When adding docstrings and comments, add citations to appropriate papers, so it's easy to find where equations are coming from.

4. Name variables sensibly. Avoid using single-letter variables unless the context is absolutely obvious.

5. Don't copy-paste code. Instead, make a function and use it.

6. Add unit and regression tests. The C.I. is there to catch bugs and prevent regressions in the code.

## Automated code checking

In order to automate checking of the code quality, we use [pre-commit](https://pre-commit.com/). For more details, see the documentation,
here we will give a quick-start guide:

1. Install and configure:

   ```console
   pip install pre-commit  # install the pre-commit package
   cd pyseobnr
   pre-commit install
   ```

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


## Merge requests

All changes to the code base go through the [merge-request
workflow](https://docs.gitlab.com/ee/user/project/merge_requests/)