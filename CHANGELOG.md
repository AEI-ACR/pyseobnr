All notable changes will be noted in this file, whenever a tagged
version is made

## [0.2.x]

* `pyCBC` plugin infrastructure: now `pyseobnr` approximants `SEOBNRv5HM` and `SEOBNRv5PHM` can be used
  directly from `pyCBC`.

## [0.2.12] 02/10/2024

Improved estimation of the reference time [!76](https://git.ligo.org/waveforms/software/pyseobnr/-/merge_requests/76).

## [0.2.11] 05/08/2024
Fix CI and packaging issues.

## [0.2.10] 27/05/2024
Fix retrograde case [!22](https://git.ligo.org/waveforms/software/pyseobnr/-/issues/22).

## [0.2.9] 09/05/2024

* Fix a performance issue with post-adiabatic [!67](https://git.ligo.org/waveforms/software/pyseobnr/-/merge_requests/67)
* Dynamics interpolation performance improvement [!65](https://git.ligo.org/waveforms/software/pyseobnr/-/merge_requests/65)
* Documentation enhancement
* import of pyseobnr improvements/speedup [!65](https://git.ligo.org/waveforms/software/pyseobnr/-/merge_requests/65)
* Refactoring, code sanity
* Change in the behaviour of f_re [!60](https://git.ligo.org/waveforms/software/pyseobnr/-/merge_requests/60)
* models are now stored as class property [!61](https://git.ligo.org/waveforms/software/pyseobnr/-/merge_requests/61)

## [0.2.8] 19/02/2024

Bugfix release. Minor update for conda builds.

## [0.2.7] 23/11/2023

Maintenance release: code and documentation improvements, better test coverage.
Removal of JAX dependency
Now accepting "ModeArray" in the settings to remain compatible with Bilby: the previous "mode_array" is de-facto deprecated.

## [0.2.6] 16/08/2023

Bugfix release. Fixes another compatibility issue with Cython 3.0.0. Increase total mass upper limit to 1e12.

## [0.2.5] 01/08/2023

Bugfix release. Fixes a compatibility issue with Cython 3.0.0.

## [0.2.4] 15/05/2023

Bugfix release. Fixes a rare issue with quaternion overflow, which occurred after attachment.

## [0.2.3] 10/05/2023

Bugfix release. Fixes a minor issue where the correct message was not included in ValueError,
which was needed for `bilby` error handling

## [0.2.2] 02/05/2023

Bugfix release. Fixes a bug in stopping conditions of PN integration at very low frequency.

## [0.2.0] 06/04/2023

Initial release for conda. Several housekeeping improvements, add references to papers.

## [0.1.0] 31/03/2023

Initial release of the aligned-spin model `SEOBNRv5HM` and the
precessing model `SEOBNRv5PHM`.
