import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# see https://cython.readthedocs.io/en/latest/src/userguide/migrating_to_cy30.html#numpy-c-api
# for the NPY_NO_DEPRECATED_API macro definition.

_numpy_no_deprecated_api = ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")

extensions = [
    Extension(
        "pyseobnr.eob.utils.containers",
        ["pyseobnr/eob/utils/containers.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[_numpy_no_deprecated_api],
    ),
    Extension(
        "pyseobnr.eob.waveform.waveform",
        ["pyseobnr/eob/waveform/waveform.pyx"],
        include_dirs=[np.get_include(), "pyseobnr/eob/utils"],
        define_macros=[_numpy_no_deprecated_api],
    ),
    Extension(
        "pyseobnr.eob.hamiltonian.Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C",
        ["pyseobnr/eob/hamiltonian/Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C.pyx"],
        include_dirs=[np.get_include(), "pyseobnr/eob/utils"],
        define_macros=[_numpy_no_deprecated_api],
    ),
    Extension(
        "pyseobnr.eob.hamiltonian.Hamiltonian_C",
        ["pyseobnr/eob/hamiltonian/Hamiltonian_C.pyx"],
        include_dirs=[np.get_include(), "pyseobnr/eob/utils"],
        define_macros=[_numpy_no_deprecated_api],
    ),
    Extension(
        "pyseobnr.eob.dynamics.rhs_aligned",
        ["pyseobnr/eob/dynamics/rhs_aligned.pyx"],
        include_dirs=[np.get_include(), "pyseobnr/eob/utils"],
        define_macros=[_numpy_no_deprecated_api],
    ),
    Extension(
        "pyseobnr.eob.dynamics.rhs_precessing",
        ["pyseobnr/eob/dynamics/rhs_precessing.pyx"],
        include_dirs=[np.get_include(), "pyseobnr/eob/utils"],
        define_macros=[_numpy_no_deprecated_api],
    ),
    Extension(
        "pyseobnr.eob.dynamics.postadiabatic_C",
        ["pyseobnr/eob/dynamics/postadiabatic_C.pyx"],
        include_dirs=[
            np.get_include(),
            "pyseobnr/eob/utils",
            "pyseobnr/eob/hamiltonian",
        ],
        define_macros=[_numpy_no_deprecated_api],
    ),
    Extension(
        "pyseobnr.eob.dynamics.postadiabatic_C_prec",
        ["pyseobnr/eob/dynamics/postadiabatic_C_prec.pyx"],
        include_dirs=[
            np.get_include(),
            "pyseobnr/eob/utils",
            "pyseobnr/eob/hamiltonian",
        ],
        define_macros=[_numpy_no_deprecated_api],
    ),
    Extension(
        "pyseobnr.eob.dynamics.postadiabatic_C_fast",
        ["pyseobnr/eob/dynamics/postadiabatic_C_fast.pyx"],
        include_dirs=[
            np.get_include(),
            "pyseobnr/eob/utils",
            "pyseobnr/eob/hamiltonian",
        ],
        define_macros=[_numpy_no_deprecated_api],
    ),
    Extension(
        "pyseobnr.eob.hamiltonian.Hamiltonian_v5PHM_C",
        ["pyseobnr/eob/hamiltonian/Hamiltonian_v5PHM_C.pyx"],
        include_dirs=[
            np.get_include(),
            "pyseobnr/eob/utils",
            "pyseobnr/eob/hamiltonian",
        ],
        define_macros=[_numpy_no_deprecated_api],
    ),
    Extension(
        "pyseobnr.eob.hamiltonian.Ham_AvgS2precess_simple_cython_PA_AD",
        ["pyseobnr/eob/hamiltonian/Ham_AvgS2precess_simple_cython_PA_AD.pyx"],
        include_dirs=[
            np.get_include(),
            "pyseobnr/eob/utils",
            "pyseobnr/eob/hamiltonian",
        ],
        define_macros=[_numpy_no_deprecated_api],
    ),
]


setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
