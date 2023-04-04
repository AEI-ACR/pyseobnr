from setuptools import setup,Extension
from Cython.Build import cythonize
from setuptools import find_packages
import numpy as np



extensions = [
    Extension("pyseobnr.eob.utils.containers", ["pyseobnr/eob/utils/containers.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension("pyseobnr.eob.waveform.waveform", ["pyseobnr/eob/waveform/waveform.pyx"],
              include_dirs=[np.get_include(),"pyseobnr/eob/utils"],
              extra_compile_args=['-O3'],
    ),
    Extension("pyseobnr.eob.hamiltonian.Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C", ["pyseobnr/eob/hamiltonian/Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C.pyx"],
                include_dirs=[np.get_include(),"pyseobnr/eob/utils"],
              extra_compile_args=['-O3'], define_macros=[('CYTHON_TRACE', '1')]
     ),
    Extension("pyseobnr.eob.hamiltonian.Hamiltonian_C", ["pyseobnr/eob/hamiltonian/Hamiltonian_C.pyx"],
              include_dirs=[np.get_include(),"pyseobnr/eob/utils"],
              extra_compile_args=['-O3'],
    ),
    Extension("pyseobnr.eob.dynamics.rhs_aligned", ["pyseobnr/eob/dynamics/rhs_aligned.pyx"],
              include_dirs=[np.get_include(),"pyseobnr/eob/utils"],
              extra_compile_args=['-O3'],
              define_macros=[('CYTHON_TRACE', '1')]
    ),
    Extension("pyseobnr.eob.dynamics.rhs_precessing", ["pyseobnr/eob/dynamics/rhs_precessing.pyx"],
              include_dirs=[np.get_include(),"pyseobnr/eob/utils"],
              extra_compile_args=['-O3']
    ),
    Extension("pyseobnr.eob.dynamics.postadiabatic_C", ["pyseobnr/eob/dynamics/postadiabatic_C.pyx"],
              include_dirs=[np.get_include(),"pyseobnr/eob/utils","pyseobnr/eob/hamiltonian"],
              extra_compile_args=['-O3']
    ),
    Extension("pyseobnr.eob.dynamics.postadiabatic_C_prec", ["pyseobnr/eob/dynamics/postadiabatic_C_prec.pyx"],
              include_dirs=[np.get_include(),"pyseobnr/eob/utils","pyseobnr/eob/hamiltonian"],
              extra_compile_args=['-O3']
    ),
    Extension("pyseobnr.eob.dynamics.postadiabatic_C_fast", ["pyseobnr/eob/dynamics/postadiabatic_C_fast.pyx"],
              include_dirs=[np.get_include(),"pyseobnr/eob/utils","pyseobnr/eob/hamiltonian"],
              extra_compile_args=['-O3']
              ),
    Extension("pyseobnr.eob.hamiltonian.Hamiltonian_v5PHM_C", ["pyseobnr/eob/hamiltonian/Hamiltonian_v5PHM_C.pyx"],
             include_dirs=[np.get_include(),"pyseobnr/eob/utils","pyseobnr/eob/hamiltonian"],
             extra_compile_args=['-O3']
    ),
    Extension("pyseobnr.eob.hamiltonian.Ham_AvgS2precess_simple_cython_PA_AD", ["pyseobnr/eob/hamiltonian/Ham_AvgS2precess_simple_cython_PA_AD.pyx"],
             include_dirs=[np.get_include(),"pyseobnr/eob/utils","pyseobnr/eob/hamiltonian"],
             extra_compile_args=['-O3'],
    ),
]



setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
