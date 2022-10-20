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
              extra_compile_args=['-O3']
     ),
    Extension("pyseobnr.eob.hamiltonian.Hamiltonian_C", ["pyseobnr/eob/hamiltonian/Hamiltonian_C.pyx"],
              include_dirs=[np.get_include(),"pyseobnr/eob/utils"],
    ),
    Extension("pyseobnr.eob.dynamics.rhs_aligned", ["pyseobnr/eob/dynamics/rhs_aligned.pyx"],
              include_dirs=[np.get_include(),"pyseobnr/eob/utils"],
    ),
    Extension("pyseobnr.eob.dynamics.postadiabatic_C", ["pyseobnr/eob/dynamics/postadiabatic_C.pyx"],
              include_dirs=[np.get_include(),"pyseobnr/eob/utils","pyseobnr/eob/hamiltonian"],
    ),
    Extension("pyseobnr.auxiliary.interpolate._ppoly", ["pyseobnr/auxiliary/interpolate/_ppoly.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args = ["-flto", "-O3", "-ftree-vectorize"]),

]



setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
