from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
            'compresso',
            include_dirs=[np.get_include()],
            sources=['compresso.pyx', 'cpp-compresso.cpp'],
            extra_compile_args=['-O4', '-std=c++0x'],
            language='c++'
        )
]

setup(
    ext_modules = cythonize(extensions)
)
