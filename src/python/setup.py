## example:
## http://stackoverflow.com/questions/16792792/project-organization-with-cython-and-c

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        'compresso',
        include_dirs=[np.get_include(), '../c++/'],
        sources=['compresso.pyx'],
        extra_compile_args=['-O4', '-std=c++11', '-C'],
        language='c++'
    )
]

setup(
    ext_modules=cythonize(extensions)
)
