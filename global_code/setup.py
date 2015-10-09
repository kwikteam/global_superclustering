from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["supercluster.pyx","e_step_cy.pyx"]),
    include_dirs=[numpy.get_include()]
   # ext_modules = cythonize("supercluster.pyx")
)
