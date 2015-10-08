from distutils.core import setup
from Cython.Build import cythonize

setup(
    #ext_modules = cythonize(["supercluster.pyx","e_step_cy.pyx"])
    ext_modules = cythonize("supercluster.pyx")
)
