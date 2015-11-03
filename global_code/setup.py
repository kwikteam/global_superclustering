import os
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import platform

extensions = cythonize(["supercluster.pyx","e_step_cy.pyx","m_step_cy.pyx"])

for ext in extensions:
    if 'e_step_cy' in ext.name:
        if os.name=='nt': # Windows
            ext.extra_compile_args = ['/openmp']
        elif platform.system()=='Darwin' and not with_openmp: # Mac
            pass
        else:
            ext.extra_compile_args =['-fopenmp']
            ext.extra_link_args = ['-fopenmp']

setup_kwds = dict(ext_modules=extensions,include_dirs=[numpy.get_include()])
                  
setup(**setup_kwds)



#setup(
 #   ,
  #  include_dirs=[numpy.get_include()]
    
   # ext_modules = cythonize("supercluster.pyx")
#)
