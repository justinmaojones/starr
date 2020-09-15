from distutils.core import setup, Extension
from Cython.Distutils import build_ext

import numpy

setup(
	name='prefix_sum_tree',
	version='0.1',
	description='Prefix sum trees in Cython',
	author='Justin Mao-Jones',
	author_email='justinmaojones@gmail.com',
	install_requires=open('requirements.txt').read(),
	packages=['prefix_sum_tree'],
    cmdclass={'build_ext': build_ext},
    ext_modules = [ 
        Extension(
            'prefix_sum_tree._cython',
            sources=[
                'prefix_sum_tree/src/_prefix_sum_tree.pyx'
            ],
            include_dirs=['.'],
        ),
        Extension(
            'prefix_sum_tree.experimental._cython',
            sources=[
                'prefix_sum_tree/experimental/src/_prefix_sum_tree.pyx',
                'prefix_sum_tree/experimental/src/prefix_sum_tree.cpp'
            ],
            include_dirs=[numpy.get_include(),'.'],
            language='C++'
        ),
    ]
 )
