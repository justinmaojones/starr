from setuptools import find_packages, setup, Extension
from Cython.Distutils import build_ext

import numpy

requirements = open('requirements.txt').read(),
requirements_dev = open('requirements_dev.txt').read(),
extras_require = {
    'dev': requirements_dev
}

setup(
	name='prefix_sum_tree',
	version='0.1',
	description='Prefix sum trees in Cython for Numpy',
	author='Justin Mao-Jones',
	author_email='justinmaojones@gmail.com',
	install_requires=requirements,
    extras_require=extras_require,
    packages=find_packages(include=['prefix_sum_tree','prefix_sum_tree.*']),
    cmdclass={'build_ext': build_ext},
    ext_modules = [ 
        Extension(
            'prefix_sum_tree._cython',
            sources=[
                'prefix_sum_tree/src/_prefix_sum_tree.pyx'
            ],
            include_dirs=['.'],
            language='c',
        ),
        Extension(
            'prefix_sum_tree.experimental._cython',
            sources=[
                'prefix_sum_tree/experimental/src/_prefix_sum_tree.pyx',
                'prefix_sum_tree/experimental/src/prefix_sum_tree.cpp'
            ],
            include_dirs=[numpy.get_include(),'.'],
            language='c++'
        ),
    ]
 )
