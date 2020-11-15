from setuptools import find_packages, setup, Extension
from Cython.Distutils import build_ext

import numpy

requirements = (open("requirements.txt").read(),)
try:
    requirements_dev = (open("requirements_dev.txt").read(),)
    extras_require = {"dev": requirements_dev}
except IOError:
    extras_require = None

setup(
    name="sumtree_array",
    version="0.1",
    description="Prefix sum trees in Cython for Numpy",
    author="Justin Mao-Jones",
    author_email="justinmaojones@gmail.com",
    install_requires=requirements,
    extras_require=extras_require,
    packages=find_packages(include=["sumtree_array", "sumtree_array.*"]),
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension(
            "sumtree_array._cython",
            sources=["sumtree_array/src/_sumtree_array.pyx"],
            include_dirs=["."],
            language="c",
        ),
        Extension(
            "sumtree_array.experimental._cython",
            sources=[
                "sumtree_array/experimental/src/_sumtree_array.pyx",
                "sumtree_array/experimental/src/sumtree_array.cpp",
            ],
            include_dirs=[numpy.get_include(), "."],
            language="c++",
        ),
    ],
)
