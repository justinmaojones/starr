from setuptools import find_packages, setup, Extension
from Cython.Distutils import build_ext

import numpy

requirements = (open("requirements.txt").read(),)
try:
    requirements_dev = (open("requirements_dev.txt").read(),)
    extras_require = {"dev": requirements_dev}
except IOError:
    extras_require = None

def get_long_description():
    # TODO: fix this hacky method of resolving relative links in pypi readme viewer 
    README = open("README.md").read()
    base_url = "https://github.com/justinmaojones/starr/blob/master/"
    relative_links = [
        "docs/badges/python.svg",
        "docs/badges/coverage.svg",
    ]
    for rlink in relative_links:
        abs_link = base_url + rlink 
        README = README.replace(rlink, abs_link)
    return README

setup(
    name="starr",
    version="0.1.0",
    description="Sumtrees for NumPy arrays",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/justinmaojones/starr",
    author="Justin Mao-Jones",
    author_email="justinmaojones@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    include_package_data=True,
    install_requires=requirements,
    extras_require=extras_require,
    packages=find_packages(include=["starr", "starr.*"]),
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension(
            "starr._cython",
            sources=["starr/src/_sumtree_array.pyx"],
            include_dirs=["."],
            language="c",
        ),
        Extension(
            "starr.experimental._cython",
            sources=[
                "starr/experimental/src/_sumtree_array.pyx",
                "starr/experimental/src/sumtree_array.cpp",
            ],
            include_dirs=[numpy.get_include(), "."],
            language="c++",
        ),
    ],
)
