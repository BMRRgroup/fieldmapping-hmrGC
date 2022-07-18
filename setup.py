import setuptools
from setuptools import setup

__version__ = "3.1.1"
install_deps = [
    "numba",
    "pymaxflow",
    "opencv-python",
    "scipy"
]

setup(
    name='hmrGC',
    description='water-fat(-silicone) seperation using hierarchical multi-resolution graph-cuts',
    author='Jonathan Stelter',
    author_email='jonathan.stelter@tum.de',
    packages=setuptools.find_packages(),
    version = __version__,
    install_requires=install_deps,
    package_data={'': ['*.json']},
)
