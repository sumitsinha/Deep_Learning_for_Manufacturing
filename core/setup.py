from setuptools import setup, find_packages
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))


if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.md')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

version = {}
with open(os.path.join(_here, 'mdn', 'version.py')) as f:
    exec(f.read(), version)

with open('requirements.txt') as f:
    required = f.read().splitlines()
#requirements = ["ipython>=6"]

setup(
    name="deep-learning-for-manufacturing",
    version=version['__version__'],
    author="Sumit Sinha",
    author_email="sumit.sinha.1@warwick.ac.uk",
    description="Deep Learning applications for manufacturing",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/sumitsinha/Deep_Learning_for_Manufacturing",
    packages=find_packages(),
    license='MIT',
    packages=['dlmf'],
    include_package_data=True,
    install_requires=required,
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
    ],
)