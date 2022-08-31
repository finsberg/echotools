#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System imports
from setuptools import setup, Command

from setuptools import setup
import glob

# Version number
major = 0
minor = 1

scripts = glob.glob("bin/*")

with open("README.md", "r") as fh:
    long_description = fh.read()

    
setup(name="echotools",
      version="{0}.{1}".format(major, minor),
      description="""
      A toolbox for analyzing echocardiographic data
      """,
      author="Henrik Finsberg",
      author_email="henriknf@simula.no",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['echotools', 'echotools.gmsh'],
      install_requires=['h5py',
                        'scipy',
                        'numpy'],
      scripts=scripts,
      )
