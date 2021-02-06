import os
from setuptools import setup, find_packages
from glob import glob

scripts = glob('scripts/*')
version = '0.1.0'

setup(name='ssp',
      version=version,
      description='Slice selection profile estimation in 2D MRI',
      author='Shuo Han',
      url='https://github.com/shuohan/ssp',
      author_email='shan50@jhu.edu',
      scripts=scripts,
      license='MIT',
      packages=['ssp']
      )
