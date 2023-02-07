from setuptools import setup,find_packages
from os.path import splitext
from os.path import basename
from glob import glob


with open('README.md') as f:
    readme = f.read()

setup(name='CARAMEL',
      version='0.0.1',
      description='Comprehensive vAlidation fRamework for Atmospheric ModELs',
      long_description=readme,
      long_description_content_type='text/markdown',
      author='Caramels',
      author_email='Caramel.git@gmail.com',
      license='MIT',
      packages=['caramel'],
      install_requires=[
          'numpy','matplotlib','scipy','netCDF4','scikit-gstat'
          ],
      zip_safe=False)
