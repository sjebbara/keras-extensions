from setuptools import setup
from setuptools import find_packages


setup(name='keras-extensions',
      version='0.1.0',
      description='Useful extensions and plugins for the Keras Deep Learning Library',
      author='Soufian Jebbara',
      author_email='s.jebbara@gmail.com',
      url='https://github.com/sjebbara/keras-extensions',
      download_url='https://github.com/sjebbara/keras-extensions/tarball/0.3.1',
      license='MIT',
      install_requires=['keras'],
      packages=find_packages())
