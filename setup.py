"""Install script for setuptools."""

from distutils import cmd
import os
import urllib.request

from setuptools import find_packages
from setuptools import setup
from setuptools.command import install

SIMCLR_DIR = 'simclr'
DATA_UTILS_URL = 'https://raw.githubusercontent.com/google-research/simclr/master/data_util.py'


class DownloadSimCLRAugmentationCommand(cmd.Command):
  """Downloads SimCLR data_utils.py as it's not built into an egg."""
  description = __doc__
  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    build_cmd = self.get_finalized_command('build')
    dist_root = os.path.realpath(build_cmd.build_lib)
    output_dir = os.path.join(dist_root, SIMCLR_DIR)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'data_util.py')
    downloader = urllib.request.URLopener()
    downloader.retrieve(DATA_UTILS_URL, output_path)


class InstallCommand(install.install):

  def run(self):
    self.run_command('simclr_download')
    install.install.run(self)


setup(
    name='meta_dataset',
    version='0.2.0',
    description='A benchmark for few-shot classification.',
    author='Google LLC',
    license='Apache License, Version 2.0',
    python_requires='>=2.7, <3.10',
    packages=find_packages(),
    install_requires=[
        'absl-py>=0.7.0',
        'gin-config>=0.1.2',
        'numpy>=1.13.3',
        'scipy>=1.0.0',
        'setuptools',
        'six>=1.12.0',
        # Note that this will install tf 2.0, even though this is a tf 1.0
        # project. This is necessary because we rely on augmentation from
        # tf-models-official that wasn't added until after tf 2.0 was released.
        'tensorflow-gpu',
        'sklearn',
        'tensorflow_probability<=0.7',
        'tf-models-official',
    ],
    cmdclass={
        'simclr_download': DownloadSimCLRAugmentationCommand,
        'install': InstallCommand,
    })
