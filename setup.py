from setuptools import setup, find_packages

version = '0.1.1'

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='topwave',
    version=version, # put this in a version.py-file. # create funciton that reads the file.
    python_requires='>=3.11',
    url='https://github.com/nheinsdorf/topwave',
    author='Niclas Heinsdorf',
    author_email='nheinsdorf@gmail.com',
    description='Toolbox for Topology of Single-Particle Spectra',
    packages=find_packages(),
    install_requires=required_packages,
)
