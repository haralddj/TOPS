from setuptools import setup, find_packages

setup(
    name='tops_project',                # Name of your project
    version='0.1.0',                    # Version number
    packages=find_packages(where='src'), # Automatically find all packages in 'src'
    package_dir={'': 'src'},             # Tell setuptools where to look for the packages
)