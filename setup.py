from spring import __version__
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="spring",
    version=__version__,
    author="Mateusz Malenta",
    author_email="mateusz.malenta@gmail.com",
    description="Single pulse MeerTRAP post-processing pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="spring"),
    package_dir={"": "spring"},
    scripts=["bin/post_processing.py"]
)