from spring import __version__ 
from setuptools import setup

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
    package_dir={"": "spring"},
    packages=["spcandidate", "spmodule", "spmodule.sputility", "spmodule.spcompute", "sppipeline", "spqueue"],
    scripts=["bin/post_processing.py"]
)
