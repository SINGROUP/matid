import sys
from setuptools import setup, find_packages

# Check python version
if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")

if __name__ == "__main__":
    setup(name="matid",
        version="0.6.2",
        url="https://singroup.github.io/matid/",
        description=(
            "MatID is a python package for identifying and analyzing atomistic "
            "systems based on their structure."
        ),
        long_description=(
            "MatID is a python package for identifying and analyzing atomistic "
            "systems based on their structure."
        ),
        author='Lauri Himanen',
        license="Apache License 2.0",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Physics",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3 :: Only",
        ],
        keywords='atoms structure materials science crystal symmetry',
        packages=find_packages(),
        install_requires=[
            "future",
            "numpy",
            "scipy",
            "ase",
            "spglib>=1.15.0",
            "sklearn",
            "networkx>=2.4",
            "chronic"
        ],
        python_requires=">=3.6",
    )
