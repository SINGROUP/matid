from __future__ import absolute_import, division, print_function, unicode_literals
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(name="matid",
        version="0.5.0",
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
        author_email='lauri.himanen@aalto.fi',
        license="Apache 2.0",
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Physics',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
        ],
        keywords='atoms structure materials science crystal symmetry',
        packages=find_packages(),
        install_requires=[
            "future",
            "numpy",
            "scipy",
            "ase",
            "spglib>=1.10.1",
            "sklearn",
            "networkx",
            "chronic"
        ],
        python_requires='>=2.6, <4',
    )
