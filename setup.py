from __future__ import absolute_import, division, print_function, unicode_literals
from setuptools import setup, find_packages

with open("README.md", "r") as fin:
    long_description = fin.read()

if __name__ == "__main__":
    setup(name="systax",
        version="0.1.0",
        description=(
            "Systax is a python package for the analysis and classification of "
            "atomistic systems."
        ),
        long_description=long_description,
        author='Lauri Himanen',
        author_email='lauri.himanen@aalto.fi',
        license="LGPL3",
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Physics',
            'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
        ],
        keywords='atoms structure material science crystal symmetry spglib',
        packages=find_packages(),
        install_requires=[
            "future",
            "enum",
            "numpy",
            "scipy",
            "ase",
            "spglib",
            "sklearn",
            "networkx",
        ],
        python_requires='>=2.6, <4',
    )
