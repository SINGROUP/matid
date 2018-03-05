from __future__ import absolute_import, division, print_function, unicode_literals
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(name="systax",
        version="0.2.0",
        url="https://gitlab.com/laurih/systax",
        description=(
            "Systax is a python package for the analysis and classification of "
            "atomistic systems."
        ),
        author='Lauri Himanen',
        author_email='lauri.himanen@aalto.fi',
        license="Apache 2.0",
        classifiers=[
            'Development Status :: 3 - Alpha',
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
        keywords='atoms structure material science crystal symmetry spglib',
        packages=find_packages(),
        install_requires=[
            "future",
            "numpy",
            "scipy",
            "ase",
            "spglib",
            "sklearn",
            "networkx",
            "chronic"
        ],
        python_requires='>=2.6, <4',
    )
