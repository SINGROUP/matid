from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(name='systax',
        version='0.1',
        description='Python package for detecting components in atomistic surface systems.',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'ase',
            'spglib',
            'sklearn',
        ],
    )
