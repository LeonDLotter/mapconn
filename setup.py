from pathlib import Path

from setuptools import find_packages, setup


def read_version() -> str:
    version_file = Path(__file__).parent / "mapconn" / "__init__.py"
    for line in version_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"")
    raise RuntimeError("Unable to find __version__ in mapconn/__init__.py")

setup(
    name='mapconn',
    version=read_version(),
    description='A package for reference map-dependent connectivity analyses (NEOFC).',
    author='Leon D. Lotter',
    author_email='leondlotter@gmail.com',
    url='https://github.com/leondlotter/mapconn',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'xarray',
        'nibabel',
        'nilearn',
        'pingouin',
        'nispace @ git+https://github.com/leondlotter/nispace@5470ca5cb68d8cce3100a014d1a4adfb32f92306',
        'scikit-learn',
        'joblib',
        'tqdm',        
        'scipy',
        'spatiotemporal>=1.0.1',
        'setuptools',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)