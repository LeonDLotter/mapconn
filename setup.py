from setuptools import setup, find_packages

setup(
    name='mapconn',
    version='0.0.1-dev',
    description='A package for reference map-dependent connectivity analyses.',
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
        'nispace @ git+https://github.com/leondlotter/nispace',
        'brainsmash',
        'brainspace',
        'scikit-learn',
        'joblib',
        'tqdm',        
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)