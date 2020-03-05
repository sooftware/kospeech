from distutils.core import setup
import setuptools

setup(
    name='Korean-Speech-Recognition',
    version='1.0',
    install_requires=[
        'torch>=1.2.0',
        'python-Levenshtein',
        'librosa >= 0.7.0',
        'numpy',
        'pandas',
        'tqdm'
    ]
)