from distutils.core import setup

setup(
    name='KoSpeech',
    version='1.0',
    install_requires=[
        'torch>=1.4.0',
        'python-Levenshtein',
        'librosa >= 0.7.0',
        'numpy',
        'pandas',
        'tqdm',
        'matplotlib',
        'astropy'
    ]
)
