from distutils.core import setup

setup(
    name='KoSpeech',
    version='latest',
    description='Open-Source Toolkit for End-to-End Korean Speech Recognition',
    author='Soohwan Kim',
    author_email='kaki.brain@kakaobrain.com',
    url='https://github.com/sooftware/KoSpeech',
    install_requires=[
        'torch>=1.4.0',
        'python-Levenshtein',
        'librosa >= 0.7.0',
        'numpy',
        'pandas',
        'tqdm',
        'matplotlib',
        'astropy',
        'sentencepiece'
    ],
    keywords=['asr', 'speech_recognition', 'korean'],
    python_requires='>=3'
)
