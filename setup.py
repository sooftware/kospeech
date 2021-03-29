# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from distutils.core import setup

setup(
    name='KoSpeech',
    version='latest',
    description='Open-Source Toolkit for End-to-End Korean Speech Recognition',
    author='Soohwan Kim',
    author_email='kaki.ai@tunib.ai',
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
        'sentencepiece',
        'hydra-core',
    ],
    keywords=['asr', 'speech_recognition', 'korean'],
    python_requires='>=3',
)
