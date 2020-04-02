# ***Korean Speech Recognition***  
  
### Character-unit based End-to-End Korean Speech Recognition  
   
[<img src="https://github.com/gentaiscool/end2end-asr-pytorch/raw/master/img/pytorch-logo-dark.png" height=18>](https://pytorch.org/) <img src="https://img.shields.io/badge/License-Apache--2.0-yellow" height=20> [<img src="https://img.shields.io/badge/chat-on%20gitter-4fb99a" height=20>](https://gitter.im/Korean-Speech-Recognition/community)
  
### [**Documentation**](https://sooftware.github.io/Korean-Speech-Recognition/)   
  
## Intro

This is project for Korean Speech Recognition using LAS (Listen, Attend and Spell) models   
implemented in [PyTorch](http://pytorch.org).  
We appreciate any kind of feedback or contribution.
  
<img src="https://postfiles.pstatic.net/MjAyMDAyMjVfODIg/MDAxNTgyNjE5NzE3NjU5.51D-0F_nvBCZQ89XpgaycjPsX92z_lZK-vCQIHXfOmkg.kK0ILmnHM-LXMRxjTB5o1vJjKnhI4cw73me3LpvRkxUg.PNG.sooftware/LAS.png?type=w773" width=500> 
  
## Roadmap
  
Speech recognition is an interdisciplinary subfield of computational linguistics that develops methodologies and technologies that enables the recognition and translation of spoken language into text by computers.  
  
We mainly referred to following papers.  
  
 [「Listen, Attend and Spell」](https://arxiv.org/abs/1508.01211)  
 
[「State-of-the-art Speech Recognition with Sequence-to-Sequence Models」](https://arxiv.org/abs/1712.01769)
   
[「SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition」](https://arxiv.org/abs/1904.08779).   
  
if you want to study the feature of audio, we recommend this papers.  
  
[「Voice Recognition Using MFCC Algirithm」](https://ijirae.com/volumes/vol1/issue10/27.NVEC10086.pdf).  
  
Our project based on Seq2seq with Attention Architecture.  
Seq2seq is a fast evolving field with new techniques and architectures being published frequently.  
  
We use [AI Hub 1000h](http://www.aihub.or.kr/aidata/105) dataset which contains 1,000 hours korean voice data. and, our project is currently in progress.   
At present our top model has recorded an **80% CRR**, and we are working for a higher recognition rate.  
  
Also our model has recorded **91% CRR** in [Kadi-zeroth dataset](https://github.com/goodatlas/zeroth).  
  
###### ( **CRR** : Character Recognition Rate )  
  
We are constantly updating the progress of the project on the [Wiki page](https://github.com/sooftware/Korean-Speech-Recognition/wiki).  Please check this page.  
  
[More details](https://sh951011.github.io/Korean-Speech-Recognition/notes/More-details.html)

## Installation
This project recommends Python 3.7 or higher.   
We recommend creating a new virtual environment for this project (using virtual env or conda).  

### Prerequisites
  
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* PyTorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.
* Pandas: `pip install pandas` (Refer [here](https://github.com/pandas-dev/pandas) for problem installing Pandas)  
* librosa: `pip install librosa` (Refer [here](https://github.com/librosa/librosa) for problem installing librosa)
* tqdm: `pip install tqdm` (Refer [here](https://github.com/tqdm/tqdm) for problem installing tqdm)  
  
### Install from source
Currently we only support installation from source code using setuptools. Checkout the source code and run the   
following commands:  
```
pip install -r requirements.txt
```
```
python setup.py build
python setup.py install
```
  
## Get Started
### Preparation before Training

Refer [here](https://sh951011.github.io/Korean-Speech-Recognition/notes/Preparation.html) before Training.  
The above document is written in Korean.  
We will also write a document in English as soon as possible, so please wait a little bit.  
  
If you already have another dataset, please modify the data set path to [definition.py](https://github.com/sh951011/Korean-Speech-Recognition/blob/master/package/definition.py) as appropriate.  

### Train and Test
if you want to start training, you should run [train.py](https://github.com/sh951011/Korean-Speech-Recognition/blob/master/train.py).    
or after training, you want to start testing, you should run [test.py](https://github.com/sh951011/Korean-Speech-Recognition/blob/master/test.py).  
  
you can set up a configuration [config.py](https://github.com/sh951011/Korean-Speech-Recognition/blob/master/package/config.py).  
An explanation of configuration is [here](https://sh951011.github.io/Korean-Speech-Recognition/Hparams.html).  
  

## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sh951011/Korean-Speech-Recognition/issues) on Github.   
For live discussions, please go to our [gitter](https://gitter.im/Korean-Speech-Recognition/community) or Contacts sh951011@gmail.com please.
  
We appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  

### Code Style
We follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
    
### Reference   
[[1] 「Listen, Attend and Spell」  Paper](https://arxiv.org/abs/1508.01211)   
[[2] 「State-of-the-art Speech Recognition with Sequence-to-Sequence Models」   Paper](https://arxiv.org/abs/1712.01769)  
[[3] 「A Simple Data Augmentation Method for Automatic Speech Recognition」  Paper](https://arxiv.org/abs/1904.08779)     
[[4] 「Voice Recognition Using MFCC Algorithm」  Paper](https://ijirae.com/volumes/vol1/issue10/27.NVEC10086.pdf)        
[[5]    IBM pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)   
[[6]    A.I Hub Korean Voice Dataset](http://www.aihub.or.kr/aidata/105)   
  
### License
```
Copyright (c) 2020 Kai.Lib

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
