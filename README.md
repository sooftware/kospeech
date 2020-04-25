# **End-to-End Korean Speech Recognition**  
  
### Character-unit based End-to-End Korean Speech Recognition  
   
[<img src="https://github.com/gentaiscool/end2end-asr-pytorch/raw/master/img/pytorch-logo-dark.png" height=18>](https://pytorch.org/) <img src="https://img.shields.io/badge/License-Apache--2.0-yellow" height=20> [<img src="https://img.shields.io/badge/chat-on%20gitter-4fb99a" height=20>](https://gitter.im/Korean-Speech-Recognition/community)
  
### [**Documentation**](https://sooftware.github.io/End-to-End-Korean-Speech-Recognition/)   
  
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
Our model architeuture is as follows.
  
```python
ListenAttendSpell(
  (listener): Listener(
    (conv): Sequential(
      (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): Hardtanh(min_val=0, max_val=20, inplace=True)
      (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): Hardtanh(min_val=0, max_val=20, inplace=True)
      (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): Hardtanh(min_val=0, max_val=20, inplace=True)
      (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): Hardtanh(min_val=0, max_val=20, inplace=True)
      (12): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (13): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (14): Hardtanh(min_val=0, max_val=20, inplace=True)
      (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (rnn): GRU(5120, 256, num_layers=5, batch_first=True, dropout=0.3, bidirectional=True)
  )
  (speller): Speller(
    (rnn): GRU(512, 512, num_layers=3, batch_first=True, dropout=0.3)
    (embedding): Embedding(2040, 512)
    (input_dropout): Dropout(p=0.3, inplace=False)
    (fc): Linear(in_features=512, out_features=2040, bias=True)
    (attention): MultiHeadAttention(
      (W): Linear(in_features=512, out_features=512, bias=True)
      (V): Linear(in_features=512, out_features=512, bias=True)
      (fc): Linear(in_features=1024, out_features=512, bias=True)
    )
  )
)
```
  
We use [AI Hub 1000h](http://www.aihub.or.kr/aidata/105) dataset which contains 1,000 hours korean voice data. and, our project is currently in progress.   
At present our top model has recorded an **80% CRR**, and we are working for a higher recognition rate.  
  
Also our model has recorded **91% CRR** in [Kadi-zeroth dataset](https://github.com/goodatlas/zeroth).  
  
###### ( **CRR** : Character Recognition Rate )  
  
We are constantly updating the progress of the project on the [Wiki page](https://github.com/sooftware/Korean-Speech-Recognition/wiki).  Please check this page.  
  
## Installation
This project recommends Python 3.7 or higher.   
We recommend creating a new virtual environment for this project (using virtual env or conda).  

### Prerequisites
  
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* Pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.
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

Refer [here](https://github.com/sooftware/End-to-End-Korean-Speech-Recognition/wiki/Preparation-before-Training) before Training.  
The above document is written in Korean.  
We will also write a document in English as soon as possible, so please wait a little bit.  
  
If you already have another dataset, please modify the data set path to [definition.py](https://github.com/sooftware/End-to-End-Korean-Speech-Recognition/blob/master/package/definition.py) as appropriate.  

### Train and Test
if you want to start training, you should run [train.py](https://github.com/sooftware/End-to-End-Korean-Speech-Recognition/blob/master/train.py).    
or after training, you want to start testing, you should run [test.py](https://github.com/sooftware/End-to-End-Korean-Speech-Recognition/blob/master/test.py).  
  
you can set up a configuration [config.py](https://github.com/sooftware/End-to-End-Korean-Speech-Recognition/blob/master/package/config.py).  
An explanation of configuration is [here](https://sooftware.github.io/End-to-End-Korean-Speech-Recognition/Config.html).  
  
### Incorporating External Language Model in Performance Test
We introduce incorporating external language model in performance test.  
if you are interested in this content, please check [here](https://github.com/sooftware/char-rnnlm).
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/Korean-Speech-Recognition/issues) on Github.   
For live discussions, please go to our [gitter](https://gitter.im/Korean-Speech-Recognition/community) or Contacts sh951011@gmail.com please.
  
We appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  

### Code Style
We follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
    
### Reference   
[[1] 「Listen, Attend and Spell」  Paper](https://arxiv.org/abs/1508.01211)   
[[2] 「State-of-the-art Speech Recognition with Sequence-to-Sequence Models」   Paper](https://arxiv.org/abs/1712.01769)  
[[3] 「A Simple Data Augmentation Method for Automatic Speech Recognition」  Paper](https://arxiv.org/abs/1904.08779)  
[[4] 「An analysis of incorporating an external language model into a sequence-to-sequence model」  Paper](https://arxiv.org/abs/1712.01996)  
[[5] 「Voice Recognition Using MFCC Algorithm」  Paper](https://ijirae.com/volumes/vol1/issue10/27.NVEC10086.pdf)        
[[6]    IBM pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)   
[[7]    Character RNN Language Model](https://github.com/sooftware/char-rnnlm)  
[[8]    A.I Hub Korean Voice Dataset](http://www.aihub.or.kr/aidata/105)    
[[9]    Documentation](https://sooftware.github.io/End-to-End-Korean-Speech-Recognition/)  
   
### Citing
```
@source_code{
  title={Character-unit based End-to-End Korean Speech Recognition},
  author={Soohwan Kim, Seyoung Bae, Cheolhwang Won},
  year={2020}
}
```
