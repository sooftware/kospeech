# **End-to-end Speech Recognition**  
  
### Character unit based end2end automatic speech recognition in Korean  
   
[<img src="https://github.com/gentaiscool/end2end-asr-pytorch/raw/master/img/pytorch-logo-dark.png" height=18>](https://pytorch.org/) <img src="https://img.shields.io/badge/License-Apache--2.0-yellow" height=20> [<img src="https://img.shields.io/badge/chat-on%20gitter-4fb99a" height=20>](https://gitter.im/Korean-Speech-Recognition/community)   
  
### [**Documentation**](https://sooftware.github.io/End-to-end-Speech-Recognition/)   
  
## Intro

`End-to-end Speech Recognition` is project for E2E automatic speech recognition implemented in [PyTorch](http://pytorch.org).   
`e2e` has modularized and extensible components for las models, training and inference, checkpoints, parsing etc.   
We appreciate any kind of [feedback or contribution](https://github.com/sooftware/End-to-end-Speech-Recognition/issues).
  
We used `KsponSpeech` corpus which containing **1000h** of Korean speech data.   
At present our model has recorded an **86.98% CRR**, and we are working for a higher recognition rate.  
Also our model has recorded **91.0% CRR** in `Kaldi-zeroth corpus`    
  
###### ( **CRR** : Character Recognition Rate ) 
  
## Features  
  
* [End-to-end (E2E) automatic speech recognition](https://sooftware.github.io/End-to-end-Speech-Recognition/)
* [Convolutional encoder](https://sooftware.github.io/End-to-end-Speech-Recognition/Model.html#module-e2e.model.listener)
* [MaskConv & pack_padded_sequence](https://sooftware.github.io/End-to-end-Speech-Recognition/Model.html#module-e2e.model.maskCNN)
* [Multi-headed Location-Aware Attention](https://sooftware.github.io/End-to-end-Speech-Recognition/Model.html#module-e2e.model.attention)
* [Top K Decoding (Beam Search)](https://sooftware.github.io/End-to-end-Speech-Recognition/Model.html#module-e2e.model.topk_decoder)
* [Spectrogram Parser](https://sooftware.github.io/End-to-end-Speech-Recognition/Feature.html#module-e2e.feature.parser)
* [Delete silence](https://sooftware.github.io/End-to-end-Speech-Recognition/Feature.html#module-e2e.feature.parser)
* [SpecAugment](https://sooftware.github.io/End-to-end-Speech-Recognition/Feature.html#module-e2e.feature.parser)
* [NoiseAugment](https://sooftware.github.io/End-to-end-Speech-Recognition/Feature.html#module-e2e.feature.parser)
* [Label Smoothing](https://sooftware.github.io/End-to-end-Speech-Recognition/Loss.html)
* [Save & load Checkpoint](https://sooftware.github.io/End-to-end-Speech-Recognition/Modules.html#module-e2e.modules.checkpoint)
* [Various options can be set using parser](https://sooftware.github.io/End-to-end-Speech-Recognition/Modules.html#module-e2e.modules.opts)
* [Implement data loader as multi-thread for speed](https://sooftware.github.io/End-to-end-Speech-Recognition/Data_loader.html#id1)
* Inference with batching
* Multi-GPU training
* Show training states as log
  
## Roadmap
  
<img src="https://user-images.githubusercontent.com/42150335/80630547-5dfc6580-8a8f-11ea-91e8-73fe5e8b9e4b.png" width=450> 
  
End-to-end (E2E) automatic speech recognition (ASR) is an emerging paradigm in the field of neural network-based speech recognition that offers multiple benefits. Traditional “hybrid” ASR systems, which are comprised of an acoustic model, language model, and pronunciation model, require separate training of these components, each of which can be complex.   
  
For example, training of an acoustic model is a multi-stage process of model training and time alignment between the speech acoustic feature sequence and output label sequence. In contrast, E2E ASR is a single integrated approach with a much simpler training pipeline with models that operate at low audio frame rates. This reduces the training time, decoding time, and allows joint optimization with downstream processing such as natural language understanding.  
  
We mainly referred to following papers.  
  
 [「Listen, Attend and Spell」](https://arxiv.org/abs/1508.01211)  
 
[「State-of-the-art Speech Recognition with Sequence-to-Sequence Models」](https://arxiv.org/abs/1712.01769)
   
[「SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition」](https://arxiv.org/abs/1904.08779).   
  
If you want to study the feature of audio, we recommend this papers.  
  
[「Voice Recognition Using MFCC Algirithm」](https://ijirae.com/volumes/vol1/issue10/27.NVEC10086.pdf).  
  
Our project based on Seq2seq with Attention Architecture.  

Sequence to sequence architecture is a field that is still actively studied in the field of speech recognition.    
Our model architeuture is as follows.
  
```python
ListenAttendSpell(
  (listener): Listener(
    (rnn): GRU(2560, 256, num_layers=5, batch_first=True, dropout=0.3, bidirectional=True)
    (cnn): MaskCNN(
      (sequential): Sequential(
        (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): Hardtanh(min_val=0, max_val=20, inplace=True)
        (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): Hardtanh(min_val=0, max_val=20, inplace=True)
        (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (8): Hardtanh(min_val=0, max_val=20, inplace=True)
        (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (11): Hardtanh(min_val=0, max_val=20, inplace=True)
        (12): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
    )
  )
  (speller): Speller(
    (rnn): GRU(512, 512, num_layers=3, batch_first=True, dropout=0.3)
    (embedding): Embedding(2038, 512)
    (input_dropout): Dropout(p=0.3, inplace=False)
    (attention): MultiHybridAttention(
      (scaled_dot): ScaledDotProductAttention()
      (conv1d): Conv1d(1, 10, kernel_size=(3,), stride=(1,), padding=(1,))
      (linear_q): Linear(in_features=512, out_features=512, bias=True)
      (linear_v): Linear(in_features=512, out_features=512, bias=False)
      (linear_u): Linear(in_features=10, out_features=64, bias=False)
      (linear_out): Linear(in_features=1024, out_features=512, bias=True)
      (normalize): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (linear_out): Linear(in_features=512, out_features=2038, bias=True)
  )
)
``` 
  
### e2e module

<img src="https://user-images.githubusercontent.com/42150335/82842192-93239880-9f13-11ea-80d6-ed1358218d5e.png" width=800>   
  
`e2e` (End-to-end) module's structure is implement as above.   
`e2e` module has modularized and extensible components for las models, trainer, evaluator, checkpoints, data_loader etc...  
  
We are constantly updating the progress of the project on the [Wiki page](https://github.com/sooftware/End-to-end-Speech-Recognition/wiki).  Please check this page.  
  
## Installation
This project recommends Python 3.7 or higher.   
We recommend creating a new virtual environment for this project (using virtual env or conda).  

### Prerequisites
  
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* Pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.   
* Pandas: `pip install pandas` (Refer [here](https://github.com/pandas-dev/pandas) for problem installing Pandas)  
* librosa: `pip install librosa` (Refer [here](https://github.com/librosa/librosa) for problem installing librosa)
* torchaudio: `pip install torchaudio` (Refer [here](https://github.com/pytorch/pytorch) for problem installing torchaudio)
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
### Step 1: Preparation dataset

Refer [here](https://github.com/sooftware/End-to-end-Speech-Recognition/wiki/Preparation-before-Training) before training. this document contains information regarding the preprocessing of `KsponSpeech`  
The above document is written in Korean.  
We will also write a document in English as soon as possible, so please wait a little bit.  

### Step 2: Run `train.py`
* Default setting  
```
$ ./train.sh
```
* Custom setting
```
python ./train.py -dataset_path dataset_path -data_list_path data_list_path \
                  -use_multi_gpu -init_uniform -mode train -batch_size 32 -num_workers 4 \
                  -num_epochs 20 -spec_augment -noise_augment -max_len 151 \
                  -use_cuda -lr 3e-04 -min_lr 1e-05 -lr_patience 1/3 -valid_ratio 0.01 \
                  -label_smoothing 0.1 -save_result_every 1000 -print_every 10 -checkpoint_every 5000 \
                  -use_bidirectional -hidden_dim 256 -dropout 0.3 -num_heads 8 -rnn_type gru \
                  -listener_layer_size 5 -speller_layer_size 3 -teacher_forcing_ratio 0.99 \ 
                  -input_reverse -normalize -del_silence -sample_rate 16000 -window_size 20 -stride 10 -n_mels 80 \
                  -feature_extract_by librosa -time_mask_para 50 -freq_mask_para 12 \
                  -time_mask_num 2 -freq_mask_num 2 -max_norm 400 -rampup_period 1000 -rampup_power 3
```
  
You can train the model by above command.  
 If you want to train by default setting, you can train by `Defaulting setting` command.   
 Or if you want to train by custom setting, you can designate hyperparameters by `Custom setting` command.


### Step 3: Run `infer.py`
* Default setting
```
$ ./infer.sh
```
* Custom setting
```
python ./infer.py -dataset_path dataset_path -data_list_path data_list_path \
                  -mode infer -use_multi_gpu -use_cuda -batch_size 32 -num_workers 4 \
                  -use_beam_search -k 5 -print_every 100 \
                  -sample_rate 16000 --window_size 20 --stride 10 --n_mels 80 -feature_extract_by librosa \
                  -normalize -del_silence -input_reverse 
```
Now you have a model which you can use to predict on new data. We do this by running beam search (or greedy search).  
Like training, you can choose between `Default setting` or `Custom setting`.  
  
### Checkpoints   
Checkpoints are organized by experiments and timestamps as shown in the following file structure.  
```
save_dir
+-- checkpoints
|  +-- YYYY_mm_dd_HH_MM_SS
   |  +-- trainer_states.pt
   |  +-- model.pt
```
You can resume and load from checkpoints.
  
### Incorporating External Language Model in Performance Test
We introduce incorporating external language model in performance test.  
If you are interested in this content, please check [here](https://github.com/sooftware/char-rnnlm).
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/End-to-end-Speech-Recognition/issues) on Github.   
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
[[6] 「IBM pytorch-seq2seq」](https://github.com/IBM/pytorch-seq2seq)   
[[7] 「SeanNaren deepspeech.pytorch」](https://github.com/SeanNaren/deepspeech.pytorch)   
[[8] 「Alexander-H-Liu End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch)   
[[9] 「Character RNN Language Model」](https://github.com/sooftware/char-rnnlm)  
[[10] 「KsponSpeech」](http://www.aihub.or.kr/aidata/105)    
[[11] 「Documentation」](https://sooftware.github.io/End-to-End-Korean-Speech-Recognition/)  
   
### Citing
```
@github{
  title = {End-to-end Speech Recognition},
  author = {Soohwan Kim, Seyoung Bae, Cheolhwang Won},
  publisher = {GitHub},
  docs = {https://sooftware.github.io/End-to-end-Speech-Recognition/},
  url = {https://github.com/sooftware/End-to-end-Speech-Recognition},
  year = {2020}
}
```
