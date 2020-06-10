<p align=center><i> <img src="https://user-images.githubusercontent.com/42150335/84234109-95e6d600-ab2e-11ea-9112-7becc7e15a66.png" width=500> </i></p>   
<p align=center> <img src="https://img.shields.io/badge/build-passing-success?logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/license-Apache--2.0-informational?logo=Apache&logoColor=white"> <img src="https://img.shields.io/badge/Windows-succeeded-success?logo=Windows&logoColor=white"> <img src="https://img.shields.io/badge/MacOS-not tested-informational?logo=Apple&logoColor=white"> <img src="https://img.shields.io/badge/Linux-succeeded-success?logo=Linux&logoColor=white"> </p> 
  
### [**Documentation**](https://sooftware.github.io/KoSpeech/)   
  
[Korean.ver](https://github.com/sooftware/KoSpeech/blob/master/README_ko.md)  

## Intro

`KoSpeech` is project for End-to-end (E2E) automatic speech recognition implemented in [PyTorch](http://pytorch.org).   
`KoSpeech` has modularized and extensible components for las models, training and evalutaion, checkpoints, parsing etc.   
We appreciate any kind of [feedback or contribution](https://github.com/sooftware/End-to-end-Speech-Recognition/issues).
  
We used `KsponSpeech` corpus which containing **1000h** of Korean speech data.   
At present our model has recorded an **86.98% CRR**, and we are working for a higher recognition rate.  
Also our model has recorded **91.0% CRR** in `Kaldi-zeroth corpus`    
  
###### ( **CRR** : Character Recognition Rate ) 
  
## Features  
  
* [End-to-end (E2E) automatic speech recognition](https://sooftware.github.io/KoSpeech/)
* [Various Options](https://sooftware.github.io/KoSpeech/notes/opts.html)
* [(VGG / DeepSpeech2) Extractor](https://sooftware.github.io/KoSpeech/Model.html#module-kospeech.model.convolutional)
* [MaskCNN & pack_padded_sequence](https://sooftware.github.io/KoSpeech/Model.html#module-kospeech.model.convolutional)
* [Multi-headed (location-aware / scaled dot-product) Attention](https://sooftware.github.io/KoSpeech/Model.html#module-kospeech.model.attention)
* [Top K Decoding (Beam Search)](https://sooftware.github.io/KoSpeech/Model.html#module-kospeech.model.beam_search)
* [MelSpectrogram Parser](https://sooftware.github.io/KoSpeech/Data.html#module-kospeech.data.preprocess.parser)
* [Delete silence](https://sooftware.github.io/KoSpeech/Data.html#module-kospeech.data.preprocess.audio)
* [SpecAugment](https://sooftware.github.io/KoSpeech/Data.html#module-kospeech.data.preprocess.augment)
* [NoiseAugment](https://sooftware.github.io/KoSpeech/Data.html#module-kospeech.data.preprocess.augment)
* [Label Smoothing](https://sooftware.github.io/KoSpeech/Optim.html#module-kospeech.optim.loss)

* [Save & load Checkpoint](https://sooftware.github.io/KoSpeech/Checkpoint.html#id1)
* [Learning Rate Scheduling](https://sooftware.github.io/KoSpeech/Optim.html#module-kospeech.optim.lr_scheduler)
* [Implement data loader as multi-thread for speed](https://sooftware.github.io/KoSpeech/Data.html#module-kospeech.data.data_loader)
* Scheduled Sampling (Teacher forcing scheduling)
* Inference with batching
* Multi-GPU training
  
We have referred to many papers to develop the best model possible. And tried to make the code as efficient and easy to use as possible. If you have any minor inconvenience, please let us know anytime. We will response as soon as possible.

## Roadmap
  
<img src="https://user-images.githubusercontent.com/42150335/83952233-5ee49c00-a872-11ea-8c98-5b98236125e1.png" width=450> 
  
End-to-end (E2E) automatic speech recognition (ASR) is an emerging paradigm in the field of neural network-based speech recognition that offers multiple benefits. Traditional “hybrid” ASR systems, which are comprised of an acoustic model, language model, and pronunciation model, require separate training of these components, each of which can be complex.   
  
For example, training of an acoustic model is a multi-stage process of model training and time alignment between the speech acoustic feature sequence and output label sequence. In contrast, E2E ASR is a single integrated approach with a much simpler training pipeline with models that operate at low audio frame rates. This reduces the training time, decoding time, and allows joint optimization with downstream processing such as natural language understanding.  
  
We mainly referred to following papers.  
  
 [「Listen, Attend and Spell」](https://arxiv.org/abs/1508.01211)  
   
[「Attention Based Models for Speech Recognition」](https://arxiv.org/abs/1506.07503)  

[「State-of-the-art Speech Recognition with Sequence-to-Sequence Models」](https://arxiv.org/abs/1712.01769)
   
[「SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition」](https://arxiv.org/abs/1904.08779).   
  
If you want to study the feature of audio, we recommend this papers.  
  
[「Voice Recognition Using MFCC Algirithm」](https://ijirae.com/volumes/vol1/issue10/27.NVEC10086.pdf).  
  
Our project based on Seq2seq with Attention Architecture.  
  
![image](https://user-images.githubusercontent.com/42150335/83260135-36b2c880-a1f4-11ea-8b38-ef88dca214bf.png)
  
`Attention mechanism` helps finding speech alignment. We apply multi-headed (`location-aware` / `scaled dot-product`) attention which you can choose. Location-aware attention proposed in `Attention Based Models for Speech Recognition` paper and Multi-headed attention proposed in `Attention Is All You Need` paper. You can choose between these two options as `attn_mechanism` option. Please [check](https://sooftware.github.io/KoSpeech/notes/opts.html) this page.    
  
Our model architeuture is as follows.
  
```python
ListenAttendSpell(
  (listener): Listener(
    (extractor): VGGExtractor(
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
    (rnn): LSTM(2560, 256, num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)
  )
  (speller): Speller(
    (embedding): Embedding(2038, 512)
    (input_dropout): Dropout(p=0.3, inplace=False)
    (rnn): LSTM(512, 512, num_layers=2, batch_first=True, dropout=0.3)
    (attention): MultiHeadAttention(
      (query_projection): Linear(in_features=512, out_features=512, bias=True)
      (value_projection): Linear(in_features=512, out_features=512, bias=True)
    )
    (out_projection): Linear(in_features=1024, out_features=2038, bias=True)
  )
)
``` 
  
### KoSpeech

<img src="https://user-images.githubusercontent.com/42150335/83944090-d8ad6300-a83b-11ea-8a2c-2f0d9ba0e54d.png" width=700>   
  
`kospeech` module has modularized and extensible components for las models, trainer, evaluator, checkpoints etc...   
In addition, `kospeech` enables learning in a variety of environments with a simple option setting.  
  
* Options
```
usage: main.py [-h] [--mode] [--sample_rate]
               [--window_size] [--stride] [--n_mels]
               [--normalize] [--del_silence] [--input_reverse]
               [--feature_extract_by] [--time_mask_para] [--freq_mask_para]
               [--time_mask_num] [--freq_mask_num]
               [--use_bidirectional] [--hidden_dim]
               [--dropout] [--num_heads] [--label_smoothing]
               [--listener_layer_size] [--speller_layer_size] [--rnn_type]
               [--extractor] [--activation]
               [--attn_mechanism] [--teacher_forcing_ratio]
               [--dataset_path] [--data_list_path]
               [--label_path] [--init_uniform] [--spec_augment]
               [--noise_augment] [--noiseset_size]
               [--noise_level] [--use_cuda]
               [--batch_size] [--num_workers]
               [--num_epochs] [--init_lr]
               [--high_plateau_lr] [--low_plateau_lr] [--valid_ratio]
               [--max_len] [--max_grad_norm]
               [--rampup_period] [--decay_threshold] [--exp_decay_period]
               [--teacher_forcing_step] [--min_teacher_forcing_ratio]
               [--seed] [--save_result_every]
               [--checkpoint_every] [--print_every] [--resume]
```

We are constantly updating the progress of the project on the [Wiki page](https://github.com/sooftware/End-to-end-Speech-Recognition/wiki).  Please check this page.  
  
## Installation
This project recommends Python 3.7 or higher.   
We recommend creating a new virtual environment for this project (using virtual env or conda).  

### Prerequisites
  
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* Pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.   
* Pandas: `pip install pandas` (Refer [here](https://github.com/pandas-dev/pandas) for problem installing Pandas)  
* Matplotlib: `pip install matplotlib` (Refer [here](https://github.com/matplotlib/matplotlib) for problem installing Matplotlib)
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
### Step 1: Data Preprocessing  
    
you can preprocess `KsponSpeech corpus` refer [here](https://github.com/sooftware/KsponSpeech.preprocess).     
Or refer [this page](https://github.com/sooftware/KoSpeech/wiki/Preparation-before-Training). This documentation contains information regarding the preprocessing of `KsponSpeech`.   

### Step 2: Run `main.py`
* Default setting  
```
$ ./run.sh
```
* Custom setting
```shell
python ./bin/main.py --batch_size 32 --num_workers 4 --num_epochs 20  --use_bidirectional \
                     --input_reverse --spec_augment --noise_augment --use_cuda --hidden_dim 256 \
                     --dropout 0.3 --num_heads 8 --label_smoothing 0.1 \
                     --listener_layer_size 5 --speller_layer_size 3 --rnn_type gru \
                     --high_plateau_lr $HIGH_PLATEAU_LR --teacher_forcing_ratio 1.0 --valid_ratio 0.01 \
                     --sample_rate 16000 --window_size 20 --stride 10 --n_mels 80 --normalize --del_silence \
                     --feature_extract_by torchaudio --time_mask_para 70 --freq_mask_para 12 \
                     --time_mask_num 2 --freq_mask_num 2 --save_result_every 1000 \
                     --checkpoint_every 5000 --print_every 10 --init_lr 1e-15  --init_uniform  \
                     --mode train --dataset_path /data3/ --data_list_path ./data/data_list/xxx.csv \
                     --max_grad_norm 400 --rampup_period 1000 --max_len 80 --decay_threshold 0.02 \
                     --exp_decay_period  160000 --low_plateau_lr 1e-05 --noiseset_size 1000 \
                     --noise_level 0.7 --attn_mechanism loc --teacher_forcing_step 0.05 \
                     --min_teacher_forcing_ratio 0.7
```
  
You can train the model by above command.  
 If you want to train by default setting, you can train by `Defaulting setting` command.   
 Or if you want to train by custom setting, you can designate hyperparameters by `Custom setting` command.

### Step 3: Run `eval.py`
* Default setting
```
$ ./eval.sh
```
* Custom setting
```
python ./bin/eval.py -dataset_path dataset_path -data_list_path data_list_path \
                     -mode eval -use_cuda -batch_size 32 -num_workers 4 \
                     -use_beam_search -k 5 -print_every 100 \
                     -sample_rate 16000 --window_size 20 --stride 10 --n_mels 80 -feature_extract_by librosa \
                     -normalize -del_silence -input_reverse 
```
Now you have a model which you can use to predict on new data. We do this by running `beam search` (or `greedy search`).  
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
[[1] 「Listen, Attend and Spell」  @Paper](https://arxiv.org/abs/1508.01211)   
[[2] 「Attention Based Models for Speech Recognition」  @Paper](https://arxiv.org/abs/1506.07503)  
[[3] 「State-of-the-art Speech Recognition with Sequence-to-Sequence Models」   @Paper](https://arxiv.org/abs/1712.01769)  
[[4] 「A Simple Data Augmentation Method for Automatic Speech Recognition」  @Paper](https://arxiv.org/abs/1904.08779)  
[[5] 「Voice Recognition Using MFCC Algorithm」  @Paper](https://ijirae.com/volumes/vol1/issue10/27.NVEC10086.pdf)        
[[6] IBM/pytorch-seq2seq @gitHub](https://github.com/IBM/pytorch-seq2seq)   
[[7] SeanNaren/deepspeech.pytorch @github](https://github.com/SeanNaren/deepspeech.pytorch)   
[[8] Alexander-H-Liu/End-to-end-ASR-Pytorch @github](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch)   
[[9] clovaai/ClovaCall @github](https://github.com/clovaai/ClovaCall)  
[[10] KsponSpeech @AIHub](http://www.aihub.or.kr/aidata/105)    
[[11] KsponSpeech.preprocess @github](https://github.com/sooftware/KsponSpeech.preprocess)    
[[12] Documentation](https://sooftware.github.io/End-to-End-Korean-Speech-Recognition/)  
   
### Citing
```
@github{
  title = {KoSpeech},
  author = {Soohwan Kim, Seyoung Bae, Cheolhwang Won},
  publisher = {GitHub},
  docs = {https://sooftware.github.io/KoSpeech/},
  url = {https://github.com/sooftware/KoSpeech},
  year = {2020}
}
```
