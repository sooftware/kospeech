<img src="https://user-images.githubusercontent.com/42150335/89766602-63a82380-db33-11ea-8f5f-916ea90e1ed1.png" width=750>
  
# KoSpeech: Korean Speech Recognition Toolkit

[![CodeFactor](https://www.codefactor.io/repository/github/sooftware/kospeech/badge)](https://www.codefactor.io/repository/github/sooftware/kospeech) [<img src="http://img.shields.io/badge/docs-passing-success">](https://sooftware.github.io/KoSpeech/) [<img src="http://img.shields.io/badge/help wanted-issue 37-ff">](https://github.com/sooftware/KoSpeech/issues/37) <img src="http://img.shields.io/badge/Run Seq2seq-success-success"> <img src="http://img.shields.io/badge/Run transformer-fail-red"> 

[KoSpeech: Korean Speech Recognition Toolkit](https://sooftware.github.io/KoSpeech/)     
  
[Soohwan Kim](https://github.com/sooftware)<sup>1,2</sup>, [Seyoung Bae](https://github.com/triplet02)<sup>1</sup>, [Cheolhwang Won](https://github.com/wch18735)<sup>1</sup>, [Suwon Park](https://ei.kw.ac.kr/introduction/professor_view.php?idx=72)<sup>1*</sup>
  
<sup>1</sup>Elcomm, Kwangwoon Univ. <sup>2</sup>Spoken Language Lab (of Sogang Univ.) <sup>*</sup> author is advisor to this work  
  
`KoSpeech` is an End-to-End open source project for Korean speech recognition. The goal of this work is to help research speech recognition. It was developed with a focus on readability and extensibility of code. Learning is possible through various options such as feature extraction, attention mechanism, and data augmentation, etc..   
We used `KsponSpeech` corpus which containing **1000h** of Korean speech data. At present our model has recorded an **10.31%** character error rate. We are always updating this work for increased recognition rate and extensibility.   
We appreciate any kind of [feedback or contribution](https://github.com/sooftware/End-to-end-Speech-Recognition/issues).    
  
|Description|Dataset|Model|  
|-----------|-------|-----|  
|kospeech_seq2seq|[KsponSpeech](http://www.aihub.or.kr/aidata/105)|[download](https://drive.google.com/file/d/1Sq3ah7UijgxR-PGZfOVEY8aT12Y4JRdB/view?usp=sharing)|   
  
## Intro
  
End-to-end (E2E) automatic speech recognition (ASR) is an emerging paradigm in the field of neural network-based speech recognition that offers multiple benefits. Traditional “hybrid” ASR systems, which are comprised of an acoustic model, language model, and pronunciation model, require separate training of these components, each of which can be complex.   
  
For example, training of an acoustic model is a multi-stage process of model training and time alignment between the speech acoustic feature sequence and output label sequence. In contrast, E2E ASR is a single integrated approach with a much simpler training pipeline with models that operate at low audio frame rates. This reduces the training time, decoding time, and allows joint optimization with downstream processing such as natural language understanding.  
  
## Features  
  
* [End-to-end (E2E) automatic speech recognition](https://sooftware.github.io/KoSpeech/)
* [Various Options](https://sooftware.github.io/KoSpeech/notes/opts.html)
* [(VGG / DeepSpeech2) Extractor](https://sooftware.github.io/KoSpeech/Seq2seq.html#module-kospeech.models.seq2seq.sublayers)
* [MaskCNN & pack_padded_sequence](https://sooftware.github.io/KoSpeech/Seq2seq.html#module-kospeech.models.seq2seq.sublayers)
* [Attention (Multi-Head / Location-Aware / Additive / Scaled-dot)](https://sooftware.github.io/KoSpeech/Seq2seq.html#module-kospeech.models.seq2seq.attention)
* [Top K Decoding (Beam Search)](https://sooftware.github.io/KoSpeech/Seq2seq.html#module-kospeech.models.seq2seq.beam_search)
* [Various Feature (Spectrogram / Mel-Spectrogram / MFCC / Filter-Bank)](https://sooftware.github.io/KoSpeech/Data.html#module-kospeech.data.audio.feature)
* [Delete silence](https://sooftware.github.io/KoSpeech/Data.html#module-kospeech.data.audio.core)
* [SpecAugment / NoiseAugment](https://sooftware.github.io/KoSpeech/Data.html#module-kospeech.data.audio.augment)
* [Label Smoothing](https://sooftware.github.io/KoSpeech/Optim.html#module-kospeech.optim.loss)

* [Save & load Checkpoint](https://sooftware.github.io/KoSpeech/Checkpoint.html#id1)
* [Learning Rate Scheduling](https://sooftware.github.io/KoSpeech/Optim.html#module-kospeech.optim.lr_scheduler)
* [Implement data loader as multi-thread for speed](https://sooftware.github.io/KoSpeech/Data.html#module-kospeech.data.data_loader)
* Scheduled Sampling (Teacher forcing scheduling)
* Inference with batching
* Multi-GPU training
  
We have referred to several papers to develop the best model possible. And tried to make the code as efficient and easy to use as possible. If you have any minor inconvenience, please let us know anytime.   
We will response as soon as possible.

## Roadmap
  
<img src="https://user-images.githubusercontent.com/42150335/87572553-afb7a200-c706-11ea-9b5e-cd7b6b832f01.png"> 
  
### Seq2seq
  
Sequence-to-Sequence can be trained with serveral options. You can choose the CNN extractor from (`ds2` /`vgg`),   
You can choose attention mechanism from (`location-aware`, `multi-head`, `additive`, `scaled-dot`) attention.
  
Our architecture based on Listen Attend and Spell.   
We mainly referred to following papers.  
  
[Wiliam Chan et al.「Listen, Attend and Spell」ICASSP 2016](https://arxiv.org/abs/1508.01211)  
  
[Ashish Vaswani et al 「Attention Is All You Need」 NIPS 2017
](https://arxiv.org/abs/1706.03762)  
  
[Chiu et al 「StateOf-The-Art Speech Recognition with Sequence-to-Sequence Models」 ICASSP 2018
](https://arxiv.org/abs/1712.01769)  
  
[Daniel S. Park et al 「SpecAugment: A Simple Data Augmentation Method for ASR」 Interspeech 2019
](https://arxiv.org/abs/1904.08779)  
      
Our Seq2seq architeuture is as follows.
  
```python
Seq2seq(
  (encoder): Seq2seqEncoder(
    (conv): VGGExtractor(
      (conv): Sequential(
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
    (rnn): LSTM(2560, 512, num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)
  )
  (decoder): Seq2seqDecoder(
    (embedding): Embedding(2038, 1024)
    (input_dropout): Dropout(p=0.3, inplace=False)
    (rnn): LSTM(1024, 1024, num_layers=2, batch_first=True, dropout=0.3)
    (attention): AddNorm(
      (sublayer): MultiHeadAttention(
        (query_proj): Linear(in_features=1024, out_features=1024, bias=True)
        (key_proj): Linear(in_features=1024, out_features=1024, bias=True)
        (value_proj): Linear(in_features=1024, out_features=1024, bias=True)
      )
      (layer_norm): LayerNorm(1024)
    )
    (projection): AddNorm(
      (sublayer): Linear(in_features=1024, out_features=1024, bias=True)
      (layer_norm): LayerNorm(1024)
    )
    (generator): Linear(in_features=1024, out_features=2038, bias=False)
  )
)
``` 
  
### Transformer  
  
The Transformer model is currently implemented, but the code for learning is not implemented.  
We will implement as soon as possible.  
  
We mainly referred to following papers.
  
[Ashish Vaswani et al 「Attention Is All You Need」 NIPS 2017
](https://arxiv.org/abs/1706.03762)  
  
### Various Options   
  
You can choose feature extraction method from (`spectrogram`, `mel-spectrogram`, `mfcc`, `filter-bank`).   
In addition to this, You can see a variety of options [here](https://sooftware.github.io/KoSpeech/notes/opts.html).  
  
* Options
```
usage: main.py [-h] [--mode MODE] [--sample_rate SAMPLE_RATE]
               [--frame_length FRAME_LENGTH] [--frame_shift FRAME_SHIFT]
               [--n_mels N_MELS] [--normalize] [--del_silence]
               [--input_reverse] [--feature_extract_by FEATURE_EXTRACT_BY]
               [--transform_method TRANSFORM_METHOD]
               [--time_mask_para TIME_MASK_PARA]
               [--freq_mask_para FREQ_MASK_PARA]
               [--time_mask_num TIME_MASK_NUM] [--freq_mask_num FREQ_MASK_NUM]
               [--architecture ARCHITECTURE] [--use_bidirectional]
               [--mask_conv] [--hidden_dim HIDDEN_DIM] [--dropout DROPOUT]
               [--num_heads NUM_HEADS] [--label_smoothing LABEL_SMOOTHING]
               [--num_encoder_layers NUM_ENCODER_LAYERS]
               [--num_decoder_layers NUM_DECODER_LAYERS] [--rnn_type RNN_TYPE]
               [--extractor EXTRACTOR] [--activation ACTIVATION]
               [--attn_mechanism ATTN_MECHANISM]
               [--teacher_forcing_ratio TEACHER_FORCING_RATIO]
               [--num_classes NUM_CLASSES] [--d_model D_MODEL]
               [--ffnet_style FFNET_STYLE] [--dataset_path DATASET_PATH]
               [--data_list_path DATA_LIST_PATH] [--label_path LABEL_PATH]
               [--spec_augment] [--noise_augment]
               [--noiseset_size NOISESET_SIZE] [--noise_level NOISE_LEVEL]
               [--use_cuda] [--batch_size BATCH_SIZE]
               [--num_workers NUM_WORKERS] [--num_epochs NUM_EPOCHS]
               [--init_lr INIT_LR] [--high_plateau_lr HIGH_PLATEAU_LR]
               [--low_plateau_lr LOW_PLATEAU_LR] [--valid_ratio VALID_RATIO]
               [--max_len MAX_LEN] [--max_grad_norm MAX_GRAD_NORM]
               [--rampup_period RAMPUP_PERIOD]
               [--decay_threshold DECAY_THRESHOLD]
               [--exp_decay_period EXP_DECAY_PERIOD]
               [--teacher_forcing_step TEACHER_FORCING_STEP]
               [--min_teacher_forcing_ratio MIN_TEACHER_FORCING_RATIO]
               [--seed SEED] [--save_result_every SAVE_RESULT_EVERY]
               [--checkpoint_every CHECKPOINT_EVERY]
               [--print_every PRINT_EVERY] [--resume]

```
  
### KoSpeech
  
`kospeech` module has modularized and extensible components for las models, trainer, evaluator, checkpoints etc...   
In addition, `kospeech` enables learning in a variety of environments with a simple option setting.  
  
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
python bin/setup.py build
python bin/setup.py install
```
  
## Get Started

### Step 0: Try quick speech recognition with a pre-trained model.

* Command
```
$ ./run_pretrain.sh
```
* Output
```
아 뭔 소리야 그건 또
```  
You can get a quick look of pre-trained model's inference, with a sample data.  
  
### Step 1: Data Preprocessing  
    
you can preprocess `KsponSpeech corpus` refer [wiki](https://github.com/sooftware/KoSpeech/wiki/Preparation-before-Training) or [this repo](https://github.com/sooftware/KsponSpeech-preprocess).       
This documentation contains information regarding the preprocessing of `KsponSpeech`.   

### Step 2: Run `main.py`
* Default setting  
```
$ ./run.sh
```
* Custom setting
```shell
python ./bin/main.py -batch_size 32 -num_workers 4 -num_epochs 20  -spec_augment
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
python ./bin/eval.py -dataset_path dataset_path -data_list_path data_list_path -mode eval
```
Now you have a model which you can use to predict on new data. We do this by running `greedy search` or `beam search`.  
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
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/End-to-end-Speech-Recognition/issues) on Github.   
For live discussions, please go to our [gitter](https://gitter.im/Korean-Speech-Recognition/community) or Contacts sh951011@gmail.com please.
  
We appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
## TODO List
  
* [X] Add Transformer model 
* [ ] Train with Transformer model
* [ ] Inference with Transformer model
* [ ] Add CTC with beam search (Connectionist Temporal Classification)
  
### Code Style
We follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
    
### Paper References
  
Ilya Sutskever et al. [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) arXiv: 1409.3215  
  
Dzmitry Bahdanau et al. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) arXiv: 1409.0473   
  
Jan Chorowski et al. [Attention Based Models for Speech Recognition](https://arxiv.org/abs/1506.07503) arXiv: 1506.07503    
  
Wiliam Chan et al. [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211) arXiv: 1508.01211   
   
Dario Amodei et al. [Deep Speech2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595) arXiv: 1512.02595   
   
Takaaki Hori et al. [Advances in Joint CTC-Attention based E2E Automatic Speech Recognition with a Deep CNN Encoder and RNN-LM](https://arxiv.org/abs/1706.02737) arXiv: 1706.02737   
  
Ashish Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) arXiv: 1706.03762     
  
Chung-Cheng Chiu et al. [State-of-the-art Speech Recognition with Sequence-to-Sequence Models](https://arxiv.org/abs/1712.01769) arXiv: 1712.01769   
  
Anjuli Kannan et al. [An Analysis Of Incorporating An External LM Into A Sequence-to-Sequence Model](https://arxiv.org/abs/1712.01996) arXiv: 1712.01996  
  
Daniel S. Park et al. [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779) arXiv: 1904.08779     
    
Rafael Muller et al. [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) arXiv: 1906.02629   
    
Jung-Woo Ha et al. [ClovaCall: Korean Goal-Oriented Dialog Speech Corpus for Automatic Speech Recognition of Contact Centers](https://arxiv.org/abs/2004.09367) arXiv: 2004.09367
    
### Github References
  
[IBM/Pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)  
  
[SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)  
  
[kaituoxu/Speech-Transformer](https://github.com/kaituoxu/Speech-Transformer)  
  
[OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)  
  
[clovaai/ClovaCall](https://github.com/clovaai/ClovaCall)
  
### Citing
```
@github{
  title = {KoSpeech: Open Source Project for Korean End-to-End Automatic Speech Recognition in PyTorch},
  author = {Soohwan Kim, Seyoung Bae, Cheolhwang Won, Suwon Park},
  publisher = {GitHub},
  docs = {https://sooftware.github.io/KoSpeech/},
  url = {https://github.com/sooftware/KoSpeech},
  year = {2020}
}
```
