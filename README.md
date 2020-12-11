# [UPDATED] KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition

[![CodeFactor](https://www.codefactor.io/repository/github/sooftware/kospeech/badge)](https://www.codefactor.io/repository/github/sooftware/kospeech) 
[<img src="http://img.shields.io/badge/docs-passing-success">](https://sooftware.github.io/KoSpeech/) 
[<img src="http://img.shields.io/badge/help wanted-issue 37-ff">](https://github.com/sooftware/KoSpeech/issues/37) 
<img src="https://img.shields.io/github/stars/sooftware/KoSpeech"> 
<img src="https://img.shields.io/github/forks/sooftware/KoSpeech">  
  
### What's New
  
- December 2020: Joint CTC-Attention Updated (*Currently, Not Supports Multi-GPU*)
- November 2020: Deep Speech 2 Architecture Updated
- November 2020: KsponSpeech Subword & Grapheme Unit Updated (*Not Tested*)
- November 2020: RAdam & AdamP Optimizer Updated
  
### Note
  
- Currently, beam search may not work properly.  
- The pre-train model is currently not working properly, but will be uploaded as soon as the current learning is complete.  
- Currently, CUDA OOM error is occurring at the end of 1 epoch. We will fix it as soon as I know the cause.
  
### ***[KoSpeech:  Open-Source Toolkit for End-to-End Korean Speech Recognition \[Technical Report\]](https://arxiv.org/abs/2009.03092)***
  
***KoSpeech***, an open-source software, is modular and extensible end-to-end Korean automatic speech recognition (ASR) toolkit based on the deep learning library PyTorch. Several automatic speech recognition open-source toolkits have been released, but all of them deal with non-Korean languages, such as English (e.g. ESPnet, Espresso). Although AI Hub opened 1,000 hours of Korean speech corpus known as KsponSpeech, there is no established preprocessing method and baseline model to compare model performances. Therefore, we propose preprocessing methods for KsponSpeech corpus and a baseline model for benchmarks. Our baseline model is based on Listen, Attend and Spell (LAS) architecture and ables to customize various training hyperparameters conveniently. By KoSpeech, we hope this could be a guideline for those who research Korean speech recognition. Our baseline model achieved **10.31%** character error rate (CER) at KsponSpeech corpus only with the acoustic model.  
  
### Pre-train Models
  
|Description|Feature|Dataset|Model|  
|-----------|:-----:|-------|-----|  
|las_vgg_multihead|librosa_mfcc_40|[KsponSpeech](http://www.aihub.or.kr/aidata/105)|[download](https://drive.google.com/file/d/1Lr-WYpXSlhPIxSE_sBxUedBtcBJzWni2/view?usp=sharing)|  
|las_vgg_multihead|kaldi_fbank_80|[KsponSpeech](http://www.aihub.or.kr/aidata/105)|[download](https://drive.google.com/file/d/1qhmV1vV8viB5W-rotDez2NztsmAt3mx0/view?usp=sharing)|  
|vad_model|-|-|[download](https://drive.google.com/file/d/14lLxfCiFgXqnb1a8dZ_AYhlKQeaMz7Jd/view?usp=sharing)|  
  
※ Please share the results of the experiment. Contribution is always welcome.
  
### Pre-processed Transcripts
  
|Dataset|Authentication|Output-Unit|Transcript|  
|-----------|-----|---|:-------:|  
|KsponSpeech|*Required*|Character|[download](https://drive.google.com/file/d/12IAJSTRqkPALx9AX_SKyC4KkFVQN6Lge/view?usp=sharing)|  
|KsponSpeech|*Required*|Subword|[download](https://drive.google.com/file/d/1awhfTpqAaDs7K5R9npvFoqeMYWiUtGtq/view?usp=sharing)|  
|KsponSpeech|*Required*|Grapheme|[download](https://drive.google.com/file/d/1awhfTpqAaDs7K5R9npvFoqeMYWiUtGtq/view?usp=sharing)|   
|LibriSpeech|*Unrequired* |Subword|[download](https://drive.google.com/file/d/1RA29SLtNIo1zmnk0OgVeXNH553Ul_DhY/view?usp=sharing)|  
  
※ Authentication : Refer to [[Link]](https://github.com/sooftware/KoSpeech/issues/54)  
  
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
  
![image](https://user-images.githubusercontent.com/42150335/101286627-df3e8680-382e-11eb-8d48-0536b4714ba0.png)
  
### (a) Deep Speech 2  
  
The Deep Speech 2 model can be trained with several options.   
Depending on the conditions, the loss may explode in the middle.  
We don't recommend too big a vocab size.  
  
### (b) Listen, Attend and Spell
  
Listen, Attend and Spell can be trained with serveral options. You can choose the CNN extractor from (`ds2` /`vgg`),   
You can choose attention mechanism from (`location-aware`, `multi-head`, `additive`, `scaled-dot`) attention.  
Also, you can train with joint CTC-Attention training.  
  
### (c) Transformer  
  
The Transformer model is currently implemented, but There is a bug, so I can't learn at the moment.    
We will fix as soon as possible.   
  
### Various Options   
  
We support various options for training. More details please check [here](https://sooftware.github.io/KoSpeech/notes/opts.html).  
  
* Options
```
usage: main.py [-h] [--mode MODE] [--sample_rate SAMPLE_RATE]
               [--frame_length FRAME_LENGTH] [--frame_shift FRAME_SHIFT]
               [--n_mels N_MELS] [--normalize] [--del_silence]
               [--input_reverse] [--feature_extract_by FEATURE_EXTRACT_BY]
               [--transform_method TRANSFORM_METHOD]
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
               [--transcripts_path TRANSCRIPTS_PATH]
               [--data_list_path DATA_LIST_PATH] [--spec_augment] [--use_cuda]
               [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
               [--num_epochs NUM_EPOCHS] [--init_lr INIT_LR]
               [--peak_lr PEAK_LR] [--final_lr FINAL_LR]
               [--final_lr_scale FINAL_LR_SCALE]
               [--init_lr_scale INIT_LR_SCALE] [--max_len MAX_LEN]
               [--max_grad_norm MAX_GRAD_NORM] [--weight_decay WEIGHT_DECAY]
               [--reduction REDUCTION] [--warmup_steps WARMUP_STEPS]
               [--teacher_forcing_step TEACHER_FORCING_STEP]
               [--min_teacher_forcing_ratio MIN_TEACHER_FORCING_RATIO]
               [--seed SEED] [--save_result_every SAVE_RESULT_EVERY]
               [--checkpoint_every CHECKPOINT_EVERY]
               [--print_every PRINT_EVERY] [--resume]
```
  
## Installation
This project recommends Python 3.7 or higher.   
We recommend creating a new virtual environment for this project (using virtual env or conda).  

### Prerequisites
  
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* Pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.   
* Pandas: `pip install pandas` (Refer [here](https://github.com/pandas-dev/pandas) for problem installing Pandas)  
* Matplotlib: `pip install matplotlib` (Refer [here](https://github.com/matplotlib/matplotlib) for problem installing Matplotlib)
* librosa: `conda install -c conda-forge librosa` (Refer [here](https://github.com/librosa/librosa) for problem installing librosa)
* torchaudio: `pip install torchaudio` (Refer [here](https://github.com/pytorch/pytorch) for problem installing torchaudio)
* tqdm: `pip install tqdm` (Refer [here](https://github.com/tqdm/tqdm) for problem installing tqdm)
* sentencepiece: `pip install sentencepiece` (Refer [here](https://github.com/google/sentencepiece) for problem installing sentencepiece)
  
### Install from source
Currently we only support installation from source code using setuptools. Checkout the source code and run the   
following commands:  
```
pip install -e .
```
  
## Get Started
    
### Preparing KsponSpeech Dataset (LibriSpeech also supports)
  
- KsponSpeech : [Check this page](https://github.com/sooftware/KoSpeech/tree/master/dataset/kspon)
- LibriSpeech : [Check this page](https://github.com/sooftware/KoSpeech/tree/master/dataset/libri)
  
### Training KsponSpeech Dataset
```
$ ./run-las.sh
```
  
After properly editing `run-las.sh`, You can train the model by above command. 

### Evaluate for KsponSpeech
```
$ ./eval.sh
```
  
Now you have a model which you can use to predict on new data. We do this by running `greedy search` or `beam search`.  
  
### Inference One Audio with Pre-train Models

* Command
```
$ ./infer-with-pretrain.sh
```
* Output
```
아 뭔 소리야 그건 또
```  
You can get a quick look of pre-trained model's inference, with a sample data.  
  
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
  
### Code Style
We follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
    
### Paper References
  
*Ilya Sutskever et al. [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) arXiv: 1409.3215*  
  
*Dzmitry Bahdanau et al. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) arXiv: 1409.0473*   
  
*Jan Chorowski et al. [Attention Based Models for Speech Recognition](https://arxiv.org/abs/1506.07503) arXiv: 1506.07503*    
  
*Wiliam Chan et al. [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211) arXiv: 1508.01211*   
   
*Dario Amodei et al. [Deep Speech2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595) arXiv: 1512.02595*   
   
*Takaaki Hori et al. [Advances in Joint CTC-Attention based E2E Automatic Speech Recognition with a Deep CNN Encoder and RNN-LM](https://arxiv.org/abs/1706.02737) arXiv: 1706.02737*   
  
*Ashish Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) arXiv: 1706.03762*     
  
*Chung-Cheng Chiu et al. [State-of-the-art Speech Recognition with Sequence-to-Sequence Models](https://arxiv.org/abs/1712.01769) arXiv: 1712.01769*   
  
*Anjuli Kannan et al. [An Analysis Of Incorporating An External LM Into A Sequence-to-Sequence Model](https://arxiv.org/abs/1712.01996) arXiv: 1712.01996*  
  
*Daniel S. Park et al. [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779) arXiv: 1904.08779*     
    
*Rafael Muller et al. [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) arXiv: 1906.02629*   
  
*Daniel S. Park et al. [SpecAugment on large scale datasets](https://arxiv.org/abs/1912.05533) arXiv:1912.05533* 
    
*Jung-Woo Ha et al. [ClovaCall: Korean Goal-Oriented Dialog Speech Corpus for Automatic Speech Recognition of Contact Centers](https://arxiv.org/abs/2004.09367) arXiv: 2004.09367*
    
### Github References
  
*[IBM/Pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)*  
  
*[SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)*  
  
*[kaituoxu/Speech-Transformer](https://github.com/kaituoxu/Speech-Transformer)*  
  
*[OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)*  
  
*[clovaai/ClovaCall](https://github.com/clovaai/ClovaCall)*  
  
*[LiyuanLucasLiu/RAdam](https://github.com/LiyuanLucasLiu/RAdam)*
  
*[AppleHolic/2020 AI Challenge - SpeechRecognition](https://github.com/AppleHolic/2020AIChallengeSpeechRecognition)*
   
### License
This project is licensed under the Apache-2.0 LICENSE - see the [LICENSE.md](https://github.com/sooftware/KoSpeech/blob/master/LICENSE) file for details
  
## Citation
  
A [technical report](https://arxiv.org/abs/2009.03092) on KoSpeech is available. If you use the system for academic work, please cite:
  
```
@ARTICLE{2020kospeech,
  author = {Soohwan Kim, Seyoung Bae, Cheolhwang Won},
  title = "{KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition}",
  journal = {ArXiv e-prints},
  eprint = {2009.03092}
}
```
