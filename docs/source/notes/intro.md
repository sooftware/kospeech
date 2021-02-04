<p  align="center"><img src="https://user-images.githubusercontent.com/42150335/105211661-e1b33080-5b8f-11eb-9956-184d60ccc55a.png" height=100>

<p  align="center">An Apache 2.0 ASR research library, built on PyTorch, for developing end-to-end speech recognition models.

***

<p  align="center"> 
     <a href="https://www.codefactor.io/repository/github/sooftware/kospeech">
          <img src="https://www.codefactor.io/repository/github/sooftware/kospeech/badge"> 
     </a>
     <a href="https://github.com/sooftware/KoSpeech/blob/latest/LICENSE">
          <img src="http://img.shields.io/badge/license-Apache--2.0-informational"> 
     </a>
     <a href="https://github.com/pytorch/pytorch">
          <img src="http://img.shields.io/badge/framework-PyTorch-informational"> 
     </a>
     <a href="https://www.python.org/dev/peps/pep-0008/">
          <img src="http://img.shields.io/badge/codestyle-PEP--8-informational"> 
     </a>
     <a href="https://sooftware.github.io/KoSpeech/">
          <img src="http://img.shields.io/badge/docs-passing-success">
     </a>
     <a href="https://gitter.im/Korean-Speech-Recognition/community">
          <img src="https://img.shields.io/gitter/room/sooftware/KoSpeech">
     </a>
     
   
### What's New
- Febuary 2021: Add RNN-Transducer model
- January 2021: Release v1.3
- January 2021: Add Conformer model
- January 2021: Add Jasper model
- January 2021: Add Joint CTC-Attention Transformer model
- January 2021: Add Speech Transformer model
- January 2021: Apply [Hydra: framework for elegantly configuring complex applications](https://github.com/facebookresearch/hydra)
  
### Note
  
- Currently, beam search may not work properly.  
- Subword and Grapheme unit currently not tested.
  
### ***[KoSpeech:  Open-Source Toolkit for End-to-End Korean Speech Recognition \[Paper\]](https://www.sciencedirect.com/science/article/pii/S2665963821000026)***
  
***KoSpeech***, an open-source software, is modular and extensible end-to-end Korean automatic speech recognition (ASR) toolkit based on the deep learning library PyTorch. Several automatic speech recognition open-source toolkits have been released, but all of them deal with non-Korean languages, such as English (e.g. ESPnet, Espresso). Although AI Hub opened 1,000 hours of Korean speech corpus known as KsponSpeech, there is no established preprocessing method and baseline model to compare model performances. Therefore, we propose preprocessing methods for KsponSpeech corpus and a several models (Deep Speech 2, LAS, Transformer, Jasper, Conformer). By KoSpeech, we hope this could be a guideline for those who research Korean speech recognition.  
  
### Supported Models
  
|Acoustic Model|Notes|Citation|  
|--------------|------|--------:|  
|Deep Speech 2|2D-invariant convolution & RNN & CTC|[Dario Amodei et al., 2015](https://arxiv.org/abs/1512.02595)|   
|Listen Attend Spell (LAS)|Attention based RNN sequence to sequence|[William Chan et al., 2016](https://arxiv.org/abs/1508.01211)|  
|Joint CTC-Attention LAS|Joint CTC-Attention LAS|[Suyoun Kim et al., 2017](https://arxiv.org/pdf/1609.06773.pdf)|  
|RNN-Transducer|RNN Transducer|[Ales Graves. 2012](https://arxiv.org/pdf/1211.3711.pdf)|  
|Speech Transformer|Convolutional extractor & transformer|[Linhao Dong et al., 2018](https://ieeexplore.ieee.org/document/8462506)|  
|Jasper|Fully convolutional & dense residual connection & CTC|[Jason Li et al., 2019](https://arxiv.org/pdf/1904.03288.pdf)|  
|Conformer|Convolution-augmented-Transformer|[Anmol Gulati et al., 2020](https://arxiv.org/abs/2005.08100)|  
  
- **Note**  
It is based on the above papers, but there may be other parts of the model implementation.
  
### Pre-processed Transcripts
  
|Dataset    |Authentication|Output-Unit|Transcript|  
|-----------|--------------|-----------|:--------:|  
|KsponSpeech|*Required*    |Character  |[download](https://drive.google.com/file/d/1ivJJiUhKhj0FcOntA2hcYWXR00XzRI_f/view?usp=sharing)|  
|KsponSpeech|*Required*    |Subword    |[download](https://drive.google.com/file/d/1awhfTpqAaDs7K5R9npvFoqeMYWiUtGtq/view?usp=sharing)|  
|KsponSpeech|*Required*    |Grapheme   |[download](https://drive.google.com/file/d/1awhfTpqAaDs7K5R9npvFoqeMYWiUtGtq/view?usp=sharing)|   
|LibriSpeech|*Unrequired*  |Subword    |[download](https://drive.google.com/file/d/1kTeQ93FU7B6bzIXlQLV6du5g-7LEukGH/view?usp=sharing)|  
   
KsponSpeech needs permission from [AI Hub](https://aihub.or.kr/). Please send e-mail including the approved screenshot to sh951011@gmail.com. It may be slow to reply, so it is recommended to execute [preprocessing code](https://github.com/sooftware/KoSpeech/tree/master/dataset/kspon).
  
## Introduction
  
End-to-end (E2E) automatic speech recognition (ASR) is an emerging paradigm in the field of neural network-based speech recognition that offers multiple benefits. Traditional “hybrid” ASR systems, which are comprised of an acoustic model, language model, and pronunciation model, require separate training of these components, each of which can be complex.   
  
For example, training of an acoustic model is a multi-stage process of model training and time alignment between the speech acoustic feature sequence and output label sequence. In contrast, E2E ASR is a single integrated approach with a much simpler training pipeline with models that operate at low audio frame rates. This reduces the training time, decoding time, and allows joint optimization with downstream processing such as natural language understanding.  
  
## Roadmap
  
So far, serveral models are implemented: *Deep Speech 2, Listen Attend and Spell (LAS), RNN-Transducer, Speech Transformer, Jasper, Conformer*.
  
- *Deep Speech 2*  
  
Deep Speech 2 showed faster and more accurate performance on ASR tasks with Connectionist Temporal Classification (CTC) loss. This model has been highlighted for significantly increasing performance compared to the previous end- to-end models.

  
- *Listen, Attend and Spell (LAS)*
   
We follow the architecture previously proposed in the "Listen, Attend and Spell", but some modifications were added to improve performance. We provide four different attention mechanisms, `scaled dot-product attention`, `additive attention`, `location aware attention`, `multi-head attention`. Attention mechanisms much affect the performance of models. 
  
- *RNN-Transducer*
  
RNN-Transducer are a form of sequence-to-sequence models that do not employ attention mechanisms. Unlike most sequence-to-sequence models, which typically need to process the entire input sequence (the waveform in our case) to produce an output (the sentence), the RNN-T continuously processes input samples and streams output symbols, a property that is welcome for speech dictation. In our implementation, the output symbols are the characters of the alphabet.
  
- *Speech Transformer*  
  
Transformer is a powerful architecture in the Natural Language Processing (NLP) field. This architecture also showed good performance at ASR tasks. In addition, as the research of this model continues in the natural language processing field, this model has high potential for further development.
  
- *Joint CTC-Attention*
  
With the proposed architecture to take advantage of both the CTC-based model and the attention-based model. It is a structure that makes it robust by adding CTC to the encoder. Joint CTC-Attention can be trained in combination with LAS and Speech Transformer.  
  
- *Jasper*  
  
Jasper (Just Another SPEech Recognizer) is a end-to-end convolutional neural acoustic model. Jasper showed powerful performance with only CNN → BatchNorm → ReLU → Dropout block and residential connection.  
  
- *Conformer*
  
Conformer combine convolution neural networks and transformers to model both local and global dependencies of an audio sequence in a parameter-efficient way. Conformer significantly outperforms the previous Transformer and CNN based models achieving state-of-the-art accuracies.
  
## Installation
This project recommends Python 3.7 or higher.   
We recommend creating a new virtual environment for this project (using virtual env or conda).  

### Prerequisites
  
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* Pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.   
* Pandas: `pip install pandas` (Refer [here](https://github.com/pandas-dev/pandas) for problem installing Pandas)  
* Matplotlib: `pip install matplotlib` (Refer [here](https://github.com/matplotlib/matplotlib) for problem installing Matplotlib)
* librosa: `conda install -c conda-forge librosa` (Refer [here](https://github.com/librosa/librosa) for problem installing librosa)
* torchaudio: `pip install torchaudio==0.6.0` (Refer [here](https://github.com/pytorch/pytorch) for problem installing torchaudio)
* tqdm: `pip install tqdm` (Refer [here](https://github.com/tqdm/tqdm) for problem installing tqdm)
* sentencepiece: `pip install sentencepiece` (Refer [here](https://github.com/google/sentencepiece) for problem installing sentencepiece)
* warp-rnnt: `pip install warp_rnnt` (Refer [here](https://github.com/1ytic/warp-rnnt)) for problem installing warp-rnnt)
* hydra: `pip install hydra-core --upgrade` (Refer [here](https://github.com/facebookresearch/hydra) for problem installing hydra)
  
### Install from source
Currently we only support installation from source code using setuptools. Checkout the source code and run the   
following commands:  
```
pip install -e .
```
  
## Get Started
  
We use [Hydra](https://github.com/facebookresearch/hydra) to control all the training configurations. If you are not familiar with Hydra we recommend visiting the [Hydra website](https://hydra.cc/). Generally, Hydra is an open-source framework that simplifies the development of research applications by providing the ability to create a hierarchical configuration dynamically.
  
### Preparing KsponSpeech Dataset (LibriSpeech also supports)
  
Download from [here](https://github.com/sooftware/KoSpeech#pre-processed-transcripts) or refer to the following to preprocess.
  
- KsponSpeech : [Check this page](https://github.com/sooftware/KoSpeech/tree/master/dataset/kspon)
- LibriSpeech : [Check this page](https://github.com/sooftware/KoSpeech/tree/master/dataset/libri)
  
### Training KsponSpeech Dataset
  
You can choose from several models and training options. There are many other training options, so look carefully and execute the following command:  
  
- **Deep Speech 2** Training
```
python ./bin/main.py model=ds2 train=ds2_train train.dataset_path=$DATASET_PATH
```
  
- **Listen, Attend and Spell** Training
```
python ./bin/main.py model=las train=las_train train.dataset_path=$DATASET_PATH
```
  
- **Joint CTC-Attention Listen, Attend and Spell** Training
```
python ./bin/main.py model=joint-ctc-attention-las train=las_train train.dataset_path=$DATASET_PATH
```
  
- **RNN Transducer** Training
```
python ./bin/main.py model=rnnt train=rnnt_train train.dataset_path=$DATASET_PATH
```
  
- **Speech Transformer** Training
```
python ./bin/main.py model=transformer train=transformer_train train.dataset_path=$DATASET_PATH
```
  
- **Joint CTC-Attention Speech Transformer** Training
```
python ./bin/main.py model=joint-ctc-attention-transformer train=transformer_train train.dataset_path=$DATASET_PATH
```
  
- **Jasper** Training
```
python ./bin/main.py model=jasper train=jasper_train train.dataset_path=$DATASET_PATH
```
  
- **Conformer** Training
```
python ./bin/main.py model=conformer-large train=conformer_large_train train.dataset_path=$DATASET_PATH
```
You can train with `conformer-medium`, `conformer-small` model.
  
  
### Evaluate for KsponSpeech
```
python ./bin/eval.py eval.dataset_path=$DATASET_PATH eval.transcripts_path=$TRANSCRIPTS_PATH eval.model_path=$MODEL_PATH
```
  
Now you have a model which you can use to predict on new data. We do this by running `greedy search` or `beam search`.  
  
### Inference One Audio with Pre-train Models

* Command
```
$ python3 ./bin/inference.py --model_path $MODEL_PATH --audio_path $AUDIO_PATH --device $DEVICE
```
* Output
```
음성인식 결과 문장이 나옵니다
```  
You can get a quick look of pre-trained model's inference, with a audio.  
  
### Checkpoints   
Checkpoints are organized by experiments and timestamps as shown in the following file structure.  
```
outputs
+-- YYYY_mm_dd
|  +-- HH_MM_SS
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
  
*Daniel S. Park et al. [SpecAugment on large scale datasets](https://arxiv.org/abs/1912.05533) arXiv: 1912.05533* 
    
*Jung-Woo Ha et al. [ClovaCall: Korean Goal-Oriented Dialog Speech Corpus for Automatic Speech Recognition of Contact Centers](https://arxiv.org/abs/2004.09367) arXiv: 2004.09367*  
  
*Jason Li et al. [Jasper: An End-to-End Convolutional Neural Acoustic Model](https://arxiv.org/pdf/1904.03288.pdf) arXiv: 1902.03288* 
    
*Anmol Gulati et al. [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) arXiv: 2005.08100*
    
### Github References
  
*[IBM/Pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)*  
  
*[SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)*  
  
*[kaituoxu/Speech-Transformer](https://github.com/kaituoxu/Speech-Transformer)*  
  
*[OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)*  
  
*[clovaai/ClovaCall](https://github.com/clovaai/ClovaCall)*  
  
*[LiyuanLucasLiu/RAdam](https://github.com/LiyuanLucasLiu/RAdam)*
  
*[AppleHolic/2020 AI Challenge - SpeechRecognition](https://github.com/AppleHolic/2020AIChallengeSpeechRecognition)*  
  
*[NVIDIA/DeepLearningExample](https://github.com/NVIDIA/DeepLearningExamples)*
  
*[espnet/espnet](https://github.com/espnet/espnet)*
  
*[sooftware/conformer](https://github.com/sooftware/conformer)*
  
*[sooftware/jasper](https://github.com/sooftware/jasper)*
  
*[sooftware/speech-transformer](https://github.com/sooftware/speech-transformer)*
  
*[sooftware/seq2seq](https://github.com/sooftware/seq2seq)*
   
### License
This project is licensed under the Apache-2.0 LICENSE - see the [LICENSE.md](https://github.com/sooftware/KoSpeech/blob/master/LICENSE) file for details
  
## Citation
  
A [paper](https://www.sciencedirect.com/science/article/pii/S2665963821000026) on KoSpeech is available. If you use the system for academic work, please cite:
  
```
@ARTICLE{2021-kospeech,
  author    = {Kim, Soohwan and Bae, Seyoung and Won, Cheolhwang},
  title     = {KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition},
  url       = {https://www.sciencedirect.com/science/article/pii/S2665963821000026},
  month     = {February},
  year      = {2021},
  publisher = {ELSEVIER},
  journal   = {SIMPAC},
  pages     = {Volume 7, 100054}
}
```
A [technical report](https://arxiv.org/abs/2009.03092) on KoSpeech in available.  
  
```
@ARTICLE{2020-kospeech,
  author    = {Kim, Soohwan and Bae, Seyoung and Won, Cheolhwang},
  title     = {KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition},
  month     = {September},
  year      = {2020},
  url       = {https://arxiv.org/abs/2009.03092},
  journal   = {ArXiv e-prints},
  eprint    = {2009.03092}
}
```
