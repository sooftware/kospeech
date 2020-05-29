
# **End-to-end Speech Recognition**  
  
### 문자 단위 기반 종단간 한국어 음성인식  
   
[<img src="https://github.com/gentaiscool/end2end-asr-pytorch/raw/master/img/pytorch-logo-dark.png" height=18>](https://pytorch.org/) <img src="https://img.shields.io/badge/License-Apache--2.0-yellow" height=20> [<img src="https://img.shields.io/badge/chat-on%20gitter-4fb99a" height=20>](https://gitter.im/Korean-Speech-Recognition/community)   
  
### [**문서**](https://sooftware.github.io/End-to-end-Speech-Recognition/)   
  
## 소개

- `End-to-end Speech Recognition` 은 [PyTorch](http://pytorch.org)를 통해 구현된 종단간 자동 한국어 음성인식 프로젝트입니다.   
- `e2e`는 LAS 모델, 학습과 추론, 체크포인트에 따른 모델 저장, 파싱까지 여러 확장 가능한 요소로 모듈화 되어있습니다.  
- KAI.Lib은 다양한 [feedback or contribution](https://github.com/sooftware/End-to-end-Speech-Recognition/issues)을 기대하고 있습니다.
  
저희는 **1000시간**의 한국어 발화 데이터 `KsponSpeech` 코퍼스를 사용했습니다. 현재 저희 모델은 **86.98% CRR**을 기록하고 있으며 더욱 높은 인식 기록을 목표로 학습 중에 있습니다. 또한 `Kaldi-zeroth dataset` 테스트 결과 **91.0% CRR**을 기록한 바 있습니다.    
  
##### ( **CRR** : Character Recognition Rate ) 
  
## 피쳐  
  
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
  
## 로드맵
  
<img src="https://user-images.githubusercontent.com/42150335/80630547-5dfc6580-8a8f-11ea-91e8-73fe5e8b9e4b.png" width=450> 
  
종단간 음성 인식은 신경망 기반 음성인식 분야에서 새롭게 부상하는 패러다임으로 여러 장점을 제공합니다. 음향모델, 언어모델, 발음 전사 모델로 구성된 전통적인 하이브리드 음성인식 모델은 각 모델들에 맞는 복잡한 학습을 필요로 했습니다.   
  
예를 들어, 음향 모델의 학습은 다단계 모델 학습 과정과 음성 피쳐 시퀀스와 레이블 시퀀스 사이의 시간적 정렬 과정으로 이루어져 있습니다. 반대로 종단간 음성인식은 훨씬 간단한 학습 파이프라인을 갖춘 단일화된 접근 방식입니다. 이로써 학습 시간, 디코딩 시간을 줄일 수 있으며 자연어 이해와 같은 다운스트림 처리 능력이 결합된 최적화가 가능합니다.  
  
저희는 아래와 같은 논문들을 참조했습니다.  
  
 [「Listen, Attend and Spell」](https://arxiv.org/abs/1508.01211)  
 
[「State-of-the-art Speech Recognition with Sequence-to-Sequence Models」](https://arxiv.org/abs/1712.01769)
   
[「SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition」](https://arxiv.org/abs/1904.08779).   
  
오디오 피쳐에 관해 공부하고 싶은 분들에게는 아래 논문을 추천합니다.  
  
[「Voice Recognition Using MFCC Algirithm」](https://ijirae.com/volumes/vol1/issue10/27.NVEC10086.pdf).  
  
저희 프로젝트는 `seq2seq with Attention Archtecture` 에 기반을 두고 있습니다.  
  
시퀀스 투 시퀀스 구조는 음성인식 분야에서 현재까지도 활발히 연구되는 영역입니다.    
저희 모델 구조는 다음과 같습니다.
  
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
  
`e2e` 모듈 구조는 위와 같이 구성되어 있습니다.   
`e2e` 는 확장성 있는 LAS 모델, 학습기, 평가기, 체크포인트, 데이터 로더 등을 제공합니다.  
  
저희는 [Wiki page](https://github.com/sooftware/End-to-end-Speech-Recognition/wiki)에 프로젝트 진행 상황을 끊임없이 업데이트하고 있습니다.   
  
## 설치
이 프로젝트는 Python 3.7 이상의 버전을 필요로 합니다. 또한 본 프로젝트를 위한 새로운 가상환경을 만드시길 권장하는 바 입니다.  

### 필수조건
  
* Numpy: `pip install numpy` (Numpy 설치 문제 시 [이곳](https://github.com/numpy/numpy) 참조).
* Pandas: `pip install pandas` (Pandas 설치 문제 시 [이곳](https://github.com/pandas-dev/pandas) 참조)  
* librosa: `pip install librosa` (librosa 설치 문제 시 [이곳](https://github.com/librosa/librosa) 참조)
* torchaudio: `pip install torchaudio` (torchaudio 설치 문제 시 [이곳](https://github.com/pytorch/pytorch) 참조)
* tqdm: `pip install tqdm` (tqdm 설치 문제 시 [이곳](https://github.com/tqdm/tqdm) 참조)
* Pytorch: [PyTorch website](http://pytorch.org/) 을 참조하여 본인의 환경에 맞는 설치 진행.   
  
### 소스 파일로부터 설치
현재 저희는 setuptools 를 이용한 소스 코드로부터의 설치를 지원하고 있습니다. 소스 코드를 확인하고 아래 명령어를 실행하여 설치를 진행해주시기 바랍니다.  
```
pip install -r requirements.txt
```
```
python setup.py build
python setup.py install
```
  
## 시작하기
### 1단계: 데이터셋 준비

학습 전 [이곳](https://github.com/sooftware/End-to-end-Speech-Recognition/wiki/Preparation-before-Training) 을 확인해 주시기 바랍니다. 해당 문서는 `KsponSpeech` 에 대한 전처리 내용을 설명하고 있습니다.
위 문서는 한국어로 기술되어 있으며, 빠른 시일 내에 영어 설명을 첨부할 예정입니다.  

### 2단계: `main.py` 실행하기
* 기본 설정  
```
$ ./main.sh
```
* 사용자 설정
```
python ./main.py -dataset_path dataset_path -data_list_path data_list_path \
                 -use_multi_gpu -init_uniform -mode train -batch_size 32 -num_workers 4 \
                 -num_epochs 20 -spec_augment -noise_augment -max_len 151 \
                 -use_cuda -valid_ratio 0.01 -max_grad_norm 400 -rampup_period 1000 \
                 -label_smoothing 0.1 -save_result_every 1000 -print_every 10 -checkpoint_every 5000 \
                 -use_bidirectional -hidden_dim 256 -dropout 0.3 -num_heads 8 -rnn_type gru \
                 -listener_layer_size 5 -speller_layer_size 3 -teacher_forcing_ratio 0.99 \ 
                 -input_reverse -normalize -del_silence -sample_rate 16000 -window_size 20 -stride 10 -n_mels 80 \
                 -feature_extract_by librosa -time_mask_para 50 -freq_mask_para 12 \
                 -time_mask_num 2 -freq_mask_num 2
```
  
위의 명령어에 따라 모델을 학습 가능합니다.  
 `Defaulting setting` 를 통해 기본 설정으로 모델 학습을 진행하실 수 있습니다.   
 또는 `Custom setting` 명령어를 통해 하이퍼 파라미터를 설정하실 수 있습니다.


### 3단계: `infer.py` 실행하기
* 기본 설정
```
$ ./infer.sh
```
* 사용자 설정
```
python ./infer.py -dataset_path dataset_path -data_list_path data_list_path \
                  -mode infer -use_multi_gpu -use_cuda -batch_size 32 -num_workers 4 \
                  -use_beam_search -k 5 -print_every 100 \
                  -sample_rate 16000 --window_size 20 --stride 10 --n_mels 80 -feature_extract_by librosa \
                  -normalize -del_silence -input_reverse 
```
학습이 끝났다면 해당 모델을 통해 새로운 데이터에 대한 추론을 진행할 수 있습니다. 저희는 추론 과정에서 탐욕 알고리즘과 빔탐색 알고리즘 모두를 지원하고 있습니다.  
학습과 마찬가지로 `Default setting` 과 `Custom setting` 중 하나를 선택하실 수 있습니다.

### **체크포인트**

체크포인트는 아래와 같은 구조로 저장됩니다.  
```
save_dir
+-- checkpoints
|  +-- YYYY_mm_dd_HH_MM_SS
   |  +-- trainer_states.pt
   |  +-- model.pt
```
체크포인트로부터 학습을 재개하거나 해당 포인트 모델로 추론을 진행할 수 있습니다.
  
### **외부 언어모델과 결합된 모델에서의 성능 평가**
외부 언어모델과 병합된 모델의 성능 평가에 관심 있으시다면 [이곳](https://github.com/sooftware/char-rnnlm)을 확인해주시기 바랍니다.
  
## 트러블 슈팅과 기여

어떠한 질문이나 버그 리포트, 피쳐 요청이 있으시면 깃허브의 [open an issue](https://github.com/sooftware/End-to-end-Speech-Recognition/issues) 로 등록해주시면 감사하겠습니다.   
또한 즉각적인 피드백이나 대화를 원하시는 분들은 저희 [지터](https://gitter.im/Korean-Speech-Recognition/community) 또는 sh951011@gmail.com 로 연락주시면 감사하겠습니다.
  
저희는 모델에 대한 여러분의 기여나 피드백을 기대하고 있습니다. 부담스럽게 생각하지 마시고 개선사항이라고 생각되는 부분을 말씀해주시면 감사하겠습니다. 부디 여러분들께서 보내주신 주요 컨트리뷰트나 여러 피쳐 이슈들에 관해 저희와 많은 대화 나눌 수 있으면 좋겠습니다.  

### 코드 스타일
저희는 [PEP-8](https://www.python.org/dev/peps/pep-0008/) 파이썬 코딩 표준을 따랐습니다.  
    
### 참조   
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
   
### 인용
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
