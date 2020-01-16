# Korean-Speech-Recognition
Further Works from https://github.com/sh951011/Naver-Hackathon-2019-Speech-Team_Kai.Lib  
Korean Speech Recognition Using PyTorch. (Korean-ASR)  
This Project is currently in progress.  
[Demonstration Video](https://www.youtube.com/watch?v=dHJnCqo2gaU)   

## Team Member  
* [김수환](https://github.com/sh951011) KW University. elcomm. senior
* [배세영](https://github.com/triplet02) KW University. elcomm. senior  
* [원철황](https://github.com/wch18735) KW University. elcomm. senior

## Model
![model](https://postfiles.pstatic.net/MjAxOTExMjdfMjM1/MDAxNTc0ODIxOTY1NDI3.KIFNl1lvjCnYHXCzkEssJLJxXGs-m6zKvSfaurZncasg.PnUqcLztGAueEecp5DoOWf61AExatLIu4ZZoEeS1Ia4g.PNG.sooftware/image.png?type=w773)  
* Model Architecture : Seq2seq with Attention  
```python
Seq2seq(
  (encoder): EncoderRNN(
    (input_dropout): Dropout(p=0.5, inplace=False)
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
      (12): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (13): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (14): Hardtanh(min_val=0, max_val=20, inplace=True)
      (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (rnn): GRU(5120, 256, num_layers=5, batch_first=True, dropout=0.5, bidirectional=True)
  )
  (decoder): DecoderRNN(
    (input_dropout): Dropout(p=0.5, inplace=False)
    (rnn): GRU(512, 512, num_layers=3, batch_first=True, dropout=0.5)
    (embedding): Embedding(2040, 512)
    (out): Linear(in_features=512, out_features=2040, bias=True)
    (attention): Attention(
      (linear_out): Linear(in_features=1024, out_features=512, bias=True)
    )
  )
)
```  
* Model based on IBM PyTorch-seq2seq  
## Data
A.I Hub에서 제공한 1,000시간의 한국어 음성데이터 사용 
### Data Format
* 음성 데이터 : 16k sampling PCM  
* 정답 스크립트 : Character level dictionary를 통해서 인덱스로 변환된 정답
### Dataset folder structure
```
* DATASET-ROOT-FOLDER
|--KaiSpeech
   +--KaiSpeech_000001.pcm, KaiSpeech_000002.pcm, ... KaiSpeech_622245.pcm
   +--KaiSpeech_000001.txt, KaiSpeech_000002.txt, ... KaiSpeech_622245.txt
   +--KaiSpeech_label_000001.pcm, KaiSpeech_label_000002.pcm, ... KaiSpeech_label_622245.pcm
```
* KaiSpeech_FileNum.txt
```
아 모 몬 소리야 칠 십 퍼센트 확률이라니
```
* KaiSpeech_lable_FileNum.txt
```
5 0 105 0 729 0 172 31 25 0 318 0 119 0 489 551 156 0 314 746 3 32 20
```
* train_list.csv    
전체 데이터셋의 80%에 해당하는 학습용 데이터 리스트  
전체 데이터셋에서 등장한 2,340개의 문자 중 1번 만 등장한 문자들은 포함된 데이터를 제외한 리스트    
  
| pcm-filename| txt-filename|   
| :-------------------| :--------------------------|     
| KaiSpeech_078903.pcm | KaiSpeech_label_078903.txt  |  
| KaiSpeech_449461.pcm | KaiSpeech_label_449461.txt  |  
| KaiSpeech_178531.pcm | KaiSpeech_label_178531.txt  |  
| KaiSpeech_374874.pcm | KaiSpeech_label_374874.txt  |  
| KaiSpeech_039018.pcm | KaiSpeech_label_039018.txt  |  
  
* test_list.csv   
전체 데이터셋의 20%에 해당하는 테스트용  리스트   
전체 데이터셋에서 등장한 2,340개의 문자 중 1번 만 등장한 문자들이 포함된 데이터 포함   
  
| pcm-filaname| txt-filename|    
| :-------------------| :--------------------------|     
| KaiSpeech_126887.pcm | KaiSpeech_label_126887.txt  |  
| KaiSpeech_067340.pcm | KaiSpeech_label_067340.txt  |    
| KaiSpeech_350293.pcm | KaiSpeech_label_350293.txt  |   
| KaiSpeech_212197.pcm | KaiSpeech_label_212197.txt  |   
| KaiSpeech_489840.pcm | KaiSpeech_label_489840.txt  |   
  
### Data Preprocessing
* Raw Data  
```
"b/ 아/ 모+ 몬 소리야 (70%)/(칠 십 퍼센트) 확률이라니 n/"  
```
* b/, n/, / .. 등의 잡음 레이블 삭제 
```
"아/ 모+ 몬 소리야 (70%)/(칠 십 퍼센트) 확률이라니"
```
* 제공된 (철자전사)/(발음전사) 중 발음전사 사용  
```
"아/ 모+ 몬 소리야 칠 십 퍼센트 확률이라니"
```
* 간투어 표현 등을 위해 사용된 '/', '*', '+' 등의 레이블 삭제
```
"아 모 몬 소리야 칠 십 퍼센트 확률이라니"
```
## Hyperparameters  
| Hyperparameter  |Help| Use|              
| ----------      |---|----------|    
| use_bidirectional| if True, becomes a bidirectional encoder|True|  
| use_attention    | flag indication whether to use attention mechanism or not|True |   
|input_reverse|flag indication whether to reverse input feature or not|True|   
|use_augment| flag indication whether to use spec-augmentation or not|True|  
|augment_ratio|ratio of spec-augmentation applied data|0.3|   
|encoder_layer_size|num of encoder`s RNN cell|5|  
| decoder_layer_size|num of decoder`s RNN cell| 3|  
| hidden_size| size of hidden state of RNN|256|
| batch_size | mini-batch size|8|
| dropout          | dropout probability|0.5  |
| teacher_forcing  | The probability that teacher forcing will be used|0.99|
| lr               | learning rate|1e-4        |
| max_epochs       | max epoch|30          |   
  
## Feature  
* MFCC (Mel-Frequency-Cepstral-Coefficients)  
  
| Parameter| Use|    
| :-----| :----|     
|Frame length|30ms|
|Stride|7.5ms|
| N_FFT | 480  |   
| hop length | 120  |
| n_mfcc | 33  |  
|window|hamming|  
  

* code   
```python
def get_librosa_mfcc(filepath = None, n_mfcc = 33, del_silence = True, input_reverse = True, format='pcm'):
    if format == 'pcm':
        pcm = np.memmap(filepath, dtype='h', mode='r')
        sig = np.array([float(x) for x in pcm])
    elif format == 'wav':
        sig, sr = librosa.core.load(filepath, sr=16000)
    else: logger.info("Invalid file format!!")

    if del_silence:
        non_silence_indices = librosa.effects.split(sig, top_db=30)
        sig = np.concatenate([sig[start:end] for start, end in non_silence_indices])
    feat = librosa.feature.mfcc(y=sig,sr=16000, hop_length=120, n_mfcc=n_mfcc, n_fft=480, window='hamming')
    if input_reverse: feat = feat[:,::-1]

    return torch.FloatTensor( np.ascontiguousarray( np.swapaxes(feat, 0, 1) ) )
```
  
## Score
```
CRR = (1.0 - CER) * 100.0
```
* CRR : Character Recognition Rate
* CER : Character Error Rate based on Edit Distance
![crr](https://github.com/AjouJuneK/NAVER_speech_hackathon_2019/raw/master/docs/edit_distance.png)

## Reference
* [[1] IBM pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)
* [[2] A.I Hub 한국어 음성 데이터셋](http://www.aihub.or.kr/aidata/105)
* [[3]「Voice Recognition Using MFCC Algorithm」  Paper](https://s3.amazonaws.com/academia.edu.documents/36789621/27.NVEC10086.pdf?response-content-disposition=inline%3B%20filename%3DIJIRAE_Voice_Recognition_Using_MFCC_Algo.pdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWOWYYGZ2Y53UL3A%2F20200115%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200115T053516Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=109747040e3e0f7ed358a3131fe2190d1fab0ba80985ffe08f819a2b5da4e36a)
* [[4]「A Simple Data Augmentation Method for Automatic Speech Recognition」  Paper](https://arxiv.org/abs/1904.08779)  
* [[5] PyTorch Spec-Augmentation](https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_pytorch.py)
* [[6]「Sequence to sequence learning with neural networks」  Paper](https://arxiv.org/abs/1409.3215)
   
## Requirements  
Install Levenshtein  
```
pip install python-levenshtein 
```
Install PyTorch
```
pip install torch
```  
Install librosa   
``` 
pip install librosa
```
Install tqdm  
```
pip install tqdm
```
## License
```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
