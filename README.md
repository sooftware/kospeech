# Korean-ASR
based on https://github.com/sh951011/Naver-Hackathon-2019-Speech-Team_Kai.Lib  
Korean Speech Recognition Using PyTorch.  
This Project is currently in progress.  
[Demonstration Video](https://www.youtube.com/watch?v=dHJnCqo2gaU)   

## Team Member  
* [김수환](https://github.com/sh951011) KWU. elcomm.  
* [배세영](https://github.com/triplet02) KWU. elcomm.  
* [원철황](https://github.com/wch18735) KWU. elcomm.  

## Model
![model](https://postfiles.pstatic.net/MjAxOTExMjdfMjM1/MDAxNTc0ODIxOTY1NDI3.KIFNl1lvjCnYHXCzkEssJLJxXGs-m6zKvSfaurZncasg.PnUqcLztGAueEecp5DoOWf61AExatLIu4ZZoEeS1Ia4g.PNG.sooftware/image.png?type=w773)  
* Model Architecture : Seq2seq with Attention  
```python
Seq2seq(
  (encoder): EncoderRNN(
    (input_dropout): Dropout(p=0.3, inplace=False)
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
    (rnn): GRU(5120, 256, num_layers=6, batch_first=True, dropout=0.3, bidirectional=True)
  )
  (decoder): DecoderRNN(
    (input_dropout): Dropout(p=0.3, inplace=False)
    (rnn): GRU(512, 512, num_layers=6, batch_first=True, dropout=0.3)
    (embedding): Embedding(800, 512)
    (out): Linear(in_features=512, out_features=800, bias=True)
    (attention): Attention(
      (linear_out): Linear(in_features=1024, out_features=512, bias=True)
    )
  )
)
```  
* Model based on IBM PyTorch-seq2seq  
## Data
A.I Hub에서 제공한 1,000시간 데이터 사용 
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
### Data Preprocessing
* b/, n/, u/ .. 등의 잡음 레이블 삭제 
```
"b/ 아/ 모+ 몬 소리야 (70%)/(칠 십 퍼센트) 확률이라니 n/" => "아/ 모+ 몬 소리야 (70%)/(칠 십 퍼센트) 확률이라니"
```
* 제공된 (철자전사)/(발음전사) 중 발음전사 사용  
```
"아/ 모+ 몬 소리야 (70%)/(칠 십 퍼센트) 확률이라니" => "아/ 모+ 몬 소리야 칠 십 퍼센트 확률이라니"
```
* 간투어 표현 등을 위해 사용된 '/', '*', '+' 등의 레이블 삭제
```
"아/ 모+ 몬 소리야 칠 십 퍼센트 확률이라니" => "아 모 몬 소리야 칠 십 퍼센트 확률이라니"
```

