## Team Member  
[![KimSooHwan](https://postfiles.pstatic.net/MjAyMDAyMjBfMTIz/MDAxNTgyMTgzMTg0NjQ0.WkBpWhKQ8YT8Ct9BHrdD44Yn6l-1f-lCNjdIE8uU5e8g.UUvRfvxb1cfn6Ml1ZQzE_4kv6QYsvgBpuiSiTWSEZMIg.PNG.sooftware/image.png?type=w773)](https://github.com/sh951011)   [![BaeSeYoung](https://postfiles.pstatic.net/MjAyMDAyMjBfMjgx/MDAxNTgyMTgzMjA5MDM1.bUVfaKWb3MZ4eJVFawmTHVdQs1aohO4CUW7qHTC38okg.NGBQL8cunnwMnh3Pt8CWkRWlMqAHVOkNMJCowKd1wAAg.PNG.sooftware/image.png?type=w773)](https://github.com/triplet02)   [![WonCheolHwang](https://postfiles.pstatic.net/MjAyMDAyMjBfMjIg/MDAxNTgyMTgzMjIzMzcx.knqFUOpdhk1l_GLZWvz0zelNf-QJtA_yjaoYuKBJpN8g.U5EhVv_elOcufKYTOaaJof1ZqjHOaYlDHAyCBcsXjdAg.PNG.sooftware/image.png?type=w773)](https://github.com/wch18735)

## Model
<img src="https://postfiles.pstatic.net/MjAyMDAyMjVfODIg/MDAxNTgyNjE5NzE3NjU5.51D-0F_nvBCZQ89XpgaycjPsX92z_lZK-vCQIHXfOmkg.kK0ILmnHM-LXMRxjTB5o1vJjKnhI4cw73me3LpvRkxUg.PNG.sooftware/LAS.png?type=w773" width="800"> 
  
### Listen, Attend and Spell Architecture 
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
      (13): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (14): Hardtanh(min_val=0, max_val=20, inplace=True)
      (15): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (bottom_rnn): GRU(4096, 256, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
    (middle_rnn): PyramidalRNN(
      (rnn): GRU(1024, 256, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
    )
    (top_rnn): PyramidalRNN(
      (rnn): GRU(1024, 256, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
    )
  )
  (speller): Speller(
    (rnn): GRU(512, 512, num_layers=3, batch_first=True, dropout=0.5)
    (embedding): Embedding(2040, 512)
    (input_dropout): Dropout(p=0.5, inplace=False)
    (attention): Attention(
      (attention): SelfAttention(
        (W): Linear(in_features=1024, out_features=512, bias=True)
      )
    )
    (out): Linear(in_features=512, out_features=2040, bias=True)
  )
)
```  
  
* Reference
  + 「Listen, Attend and Spell」 Chan et al. 2015
  + 「Attention-Based Models for Speech Recognition」 Chorowski et al. 2015
  + 「A Structured Self-attentive Sentence Embedding」 Zhouhan Lin eo al. 2017
  +  https://github.com/IBM/pytorch-seq2seq
  
## Hyperparameters  
| Hyperparameter  |Help| Default|              
| ----------      |---|:----------:|    
| use_bidirectional| if True, becomes a bidirectional encoder|True|  
| use_attention    | flag indication whether to use attention mechanism or not|True |   
| score_function    |which attention to use|self |  
| use_label_smoothing    | flag indication whether to use label smoothing or not|True |   
|input_reverse|flag indication whether to reverse input feature or not|True|   
|use_augment| flag indication whether to use spec-augmentation or not|True|  
|use_pyramidal| flag indication whether to use pLSTM or not|True|  
|augment_ratio|ratio of spec-augmentation applied data|-|   
|listener_layer_size|number of listener`s RNN layer|6|  
| speller_layer_size|number of speller`s RNN layer| 3|  
| hidden_size| size of hidden state of RNN|256|
| batch_size | mini-batch size|12|
| dropout          | dropout probability|0.5  |
| teacher_forcing  | The probability that teacher forcing will be used|0.99|
| lr               | learning rate|[Multi-Step](https://github.com/sh951011/Korean-Speech-Recognition/blob/master/docs/hparams/multi-step-lr.md)        |
| max_epochs       | max epoch|-          |   
   

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
   +--KaiSpeech_label_000001.txt, KaiSpeech_label_000002.txt, ... KaiSpeech_label_622245.txt
```  
  
* KaiSpeech_FileNum.pcm  
![signal](https://postfiles.pstatic.net/MjAyMDAxMjJfMTYx/MDAxNTc5NjcyNzMyMTkz.Kw1WWrvvv9qLEf-pa0QYOcKYL3GOqXxahw_6sBsjqLgg.nkysalfeHToY9_FbVgxVcOM_Q5_RYlbpfFrAdFsdev4g.PNG.sooftware/audio-signal.png?type=w773)
  
* KaiSpeech_FileNum.txt 
```
아 모 몬 소리야 칠 십 퍼센트 확률이라니
```
* KaiSpeech_lable_FileNum.txt
```
5 0 105 0 729 0 172 31 25 0 318 0 119 0 489 551 156 0 314 746 3 32 20
```
* train_list.csv    
학습용 데이터 리스트 - **980h**    
  
| pcm-filename| txt-filename|   
| :-------------------| :--------------------------|     
| KaiSpeech_078903.pcm | KaiSpeech_label_078903.txt  |  
| KaiSpeech_449461.pcm | KaiSpeech_label_449461.txt  |  
| KaiSpeech_178531.pcm | KaiSpeech_label_178531.txt  |  
| KaiSpeech_374874.pcm | KaiSpeech_label_374874.txt  |  
| KaiSpeech_039018.pcm | KaiSpeech_label_039018.txt  |  
  
* test_list.csv   
테스트용 데이터 리스트  - **20h**     
  
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
   
## Character label  
전체 데이터셋에서 등장한 2,340개의 문자 중 1번 만 등장한 문자들은 제외한 2,040개의 문자 레이블  
* kai_labels.csv  
  
|id|char|freq|  
|:--:|:----:|:----:|   
|0| |5774462|   
|1|.|640924|   
|2|그|556373|   
|3|이|509291|   
|.|.|.|  
|.|.|.|     
|2037|\<s\>|0|   
|2038|\</s\>|0|   
|2039|\_|0|    
  
## Feature  
* Log Mel-Spectrogram  
  
| Parameter| Use|    
| -----|:-----:|     
|Frame length|25ms|
|Stride|10ms|
| N_FFT | 400  |   
| hop length | 160  |
| n_mels | 128  |  
|window|hamming|  
  

* code   
```python
def get_librosa_melspectrogram(filepath, n_mels=128, del_silence=False, input_reverse=True, mel_type='log_mel', format='pcm'):
    if format == 'pcm':
        pcm = np.memmap(filepath, dtype='h', mode='r')
        signal = np.array([float(x) for x in pcm])
    elif format == 'wav':
        signal, _ = librosa.core.load(filepath, sr=16000)
    else:
        raise ValueError("Invalid format !!")

    if del_silence:
        non_silence_indices = librosa.effects.split(y=signal, top_db=30)
        signal = np.concatenate([signal[start:end] for start, end in non_silence_indices])

    feat = librosa.feature.melspectrogram(signal, sr=16000, n_mels=128, n_fft=400, hop_length=160, window='hamming')

    if mel_type == 'log_mel':
        feat = librosa.amplitude_to_db(feat, ref=np.max)
    if input_reverse:
        feat = feat[:,::-1]

    return torch.FloatTensor( np.ascontiguousarray( np.swapaxes(feat, 0, 1) ) )
```
   
* Reference
  + 「 Voice Recognition Using MFCC Algorithm」 Chakraborty et al. 2014
  + https://github.com/librosa/librosa
    

## SpecAugmentation
Applying Frequency Masking & Time Masking except Time Warping
* code  
```python
def spec_augment(feat, T=70, F=20, time_mask_num=2, freq_mask_num=2):
    feat_size = feat.size(1)
    seq_len = feat.size(0)

    # time mask
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=T)
        t = int(t)
        t0 = random.randint(0, seq_len - t)
        feat[t0 : t0 + t, :] = 0

    # freq mask
    for _ in range(freq_mask_num):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        f0 = random.randint(0, feat_size - f)
        feat[:, f0 : f0 + f] = 0

    return feat
```    
  
* Reference
  + 「SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition」 Google Brain Team.  
  + https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_pytorch.py
  
## Score
```
CRR = (1.0 - CER) * 100.0
```
* CRR : Character Recognition Rate
* CER : Character Error Rate based on Edit Distance
![crr](https://github.com/AjouJuneK/NAVER_speech_hackathon_2019/raw/master/docs/edit_distance.png)   
  
* Reference
  + https://en.wikipedia.org/wiki/Levenshtein_distance
  
## Training List     
  
[AI Hub Dataset #2](https://github.com/sh951011/Korean-Speech-Recognition/blob/master/docs/training/AI%20Hub%20Dataset%20%232.md)   
[AI Hub Dataset #3](https://github.com/sh951011/Korean-Speech-Recognition/blob/master/docs/training/AI%20Hub%20Dataset%20%233.md) (ongoing)   