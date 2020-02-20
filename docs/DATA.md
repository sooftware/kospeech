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
