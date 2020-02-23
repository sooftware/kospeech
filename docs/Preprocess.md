# Preprocess, Create label & data list
  
본 글은 **AI Hub**에서 제공하는 '**한국어 음성데이터**'에 대해 저희 팀이 적용한 전처리 과정을 기록한 글입니다.   
AI Hub 음성 데이터는 다음 링크에서 신청 후 다운로드 하실 수 있습니다.  
  
AI Hub 한국어 음성 데이터 : http://www.aihub.or.kr/aidata/105  
  
## *Data Structure*  
  
![data-structure](https://postfiles.pstatic.net/MjAyMDAyMjRfNjYg/MDAxNTgyNDc2NzExNDc4.xu4S2PfKcHun-I1dTODrzIQfcQTzjdjdiuVFnvtFh8Ug.kgTQvNWFfv_LoS0HjB91CMU_ochW8bSDybp7a51c3bog.PNG.sooftware/image.png?type=w773)  
  
데이터는 총 123GB로 크게 5개의 폴더로 이루어져 있고, 각 폴더 안에는 124개의 폴더가 있다. 그리고 그 폴더 안에는 1,000개씩의 PCM-TXT 파일로 구성되어 있다.  조용한 환경에서 2,000여명이 발성한 한국어 **1,000시간**의 데이터로 구성되어 있다. 총 파일의 개수는 **622,545**개의 PCM-TXT 파일로 구성되어 있다.  
  
※ 작업의 편의를 위하여 아래부터 이루어지는 작업은 모든 파일을 하나의 폴더 안에 모아서 작업했습니다 ※

  
* KaiSpeech_FileNum.pcm  
![signal](https://postfiles.pstatic.net/MjAyMDAxMjJfMTYx/MDAxNTc5NjcyNzMyMTkz.Kw1WWrvvv9qLEf-pa0QYOcKYL3GOqXxahw_6sBsjqLgg.nkysalfeHToY9_FbVgxVcOM_Q5_RYlbpfFrAdFsdev4g.PNG.sooftware/audio-signal.png?type=w773)
  
* KaiSpeech_FileNum.txt 
```
"b/ 아/ 모+ 몬 소리야 (70%)/(칠 십 퍼센트) 확률이라니 n/"  
```

## Base Function
  
전처리를 위해 필요한 기본 함수들을 정의해보자.  

### **file_num_padding()**  
```python
def file_num_padding(file_num):
    if file_num < 10: 
        return '00000' + str(file_num)
    elif file_num < 100: 
        return '0000' + str(file_num)
    elif file_num < 1000: 
        return '000' + str(file_num)
    elif file_num < 10000: 
        return '00' + str(file_num)
    elif file_num < 100000: 
        return '0' + str(file_num)
    else: 
        return str(file_num)
``` 
  
AI Hub 데이터셋에서 파일 번호는 '000001', '002545', '612543' 와 같은 형식으로 이루어져 있다.   
이러한 형식에 맞춰주기 위하여 파일 번호를 입력으로 받아 해당 포맷에 맞춰주는 함수를 미리 정의해둔다.  
  
### **get_path()**
```python
def get_path(path, fname, filenum, format):
    return path + fname + filenum + format
```
  
텍스트 파일의 경로를 잡아주는 함수를 미리 정의해둔다.  
Example )
```python
BASE_PATH = "E:/한국어 음성데이터/KaiSpeech/"
FNAME = 'KaiSpeech_'
filenum = 1348
format = '.txt'
get_path(BASE_PATH,FNAME,file_num_padding(filenum),".txt")
```
**Output**
```python
'E:/한국어 음성데이터/KaiSpeech/KaiSpeech_001348.txt'
```
  
## *Data-Preprocess* 
  
AI Hub에서 기본적으로 제공하는 음성에 대한 텍스트는 다음과 같다.  
(철자전사) / (발음전사), 노이즈, 더듬는 소리 등 세밀하게 레이블링 되어 있다.  
우리 팀은 *Sound-To-Text* 를 최대한 정확하게 하는 것이 목표였기에 다음과 같은 전처리 과정을 거쳤다.  
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
  
다음은 위와 같은 전처리를 위해 사용한 코드이다.

### **filter_bracket()**  
(A) / (B) 일 때, B만을 가져와주는 함수이다.  
(철자전사) / (발음전사) 중 발음전사를 선택하기 위해 정의했다.  
```python
test1 = "o/ 근데 (70%)/(칠십 퍼센트)가 커 보이긴 하는데 (200)/(이백) 벌다 (140)/(백 사십) 벌면 빡셀걸? b/"
test2 = "근데 (3학년)/(삼 학년) 때 까지는 국가장학금 바+ 받으면서 다녔던 건가?"

def filter_bracket(sentence):
    new_sentence = str()
    flag = False
    
    for ch in sentence:
        if ch == '(' and flag == False: 
            flag = True
            continue
        if ch == '(' and flag == True:
            flag = False
            continue
        if ch != ')' and flag == False:
            new_sentence += ch
    return new_sentence

print(filter_bracket(test1))
print(filter_bracket(test2))
```
**Output**
```python
'o/ 근데 칠십 퍼센트가 커 보이긴 하는데 이백 벌다 백 사십 벌면 빡셀걸? b/'
'근데 삼 학년 때 까지는 국가장학금 바+ 받으면서 다녔던 건가?'
```
  
### **del_special_ch()**
  
문자 단위로 특수 문자 및 노이즈 표기 필터링해주는 함수이다.  
특수 문자를 아예 필터링 해버리면 문제가 되는 '#', '%'와 같은 문자를 확인하고, 문제가 되는 특수 문자는 해당 발음으로 바꿔주었다.  
```python
test1 = "o/ 근데 칠십 퍼센트가 커 보이긴 하는데 이백 벌다 백 사십 벌면 빡셀걸? b/"
test2 = "c# 배워봤어?"

def del_special_ch(sentence):
    SENTENCE_MARK = ['.', '?', ',', '!']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';']
    import re
    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            # o/, n/ 등 처리
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx+1] == '/': 
                continue 
        if ch == '%': 
            new_sentence += '퍼센트'
        elif ch == '#': 
            new_sentence += '샾'
        elif ch not in EXCEPT: 
            new_sentence += ch
    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence

print(del_special_ch(test1))
print(del_special_ch(test2))
```
**Output**
```python
'근데 칠십 퍼센트가 커 보이긴 하는데 이백 벌다 백 사십 벌면 빡셀걸?'
'c샾 배워봤어?'
```

## *Create Character labels*  
  
위와 같이 AI Hub에서 제공되는 텍스트는 일정한 포맷의 '**한글**'로 구성되어 있다. 위의 한글 텍스트로는 학습을 시킬수가 없으므로, 컴퓨터가 이해할 수 있도록 '**숫자**'로 바꾸어 줘야 한다.  
  
그러기 위해서 먼저, 데이터셋이 어떠한 문자들로 이루어져 있는지를 파악해야한다.  
  
그럼 데이터셋에서 등장하는 모든 문자를 확인을 해보자.  
  
  
### **Create Character labels**
```python
import pandas as pd
from tqdm import trange # display progress

BASE_PATH = "E:/한국어 음성데이터/KaiSpeech/"
FNAME = 'KaiSpeech_'
TOTAL_NUM = 622545
label_list = []
label_freq = []

print('started...')
for filenum in trange(1,TOTAL_NUM):
    f = open(get_path(BASE_PATH,FNAME,file_num_padding(filenum),".txt"))
    sentence = f.readline()
    f.close()
    for ch in sentence:
        if ch not in label_list:
            label_list.append(ch)
            label_freq.append(1)
        else:
            label_freq[label_list.index(ch)] += 1
# sort together Using zip
label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
label = {'id':[], 'char':[], 'freq' :[]}
for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
    label['id'].append(idx)
    label['char'].append(ch)
    label['freq'].append(freq)

""" dictionary to csv """
label_df = pd.DataFrame(label)
label_df.to_csv("aihub_labels.csv", encoding="utf-8", index=False)
```
  
위의 코드를 실행시켜서 정상적으로 종료됐다면 다음과 같은 파일이 생길 것이다.  
( \<s>, \</s>, _ 는 수동으로 추가 )
  
|id|char|freq|  
|:--:|:----:|:----:|   
|0| |5774462|   
|1|.|640924|   
|2|그|556373|   
|3|이|509291|   
|.|.|.|  
|.|.|.|     
|2334|\<s\>|0|   
|2335|\</s\>|0|   
|2336|\_|0|  
  
수동으로 추가해 준 3개의 레이블을 포함하여 총 **2,337**개의 문자 레이블이 완성되었다.  
우리 팀은 위의 레이블 리스트 중 1번씩 등장한 문자에 주목했다.  
  
'갗', '괞', '긃' 등의 생소한 문자가 약 300개 정도를 차지했는데   우리 팀은 이러한 레이블은 노이즈가 될 것이라고 생각했고, 이에 대한 처리를 고민했다. 1번씩 등장한 파일들을 확인해서 하나하나 수작업으로 레이블을 바꿔주려 했지만, 실제로 음성 파일을 들어보게 되면 바꿔주기가 상당히 애매했다.  
  
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

그래서 우리는 이렇게 1번씩 등장한 문자가 포함된 파일은 **테스트 데이터**로 사용하고, 2번 이상 등장한 문자들만 있는 파일들로만 **트레이닝 데이터**를 구성했다. 이렇게 1번씩 등장한 300여개의 문자들이 포함된 파일들을 제외를 해서, 위의 표처럼 **2,040**개의 문자로만 트레이닝을 시킬 수 있었다.  
   
