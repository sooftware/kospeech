# Korean-ASR
based on https://github.com/sh951011/Naver-Hackathon-2019-Speech-Team_Kai.Lib  
Modifying from above repository  
(+) remove silence from audio signal  
![rm_silence](https://postfiles.pstatic.net/MjAxOTEwMzFfMjgy/MDAxNTcyNTI0ODg3Nzcw.rrhpw0MQUaT74qJTM38Q-1z7TxOXlm-rfNXEPRJTY_Ag.SdAUwOdD1loQt2CJBNUbFYUFElG3dSaAly9iZiHwu1Eg.PNG.sooftware/image.png?type=w773)  
(+) add log Mel feature  
![feature_extraction](https://postfiles.pstatic.net/MjAxOTEwMzFfMjE4/MDAxNTcyNTIxNTQ2ODk0.M17MGaHYxtsa_aTH4YO5uZgdVVJaubIkPTJdFZjPopgg.yDEQa5pRaj6Rvd1p3gLGZBYMv32fiArBMhlEYU4tdz4g.PNG.sooftware/image.png?type=w773)  
(+) Modify the size of Encoder & Decoder layer size differently  
* original  
![original](https://postfiles.pstatic.net/MjAxOTEwMzFfMTQ5/MDAxNTcyNTIxNjM5MDAw.twW68KaLExncbk5DM-LCwt9KKvXRhnnhv3KqU0vUnekg.EA-DBdyvt-YPFYqxljAUHZ07jqXw_4bUgrq5DOP5o84g.PNG.sooftware/image.png?type=w773)  
* modify  
![modify](https://postfiles.pstatic.net/MjAxOTEwMzFfMTgw/MDAxNTcyNTIxNjYzNjc2.fSIYRYDJpI_wWHmrKL_8thae3bNDR5s4AfoBipWvuC8g.DnVgwbOxpsv7j4RTKeCt56mta_RoYYUpc96lzA71sXIg.PNG.sooftware/image.png?type=w773)  
(+) Modify Convolution layer of Encoder  
* original  
```python
Sequential(
  (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): Hardtanh(min_val=0, max_val=20, inplace=True)
  (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): Hardtanh(min_val=0, max_val=20, inplace=True)
  (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (8): Hardtanh(min_val=0, max_val=20, inplace=True)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (12): Hardtanh(min_val=0, max_val=20, inplace=True)
  (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (16): Hardtanh(min_val=0, max_val=20, inplace=True)
)
```
* modify  
```python
Sequential(
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
```
