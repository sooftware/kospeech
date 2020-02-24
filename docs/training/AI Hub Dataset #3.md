## Model  
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
      (13): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (14): Hardtanh(min_val=0, max_val=20, inplace=True)
      (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (bottom_rnn): GRU(2048, 256, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
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
      (attention): DotProductAttention(
        (W): Linear(in_features=1024, out_features=512, bias=True)
      )
    )
    (out): Linear(in_features=512, out_features=2040, bias=True)
  )
)
```

### Listener - *pBLSTM*  
  
![listener](https://postfiles.pstatic.net/MjAyMDAyMjNfODcg/MDAxNTgyNDY5NTAzNjU0.442JiKr1UVgNODDCjcBrzD2_7DKIVRcYPHb3tvoUbT8g.lvIzspCahYtJqfXOeSh4zFkfzvb-3c7ISjlQqJ00ZsUg.PNG.sooftware/image.png?type=w773)  

## Hyperparameters  
| Hyperparameter  |Help| Use|              
| ----------      |---|:----------:|    
| use_bidirectional| if True, becomes a bidirectional encoder|True|  
| use_attention    | flag indication whether to use attention mechanism or not|True |   
| score_function    | which attention to use|dot-product |   
| use_label_smoothing    | flag indication whether to use label smoothing or not|True |   
|input_reverse|flag indication whether to reverse input feature or not|True|   
|use_augment| flag indication whether to use spec-augmentation or not|True|  
|use_pyramidal| flag indication whether to use pLSTM or not|True|  
|augment_ratio|ratio of spec-augmentation applied data|1.0|   
|listener_layer_size|number of listener`s RNN layer|6|  
| speller_layer_size|number of speller`s RNN layer| 3|  
| hidden_size| size of hidden state of RNN|256|
| batch_size | mini-batch size|8|
| dropout          | dropout probability|0.5  |
| teacher_forcing  | The probability that teacher forcing will be used|0.90|
| lr               | learning rate|1e-4        |


### Training Envirionment  
```
Device : GTX Titan X   
CUDA version : 10.1  
PyTorch version : 1.4.0    
```  
  
## Feature  
  
* MFCC (Mel-Frequency-Cepstral-Coefficients)  
  
| Parameter| Use|    
| -----|:-----:|     
|Frame length|25ms|
|Stride|10ms|
| N_FFT | 400  |   
| hop length | 160  |
| n_mfcc | 33  |  
|window|hamming|  

  
### Training Result  
   
 Training Start Date: 2020/02/23  
   
|Epoch|train CRR (%)|valid CRR (%)|test CRR (%)|  
|:-----:|:---------:|:--------:|:------:|    
|0|-|-|-|    

**CRR** : Character Recognition Rate
