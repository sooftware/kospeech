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
  

* code   
```python
def get_librosa_mfcc(filepath = None, n_mfcc = 33, del_silence = False, input_reverse = True, format='pcm'):
    if format == 'pcm':
        pcm = np.memmap(filepath, dtype='h', mode='r')
        sig = np.array([float(x) for x in pcm])
    elif format == 'wav':
        sig, _ = librosa.core.load(filepath, sr=16000)
    else: 
        raise ValueError("Invalid format !!")

    if del_silence:
        non_silence_indices = librosa.effects.split(sig, top_db=30)
        sig = np.concatenate([sig[start:end] for start, end in non_silence_indices])
    feat = librosa.feature.mfcc(y=sig,sr=16000, hop_length=160, n_mfcc=n_mfcc, n_fft=400, window='hamming')
    if input_reverse:
        feat = feat[:,::-1]

    return torch.FloatTensor( np.ascontiguousarray( np.swapaxes(feat, 0, 1) ) )
```
   
* Reference
  + 「 Voice Recognition Using MFCC Algorithm」 Chakraborty et al. 2014
  + https://github.com/librosa/librosa
