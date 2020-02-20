## SpecAugmentation
Applying Frequency Masking & Time Masking except Time Warping
* code  
```python
def spec_augment(feat, T=40, F=15, time_mask_num=2, freq_mask_num=2):
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
