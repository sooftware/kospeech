# Options
  
main.py
```
usage: main.py [-h] [--mode] [--sample_rate]
               [--window_size] [--stride] [--n_mels]
               [--normalize] [--del_silence] [--input_reverse]
               [--feature_extract_by] [--time_mask_para] [--freq_mask_para]
               [--time_mask_num] [--freq_mask_num]
               [--use_bidirectional] [--hidden_dim]
               [--dropout] [--num_heads] [--label_smoothing]
               [--listener_layer_size] [--speller_layer_size] [--rnn_type]
               [--extractor] [--activation]
               [--attn_mechanism] [--teacher_forcing_ratio]
               [--dataset_path] [--data_list_path]
               [--label_path] [--init_uniform] [--spec_augment]
               [--noise_augment] [--noiseset_size]
               [--noise_level] [--use_cuda]
               [--batch_size] [--num_workers]
               [--num_epochs] [--init_lr]
               [--high_plateau_lr] [--low_plateau_lr] [--valid_ratio]
               [--max_len] [--max_grad_norm]
               [--rampup_period] [--decay_threshold] [--exp_decay_period]
               [--teacher_forcing_step] [--min_teacher_forcing_ratio]
               [--seed] [--save_result_every]
               [--checkpoint_every] [--print_every] [--resume]
```

## Model options

* `--use_bidirectional` : if True, becomes a bidirectional encoder (defulat: `False`)
* `--hidden_dim` : hidden state dimension of model (default: `256`)
* `--dropout` : dropout ratio in training (default: `0.3`)
* `--num_heads` : number of head in attention (default: `8`)
* `--label_smoothing` : ratio of label smoothing (default: `0.1`)
* `--listener_layer_size` : layer size of encoder (default: `5`)
* `--speller_layer_size` : layer size of decoder (default: `3`)
* `--rnn_type` : type of rnn cell: [gru, lstm, rnn] (default: `gru`)
* `--attn_mechanism` : option to specify the attention mechanism method (default: `loc`)
* `--teacher_forcing_ratio` : teacher forcing ratio in decoding (default: `0.99`)
  
## Train options
  
* `--dataset_path` : path of dataset
* `--data_list_path` : list of training / test set
* `--label_path` : path of character labels
* `--use_multi_gpu` : flag indication whether to use multi-gpu in training
* `--init_uniform` : flag indication whether to initiate model`s parameters as uniformly
* `--spec_augment` : flag indication whether to use spec augmentation or not
* `--noise_augment` : flag indication whether to use noise augmentation or not
* `--noiseset_size` : size of noise dataset for noise augmentation (default: `1000`)
* `--noise_level` : set level of noise (default: `0.7`)
* `--use_cuda` : flag indication whether to use cuda or not
* `--batch_size` : batch size in training (default: `32`)
* `--num_workers` : number of workers in dataset loader (default: `4`)
* `--num_epochs` : number of epochs in training (default: `20`)
* `--init_lr` : initial learning rate => before ramp up lr (default: `1e-15`)
* `--high_plateau_lr` : high plateau learning rate => after rampup lr (default: `3e-04`)
* `--low_plateau_lr` : low plateau learning rate => after exponential decay (default: `1e-05`)
* `--valid_ratio` : validation dataset ratio in training dataset (default: `0.01`)
* `--max_len` : maximum characters of sentence (default: `151`)
* `--max_grad_norm` : value used for gradient norm clipping (default: `400`)
* `--rampup_period` : timestep of learning rate rampup (default: `1000`)
* `--decay_threshold` : If the improvement of cer less than this, exponential decay lr start. (default: `0.02`)
* `--exp_decay_period` : Timestep of learning rate decay (default: `160000`)
* `--teacher_forcing_step` : The value at which teacher forcing ratio will be reducing
* `--min_teacher_forcing_ratio` : The minimum value of teacher forcing ratio
* `--seed` : random seed (default: `7`)
* `--save_result_every` : to determine whether to store training results every N timesteps (default: `1000`)
* `--checkpoint_every` : to determine whether to store training checkpoint every N timesteps (default: `5000`)
* `--print_every` : to determine whether to store training progress every N timesteps (default: `10`)
* `--resume` : Indicates if training has to be resumed from the latest checkpoint (default: `False`)
  
## Preprocess options
  
* `--sample_rate` : sample rate (default: `16000`)
* `--window_size` : Window size for spectrogram (default: `20`)
* `--stride` : Window stride for spectrogram (default: `10`)
* `--n_mels` : number of mel filter (default: `80`)
* `--normalize` : flag indication whether to normalize spectrogram or not (default: `False`)
* `--del_silence` : flag indication whether to delete silence or not (default: `False`)
* `--input_reverse` : flag indication whether to reverse input or not (default: `False`)
* `--feature_extract_by` : which library to use for feature extraction: [librosa, torchaudio] (default: `librosa`)
* `--time_mask_para` : Hyper Parameter for Time Masking to limit time masking length (default: `50`)
* `--freq_mask_para` : Hyper Parameter for Freq Masking to limit freq masking length (default: `12`)
* `--time_mask_num` : how many time-masked area to make (default: `2`)
* `--freq_mask_num` : how many freq-masked area to make (default: `2`)  
  
## Inference options
  
* `--dataset_path` : path of dataset
* `--data_list_path` : list of training / test set
* `--label_path` : path of character labels
* `--use_multi_gpu` : flag indication whether to use multi-gpu in training
* `--num_workers` : number of workers in dataset loader (default: `4`)
* `--use_cuda` : flag indication whether to use cuda or not
* `--model_path` : path to load models (default: `None`)
* `--batch_size` : batch size in inference (default: `1`)
* `--k` : size of beam (default: `5`)
* `--use_beam_search` : flag indication whether to use beam search decoding or not (default: `False`)
* `--print_every` : to determine whether to store inference progress every N timesteps (default: `10`)
