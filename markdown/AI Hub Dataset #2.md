### Training Result  

|Epoch|train CRR (%)|valid CRR (%)|test CRR (%)|  
|:-----:|:---------:|:--------:|:------:|    
|0|33.86|42.35|-|   
|1|64.35|67.25|-|   
|2|71.04|73.21|72.50|  
|3|76.82|74.65|-|  
|4|77.51|75.21|-|  
|5|78.41|76.52|76.46|  

**CRR** : Character Recognition Rate

### Hyperparameters  
| Hyperparameter  |Help| Use|              
| ----------      |---|:----------:|    
| use_bidirectional| if True, becomes a bidirectional encoder|True|  
| use_attention    | flag indication whether to use attention mechanism or not|True |   
|input_reverse|flag indication whether to reverse input feature or not|True|   
|use_augment| flag indication whether to use spec-augmentation or not|True|  
|use_pyramidal| flag indication whether to use pLSTM or not|False|  
|augment_ratio|ratio of spec-augmentation applied data|0.4|   
|listener_layer_size|number of listener`s RNN layer|5|  
| speller_layer_size|number of speller`s RNN layer| 3|  
| hidden_size| size of hidden state of RNN|256|
| batch_size | mini-batch size|6|
| dropout          | dropout probability|0.5  |
| teacher_forcing  | The probability that teacher forcing will be used|0.99|
| lr               | learning rate|1e-4        |
| max_epochs       | max epoch|40          |   

### Training Envirionment  
```
Device : GTX 1080 Ti   
CUDA version : 10.1  
PyTorch version : 1.3.1    
```
