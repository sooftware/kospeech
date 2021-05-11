import torch
import torch.nn as nn

from kospeech.models.rnnt.model import RNNTransducer

batch_size, sequence_length, dim = 3, 12345, 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(batch_size, sequence_length, dim).to(device)
input_lengths = torch.IntTensor([12345, 12300, 12000])

model = nn.DataParallel(RNNTransducer(
    num_classes=10,
    input_dim=dim,
    num_encoder_layers=3,
)).to(device)

outputs = model.module.recognize(inputs, input_lengths)
print(outputs)
print(outputs.size())
print("PASS")
