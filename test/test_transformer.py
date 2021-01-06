import torch
from kospeech.models import SpeechTransformer

batch_size = 4
seq_length = 200
target_length = 20
input_size = 80

transformer = SpeechTransformer(num_classes=10, d_model=16, d_ff=32, num_encoder_layers=3, num_decoder_layers=2)

inputs = torch.FloatTensor(batch_size, seq_length, input_size)
input_lengths = torch.LongTensor([seq_length, seq_length - 10, seq_length - 20, seq_length - 30])
targets = torch.randint(0, 10, size=(batch_size, target_length), dtype=torch.long)

output = transformer(inputs, input_lengths, targets)
print(output)
