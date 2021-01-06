import torch
from kospeech.models.las.encoder import Listener

inputs = torch.rand(1, 80, 100)  # BxDxT
input_lengths = [100]
encoder = Listener(80, 512, device='cpu')
encoder_outputs, encoder_log_probs, encoder_output_lengths = encoder(inputs, input_lengths)

print(encoder_outputs)
# tensor([[[ 0.0336, -0.0324, -0.0320,  ...,  0.0731,  0.0341,  0.0223],
#          [ 0.0554, -0.0084, -0.0508,  ...,  0.0577,  0.0135,  0.0039],
#          [ 0.0292, -0.0042, -0.0784,  ...,  0.0600, -0.0215,  0.0316],
#          ...,
#          [ 0.0079, -0.0055, -0.0577,  ...,  0.0682, -0.0573,  0.0480],
#          [-0.0024,  0.0425, -0.0625,  ...,  0.0310, -0.0621,  0.0392],
#          [ 0.0007,  0.0371, -0.0968,  ...,  0.0186, -0.0425,  0.0232]]]

print(encoder_outputs.size())  # torch.Size([1, 25, 1024])
