import torch
import torch.nn as nn
import torch.nn.functional as F
from kospeech.models.modules import BaseRNN
from kospeech.models.attention import MultiHeadAttention
from torch import Tensor


class SpellingCorrectorDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,                    # number of classfication
        max_length: int = 120,               # a maximum allowed length for the sequence to be processed
        hidden_dim: int = 1024,              # dimension of RNN`s hidden state vector
        sos_id: int = 1,                     # start of sentence token`s id
        eos_id: int = 2,                     # end of sentence token`s id
        num_heads: int = 4,                  # number of attention heads
        num_layers: int = 3,                 # number of RNN layers
        rnn_type: str = 'lstm',              # type of RNN cell
        dropout_p: float = 0.3,              # dropout probability
        device: str = 'cuda'                 # device - 'cuda' or 'cpu'
    ) -> None:
        super(SpellingCorrectorDecoder, self).__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        self.input_dropout = nn.Dropout(dropout_p)

        self.layers = nn.ModuleList([
            SpellingCorrectorDecoderLayer(
                hidden_dim=hidden_dim,
                rnn_type=rnn_type,
                dropout_p=dropout_p,
                device=device
            ) for _ in range(num_layers)
        ])

        self.attention = MultiHeadAttention(hidden_dim)

    def forward(self, inputs: Tensor, encoder_outputs: Tensor) -> Tensor:
        inputs = self.embedding(inputs)
        output = self.rnn(inputs)

        context, attn = self.attention(output, encoder_outputs, encoder_outputs)

        for layer in self.layers:
            output = layer(output)

        output = F.log_softmax(output, dim=-1)
        output += F.log_softmax(context, dim=-1)

        return output


class SpellingCorrectorDecoderLayer(BaseRNN):
    def __init__(self, hidden_dim, rnn_type, dropout_p, device):
        super(SpellingCorrectorDecoderLayer, self).__init__(
            input_size=hidden_dim << 1,
            hidden_dim=hidden_dim,
            num_layers=1,
            rnn_type=rnn_type,
            dropout_p=dropout_p,
            bidirectional=False,
            device=device
        )
        self.dropout = nn.Dropout(dropout_p=dropout_p)

    def forward(self, query, context):
        residual = query
        output = torch.cat((query, context), dim=2)

        output = self.rnn(output)
        output = self.dropout(output)

        return output + residual
