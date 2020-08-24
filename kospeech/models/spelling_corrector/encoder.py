import torch.nn as nn
from typing import Tuple
from torch import Tensor
from kospeech.models.modules import BaseRNN, Linear


class SpellingCorrectorEncoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,                       # size of vocab
            hidden_dim: int = 512,                 # dimension of RNN`s hidden state
            device: str = 'cuda',                  # device - 'cuda' or 'cpu'
            dropout_p: float = 0.3,                # dropout probability
            num_layers: int = 3,                   # number of RNN layers
            bidirectional: bool = True,            # if True, becomes a bidirectional encoder
            rnn_type: str = 'lstm'                 # type of RNN cell
    ) -> None:
        super(SpellingCorrectorEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            SpellingCorrectorEncoderLayer(
                hidden_dim=hidden_dim,
                device=device,
                dropout_p=dropout_p,
                num_layers=1,
                bidirectional=bidirectional,
                rnn_type=rnn_type
            ) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        inputs = self.embedding(inputs)
        output = self.input_dropout(inputs)

        for layer in self.layers:
            output = layer(output)

        output = self.fc(output)

        return output


class SpellingCorrectorEncoderLayer(BaseRNN):
    def __init__(
            self,
            hidden_dim: int = 512,                 # dimension of RNN`s hidden state
            device: str = 'cuda',                  # device - 'cuda' or 'cpu'
            dropout_p: float = 0.3,                # dropout probability
            num_layers: int = 3,                   # number of RNN layers
            bidirectional: bool = True,            # if True, becomes a bidirectional encoder
            rnn_type: str = 'lstm'                 # type of RNN cell
    ) -> None:
        super(SpellingCorrectorEncoderLayer, self).__init__(
            hidden_dim, hidden_dim, num_layers,
            rnn_type, dropout_p, bidirectional, device
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(inputs)

        output = self.dropout(output)
        output += inputs

        return output
