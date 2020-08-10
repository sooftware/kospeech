from typing import Tuple
import torch.nn as nn
from torch import Tensor
from kospeech.models.modules import BaseRNN


class LanguageEncoderRNN(BaseRNN):
    def __init__(self,
                 vocab_size: int,                       # size of vocab
                 hidden_dim: int = 512,                 # dimension of RNN`s hidden state
                 device: str = 'cuda',                  # device - 'cuda' or 'cpu'
                 dropout_p: float = 0.3,                # dropout probability
                 num_layers: int = 3,                   # number of RNN layers
                 bidirectional: bool = True,            # if True, becomes a bidirectional encoder
                 rnn_type: str = 'lstm') -> None:       # type of RNN cell
        super(LanguageEncoderRNN, self).__init__(hidden_dim, hidden_dim, num_layers,
                                                 rnn_type, dropout_p, bidirectional, device)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.input_dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        embedded = self.embedding(inputs)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(embedded)

        return output, hidden
