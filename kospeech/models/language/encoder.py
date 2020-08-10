from typing import Tuple
import torch.nn as nn
from torch import Tensor
from kospeech.models.modules import BaseRNN


class LanguageModelEncoder(BaseRNN):
    def __init__(self,
                 vocab_size: int,                       # size of vocab
                 hidden_dim: int = 512,                 # dimension of RNN`s hidden state
                 device: str = 'cuda',                  # device - 'cuda' or 'cpu'
                 dropout_p: float = 0.3,                # dropout probability
                 num_layers: int = 3,                   # number of RNN layers
                 bidirectional: bool = True,            # if True, becomes a bidirectional encoder
                 rnn_type: str = 'lstm') -> None:       # type of RNN cell
        super(LanguageModelEncoder, self).__init__(hidden_dim, hidden_dim, num_layers,
                                                   rnn_type, dropout_p, bidirectional, device)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.input_dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, input_lengths: Tensor = None) -> Tuple[Tensor, Tensor]:
        embedded = self.embedding(inputs)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        output, hidden = self.rnn(embedded)

        if input_lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden
