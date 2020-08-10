import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from typing import Optional, Any, Tuple
from kospeech.models.modules import Linear
from kospeech.models.modules import BaseRNN
from kospeech.models.acoustic.transformer.sublayers import AddNorm
from kospeech.models.attention import MultiHeadAttention


class LanguageDecoderRNN(BaseRNN):
    def __init__(self,
                 num_classes: int,                    # number of classfication
                 max_length: int = 120,               # a maximum allowed length for the sequence to be processed
                 hidden_dim: int = 1024,              # dimension of RNN`s hidden state vector
                 sos_id: int = 1,                     # start of sentence token`s id
                 eos_id: int = 2,                     # end of sentence token`s id
                 attn_mechanism: str = 'multi-head',  # type of attention mechanism
                 num_heads: int = 4,                  # number of attention heads
                 num_layers: int = 2,                 # number of RNN layers
                 rnn_type: str = 'lstm',              # type of RNN cell
                 dropout_p: float = 0.3,              # dropout probability
                 device: str = 'cuda') -> None:       # device - 'cuda' or 'cpu'
        super(LanguageDecoderRNN, self).__init__(hidden_dim, hidden_dim, num_layers, rnn_type, dropout_p, False, device)
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.attn_mechanism = attn_mechanism.lower()
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        self.attention = AddNorm(MultiHeadAttention(hidden_dim), hidden_dim)
        self.projection = AddNorm(Linear(hidden_dim, hidden_dim, bias=True), hidden_dim)
        self.generator = Linear(hidden_dim, num_classes, bias=False)

    def forward_step(self, input_var: Tensor, hidden: Optional[Any],
                     encoder_outputs: Tensor) -> Tuple[Tensor, Optional[Any]]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)

        embedded = self.embedding(input_var).to(self.device)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(embedded, hidden)
        context, attn = self.attention(output, encoder_outputs, encoder_outputs)

        output = self.projection(context.view(-1, self.hidden_dim)).view(batch_size, -1, self.hidden_dim)
        output = self.generator(torch.tanh(output).contiguous().view(-1, self.hidden_dim))

        step_output = F.log_softmax(output, dim=1)
        step_output = step_output.view(batch_size, output_lengths, -1).squeeze(1)

        return step_output, hidden

    def forward(self, inputs: Tensor, encoder_outputs: Tensor, teacher_forcing_ratio: float = 1.0) -> Tuple[Tensor, dict]:
        hidden, attn, result = None, None, list()
        inputs, batch_size, max_length = self.validate_args(inputs, encoder_outputs, teacher_forcing_ratio)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            inputs = inputs[inputs != self.eos_id].view(batch_size, -1)
            step_outputs, hidden = self.forward_step(inputs, hidden, encoder_outputs)

            for di in range(step_outputs.size(1)):
                step_output = step_outputs[:, di, :]
                result.append(step_output)

        else:
            input_var = inputs[:, 0].unsqueeze(1)

            for di in range(max_length):
                step_output, hidden = self.forward_step(input_var, hidden, encoder_outputs)
                result.append(step_output)
                input_var = result[-1].topk(1)[1]

        return result

    def validate_args(self, inputs: Optional[Any] = None, encoder_outputs: Tensor = None,
                      teacher_forcing_ratio: float = 1.0) -> Tuple[Tensor, int, int]:
        """ Validate arguments """
        assert encoder_outputs is not None
        batch_size = encoder_outputs.size(0)

        if inputs is None:  # inference
            inputs = LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length

            if torch.cuda.is_available():
                inputs = inputs.cuda()

            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")

        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length
