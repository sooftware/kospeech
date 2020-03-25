import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.beam import Beam
from package.definition import id2char
from package.utils import label_to_string
from .attention import MultiHeadAttention


class Speller(nn.Module):
    r"""
    Converts higher level features (from listener) into output utterances
    by specifying a probability distribution over sequences of characters.

    Args:
        class_num (int): the number of classfication
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        layer_size (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention (bool, optional): flag indication whether to use attention mechanism or not (default: false)
        k (int) : size of beam

    Inputs: inputs, listener_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **listener_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the listener.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Returns: y_hats, logits
        - **y_hats** (batch, seq_len): predicted y values (y_hat) by the model
        - **logits** (batch, seq_len, class_num): predicted log probability by the model

    Examples::

        >>> speller = Speller(class_num, max_len, hidden_size, sos_id, eos_id, n_layers)
        >>> y_hats, logits = speller(inputs, listener_outputs, teacher_forcing_ratio=0.90)
    """

    def __init__(self, class_num, max_len, hidden_size,
                 sos_id, eos_id, n_layers=1, rnn_cell='gru',
                 dropout_p=0, use_attention=True, device=None, k=8):
        super(Speller, self).__init__()

        assert rnn_cell.lower() == 'lstm' or rnn_cell.lower() == 'gru' or rnn_cell.lower() == 'rnn'

        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU if rnn_cell.lower() == 'gru' else nn.RNN
        self.rnn = self.rnn_cell(hidden_size , hidden_size, n_layers, batch_first=True, dropout=dropout_p).to(device)
        self.max_len = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(class_num, self.hidden_size)
        self.n_layers = n_layers
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.k = k
        self.use_attention = use_attention
        self.w = nn.Linear(self.hidden_size, class_num)
        self.device = device
        if use_attention:
            self.attention = MultiHeadAttention(in_features=hidden_size, dim=128, n_head=4)


    def _forward_step(self, input, hidden, listener_outputs=None, function=F.log_softmax):
        """ forward one time step """
        batch_size = input.size(0)
        output_size = input.size(1)

        embedded = self.embedding(input).to(self.device)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(embedded, hidden)

        if self.use_attention:
            output = self.attention(output, listener_outputs)

        predicted_softmax = function(self.w(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)

        return predicted_softmax, hidden


    def forward(self, inputs, listener_outputs, function=F.log_softmax, teacher_forcing_ratio=0.90, use_beam_search=False):
        batch_size = inputs.size(0)
        max_len = inputs.size(1) - 1  # minus the start of sequence symbol
        decode_results = list()
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)

        if use_beam_search: # TopK Decoding
            input = inputs[:, 0].unsqueeze(1)
            beam = Beam(
                k = self.k,
                decoder = self,
                batch_size = batch_size,
                max_len = max_len,
                function = function,
                device = self.device
            )
            logits = None
            y_hats = beam.search(input, listener_outputs)

        else:
            if use_teacher_forcing:  # if teacher_forcing, Infer all at once
                speller_inputs = inputs[inputs != self.eos_id].view(batch_size, -1)
                predicted_softmax, hidden = self._forward_step(
                    input = speller_inputs,
                    hidden = hidden,
                    listener_outputs = listener_outputs,
                    function = function
                )

                for di in range(predicted_softmax.size(1)):
                    step_output = predicted_softmax[:, di, :]
                    decode_results.append(step_output)

            else:
                speller_input = inputs[:, 0].unsqueeze(1)

                for di in range(max_len):
                    predicted_softmax, hidden = self._forward_step(
                        input = speller_input,
                        hidden = hidden,
                        listener_outputs = listener_outputs,
                        function = function
                    )
                    step_output = predicted_softmax.squeeze(1)
                    decode_results.append(step_output)
                    speller_input = decode_results[-1].topk(1)[1]

            logits = torch.stack(decode_results, dim=1).to(self.device)
            y_hats = logits.max(-1)[1]

        print(label_to_string(y_hats.numpy(), id2char, self.eos_id))

        return y_hats, logits
