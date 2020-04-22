import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.beam import Beam
from .attention import MultiHeadAttention

supported_rnns = {
    'lstm': nn.LSTM,
    'gru': nn.GRU,
    'rnn': nn.RNN
}


class Speller(nn.Module):
    r"""
    Converts higher level features (from listener) into output utterances
    by specifying a probability distribution over sequences of characters.

    Args:
        n_class (int): the number of classfication
        max_length (int): a maximum allowed length for the sequence to be processed
        hidden_dim (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_type (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention (bool, optional): flag indication whether to use attention mechanism or not (default: false)
        k (int) : size of beam

    Inputs: inputs, listener_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **listener_outputs** (batch, seq_len, hidden_dim): tensor with containing the outputs of the listener.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Returns: y_hats, logits
        - **y_hats** (batch, seq_len): predicted y values (y_hat) by the model
        - **logits** (batch, seq_len, n_class): predicted log probability by the model

    Examples::

        >>> speller = Speller(n_class, max_length, hidden_dim, sos_id, eos_id, n_layers)
        >>> y_hats, logits = speller(inputs, listener_outputs, teacher_forcing_ratio=0.90)
    """

    def __init__(self, n_class, max_length, hidden_dim,
                 sos_id, eos_id, n_layers=1, rnn_type='gru', dropout_p=0,
                 use_attention=True, device=None, k=8, external_lm=None):

        super(Speller, self).__init__()
        assert rnn_type.lower() in supported_rnns.keys(), 'RNN type not supported.'

        self.rnn_cell = supported_rnns[rnn_type]
        self.rnn = self.rnn_cell(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout_p).to(device)
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(n_class, self.hidden_dim)
        self.n_layers = n_layers
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.k = k
        self.use_attention = use_attention
        self.fc = nn.Linear(self.hidden_dim, n_class)
        self.device = device
        self.external_lm = external_lm
        if use_attention:
            self.attention = MultiHeadAttention(in_features=hidden_dim, dim=128, n_head=4)

    def forward_step(self, input_, h_state, listener_outputs=None, function=F.log_softmax):
        """ forward one time step """
        batch_size = input_.size(0)
        seq_length = input_.size(1)

        embedded = self.embedding(input_).to(self.device)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        output, h_state = self.rnn(embedded, h_state)

        if self.use_attention:
            output = self.attention(output, listener_outputs)

        predicted_softmax = function(self.fc(output.contiguous().view(-1, self.hidden_dim)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, seq_length, -1)

        return predicted_softmax, h_state

    def forward(self, inputs, listener_outputs, function=F.log_softmax, teacher_forcing_ratio=0.90,
                use_beam_search=False):
        batch_size = inputs.size(0)
        max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        decode_outputs = list()
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        h_state = None

        if use_beam_search:  # TopK Decoding
            input_ = inputs[:, 0].unsqueeze(1)
            beam = Beam(
                k=self.k,
                decoder=self,
                batch_size=batch_size,
                max_length=max_length,
                function=function,
                device=self.device
            )
            logits = None
            y_hats = beam.search(input_, listener_outputs)

        else:
            if use_teacher_forcing:  # if teacher_forcing, Infer all at once
                speller_inputs = inputs[inputs != self.eos_id].view(batch_size, -1)
                predicted_softmax, h_state = self.forward_step(
                    input_=speller_inputs,
                    h_state=h_state,
                    listener_outputs=listener_outputs,
                    function=function
                )

                for di in range(predicted_softmax.size(1)):
                    step_output = predicted_softmax[:, di, :]
                    decode_outputs.append(step_output)

            else:
                speller_input = inputs[:, 0].unsqueeze(1)

                for di in range(max_length):
                    predicted_softmax, h_state = self.forward_step(
                        input_=speller_input,
                        h_state=h_state,
                        listener_outputs=listener_outputs,
                        function=function
                    )
                    step_output = predicted_softmax.squeeze(1)
                    decode_outputs.append(step_output)
                    speller_input = decode_outputs[-1].topk(1)[1]

            logits = torch.stack(decode_outputs, dim=1).to(self.device)
            y_hats = logits.max(-1)[1]

        return y_hats, logits
