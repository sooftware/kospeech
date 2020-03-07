import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.beam import Beam
from .attention import Attention

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class Speller(nn.Module):
    r"""
    Converts higher level features (from listener) into output utterances by specifying a probability distribution over sequences of characters.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        layer_size (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention (bool, optional): flag indication whether to use attention mechanism or not (default: false)
        k (int) : size of beam

    Inputs: inputs, listener_hidden, contexts, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **listener_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of listener. Used as the initial hidden state of the decoder. (default `None`)
        - **contexts** (batch, seq_len, hidden_size): tensor with containing the outputs of the listener.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Returns: y_hats, logits
        - **y_hats** (batch, seq_len): predicted y values (y_hat) by the model
        - **logits** (batch, seq_len, vocab_size): logit values by the model

    Examples::

        >>> speller = Speller(vocab_size, max_len, hidden_size, sos_id, eos_id, layer_size)
        >>> y_hats, logits = speller(inputs, listener_hidden, contexts, teacher_forcing_ratio=0.90)
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id, layer_size=1, rnn_cell='gru',
                 dropout_p=0, use_attention=True, device=None, k=8):
        super(Speller, self).__init__()
        assert rnn_cell.lower() == 'lstm' or rnn_cell.lower() == 'gru' or rnn_cell.lower() == 'rnn'
        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU if rnn_cell.lower() == 'gru' else nn.RNN
        self.device = device
        self.rnn = self.rnn_cell(hidden_size , hidden_size, layer_size, batch_first=True, dropout=dropout_p).to(self.device)
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layer_size = layer_size
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.k = k
        if use_attention:
            self.attention = Attention(decoder_hidden_size=hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def _forward_step(self, input, speller_hidden, contexts=None, function=F.log_softmax):
        """ forward one time step """
        batch_size = input.size(0)
        output_size = input.size(1)
        embedded = self.embedding(input).to(self.device)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()
        speller_output = self.rnn(embedded, speller_hidden)[0]

        if self.use_attention:
            contexts = self.attention(speller_output, contexts)
        else:
            contexts = speller_output

        predicted_softmax = function(self.out(contexts.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        return predicted_softmax

    def forward(self, inputs, listener_hidden, contexts, function=F.log_softmax, teacher_forcing_ratio=0.99, use_beam_search=False):
        y_hats, logits = None, None
        decode_results = []
        batch_size = inputs.size(0)
        max_len = inputs.size(1) - 1  # minus the start of sequence symbol
        speller_hidden = torch.FloatTensor(self.layer_size, batch_size, self.hidden_size).uniform_(-0.1, 0.1).to(self.device)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_beam_search:
            """ Beam-Search Decoding """
            inputs = inputs[:, 0].unsqueeze(1)
            beam = Beam(
                k = self.k,
                decoder_hidden = speller_hidden,
                decoder = self,
                batch_size = batch_size,
                max_len = max_len,
                function = function
            )
            y_hats = beam.search(inputs, contexts)
        else:
            if use_teacher_forcing:
                """ if teacher_forcing, Infer all at once """
                inputs = inputs[:, :-1]
                predicted_softmax = self._forward_step(
                    input = inputs,
                    speller_hidden = speller_hidden,
                    contexts = contexts,
                    function = function
                )
                for di in range(predicted_softmax.size(1)):
                    step_output = predicted_softmax[:, di, :]
                    decode_results.append(step_output)
            else:
                input = inputs[:, 0].unsqueeze(1)
                for di in range(max_len):
                    predicted_softmax = self._forward_step(
                        input = input,
                        speller_hidden = speller_hidden,
                        contexts = contexts,
                        function = function
                    )
                    step_output = predicted_softmax.squeeze(1)
                    decode_results.append(step_output)
                    input = decode_results[-1].topk(1)[1]

            logits = torch.stack(decode_results, dim=1).to(self.device)
            y_hats = logits.max(-1)[1]
        return y_hats, logits