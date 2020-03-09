import torch.nn as nn
import torch.nn.functional as F

class ListenAttendSpell(nn.Module):
    r"""
    Listen, Attend and Spell (LAS) Model

    Args:
        listener (torch.nn.Module): encoder of seq2seq
        speller (torch.nn.Module): decoder of seq2seq
        function (torch.nn.functional): A function used to generate symbols from RNN hidden state

    Inputs: feats, targets, teacher_forcing_ratio, use_beam_search
        - **feats** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **targets** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0.90)
        - **use_beam_search** (bool): flag indication whether to use beam-search or not (default: false)

    Returns: y_hats, logits
        - **y_hats** (batch, seq_len): predicted y values (y_hat) by the model
        - **logits** (batch, seq_len, vocab_size): logit values by the model

    Examples::

        >>> listener = Listener(feat_size, 256, 0.5, 6, True, 'gru', True)
        >>> speller = Speller(vocab_size, 120, 8, 256 << (1 if use_bidirectional else 0), SOS_TOKEN, EOS_TOKEN, 3, 'gru', 0.5 ,True, device)
        >>> model = ListenAttendSpell(listener, speller)
        >>> y_hats, logits = model()
    """
    def __init__(self, listener, speller, function=F.log_softmax, use_pyramidal=False):
        super(ListenAttendSpell, self).__init__()
        self.listener = listener
        self.speller = speller
        self.function = function
        self.use_pyramidal = use_pyramidal

    def forward(self, feats, targets, teacher_forcing_ratio=0.90, use_beam_search=False):
        listener_outputs = self.listener(feats)
        y_hats, logits = self.speller(
            inputs=targets,
            listener_outputs=listener_outputs,
            function=self.function,
            teacher_forcing_ratio=teacher_forcing_ratio,
            use_beam_search=use_beam_search
        )

        return y_hats, logits

    def set_beam_size(self, k):
        self.speller.k = k

    def flatten_parameters(self):
        self.listener.flatten_parameters()
        self.speller.rnn.flatten_parameters()