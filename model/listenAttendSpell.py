import torch.nn as nn


class ListenAttendSpell(nn.Module):
    r"""
    Listen, Attend and Spell (LAS) Model

    Args:
        listener (torch.nn.Module): encoder of seq2seq
        speller (torch.nn.Module): decoder of seq2seq

    Inputs: inputs, targets, teacher_forcing_ratio, use_beam_search
        - **inputs** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **targets** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0.90)
        - **use_beam_search** (bool): flag indication whether to use beam-search or not (default: false)

    Returns: y_hats, logits
        - **y_hats** (batch, seq_len): predicted y values (y_hat) by the model
        - **logits** (batch, seq_len, class_num): logit values by the model

    Examples::

        >>> listener = Listener(in_features=80, hidden_dim=256, dropout_p=0.5, ...)
        >>> speller = Speller(num_class, 120, 8, 256 << (1 if use_bidirectional else 0), ...)
        >>> model = ListenAttendSpell(listener, speller)
        >>> y_hats, logits = model()
    """
    def __init__(self, listener, speller):
        super(ListenAttendSpell, self).__init__()
        self.listener = listener
        self.speller = speller

    def forward(self, inputs, input_lengths, targets, teacher_forcing_ratio=0.90, use_beam_search=False):
        listener_outputs, h_state = self.listener(inputs, input_lengths)
        hypothesis, logits = self.speller(
            inputs=targets,
            listener_outputs=listener_outputs,
            teacher_forcing_ratio=teacher_forcing_ratio,
            use_beam_search=use_beam_search
        )

        return hypothesis, logits

    def set_beam_size(self, k):
        self.speller.k = k

    def flatten_parameters(self):
        self.listener.flatten_parameters()
        self.speller.rnn.flatten_parameters()
