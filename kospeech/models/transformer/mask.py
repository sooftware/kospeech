import torch


def get_non_pad_mask(inputs, input_lengths=None, pad_id=None):
    """ Padding position is set to 0, either use input_lengths or pad_idx """
    assert input_lengths is not None or pad_id is not None

    if input_lengths is not None:
        # inputs: N x T x ..
        batch_size = inputs.size(0)
        non_pad_mask = inputs.new_ones(inputs.size()[:-1])  # N x T
        for i in range(batch_size):
            non_pad_mask[i, input_lengths[i]:] = 0

    if pad_id is not None:
        # inputs: N x T
        assert inputs.dim() == 2
        non_pad_mask = inputs.ne(pad_id).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)


def get_subsequent_mask(x):
    """
    Makes subsequent masking like following:

    Examples::
        >>> get_subsequent_mask(x)
        [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] x batch_size
    """

    batch_size, seq_length = x.size()
    subsequent_mask = torch.triu(torch.ones((seq_length, seq_length), device=x.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)  # BxTxT

    return subsequent_mask


def get_attn_key_pad_mask(seq_k, seq_q, pad_id):
    """ For masking out the padding part of key sequence. """

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_id)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_attn_pad_mask(inputs, input_lengths, expand_length):
    """mask position is set to 1"""
    # N x Ti x 1
    non_pad_mask = get_non_pad_mask(inputs, input_lengths=input_lengths)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask
