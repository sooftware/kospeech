import torch


def get_pad_mask(inputs, input_lengths=None, pad_id=None):
    """
    Padding position is set to 0, either use input_lengths or pad_id

    Examples::
        >>> get_pad_mask(inputs, input_lengths)
    """
    assert (input_lengths is None and pad_id is not None) or (input_lengths is not None and pad_id is None)

    if input_lengths is not None:
        # inputs: N x T x ..
        batch_size = inputs.size(0)
        pad_mask = inputs.new_zeros(inputs.size()[:-1])  # N x T
        for i in range(batch_size):
            pad_mask[i, input_lengths[i]:] = 1

    else:
        # inputs: N x T
        assert inputs.dim() == 2
        pad_mask = inputs.eq(pad_id).float()
    # unsqueeze(-1) for broadcast
    return pad_mask.unsqueeze(-1).bool()


def get_subsequent_mask(inputs):
    """
    Makes subsequent masking like following:

    Examples::
        >>> get_subsequent_mask(inputs)
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

    batch_size, seq_length = inputs.size()
    subsequent_mask = torch.triu(torch.ones((seq_length, seq_length), device=inputs.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)  # BxTxT

    return subsequent_mask


def get_attn_pad_mask(key, length, pad_id):
    """ For masking out the padding part of key sequence. """

    # Expand to fit the shape of key query attention matrix.
    attn_pad_mask = key.eq(pad_id)
    attn_pad_mask = attn_pad_mask.unsqueeze(1).expand(-1, length, -1)  # b x lq x lk

    return attn_pad_mask
