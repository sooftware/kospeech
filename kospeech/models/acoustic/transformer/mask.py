import torch
from torch import Tensor
from typing import (
    Any,
    Optional
)


def get_pad_mask(inputs: Tensor, input_lengths: Optional[Any] = None, pad_id: int = None) -> Tensor:
    """
    Padding position is set to True, either use input_lengths or pad_id

    Examples::
        >>> get_pad_mask(inputs, input_lengths)
        tensor([[[False], [False], [False], [False], [False], [ True], [ True], [ True], [ True]],
                [[False], [False], [False], [False], [False], [False], [ True], [ True], [ True]],
                [[False], [False], [False], [False], [False], [False], [False], [False], [ True]]])
    """
    assert (input_lengths is None and pad_id is not None) or (input_lengths is not None and pad_id is None)

    if input_lengths is not None:
        batch_size = inputs.size(0)

        if len(inputs.size()) == 2:
            pad_mask = inputs.new_zeros(inputs.size())  # B x T
        else:
            pad_mask = inputs.new_zeros(inputs.size()[:-1])  # B x T

        for i in range(batch_size):
            pad_mask[i, input_lengths[i]:] = 1

    if pad_id is not None:
        assert inputs.dim() == 2
        pad_mask = inputs.eq(pad_id)

    return pad_mask.unsqueeze(-1).bool()


def get_decoder_self_attn_mask(seq_k: Tensor, seq_q: Tensor, pad_id):
    """
    For masking the decoder self attention

    Example::
        >>> get_decoder_self_attn_mask(seq_k, seq_q, pad_id)
        tensor([[[False,  True,  True,  True,  True,  True,  True],
                 [False, False,  True,  True,  True,  True,  True],
                 [False, False, False,  True,  True,  True,  True],
                 [False, False, False, False,  True,  True,  True],
                 [False, False, False, False,  True,  True,  True],
                 [False, False, False, False,  True,  True,  True],
                 [False, False, False, False,  True,  True,  True]],

                [[False,  True,  True,  True,  True,  True,  True],
                 [False, False,  True,  True,  True,  True,  True],
                 [False, False, False,  True,  True,  True,  True],
                 [False, False, False, False,  True,  True,  True],
                 [False, False, False, False, False,  True,  True],
                 [False, False, False, False, False,  True,  True],
                 [False, False, False, False, False,  True,  True]],

                [[False,  True,  True,  True,  True,  True,  True],
                 [False, False,  True,  True,  True,  True,  True],
                 [False, False, False,  True,  True,  True,  True],
                 [False, False, False, False,  True,  True,  True],
                 [False, False, False, False, False,  True,  True],
                 [False, False, False, False, False, False,  True],
                 [False, False, False, False, False, False,  True]]])
    """
    def get_attn_key_pad_mask(seq_k, seq_q, pad_id):
        """ For masking out the padding part of key sequence. """

        # Expand to fit the shape of key query attention matrix.
        len_q = seq_q.size(1)
        padding_mask = seq_k.eq(pad_id)
        padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

        return padding_mask

    def get_subsequent_mask(inputs: Tensor) -> Tensor:
        """
        Makes subsequent masking like following:

        Examples::
            >>> get_subsequent_mask(inputs)
            tensor([[[False,  True,  True,  True,  True,  True,  True],
                     [False, False,  True,  True,  True,  True,  True],
                     [False, False, False,  True,  True,  True,  True],
                     [False, False, False, False,  True,  True,  True],
                     [False, False, False, False, False,  True,  True],
                     [False, False, False, False, False, False,  True],
                     [False, False, False, False, False, False, False]],

                    [[False,  True,  True,  True,  True,  True,  True],
                     [False, False,  True,  True,  True,  True,  True],
                     [False, False, False,  True,  True,  True,  True],
                     [False, False, False, False,  True,  True,  True],
                     [False, False, False, False, False,  True,  True],
                     [False, False, False, False, False, False,  True],
                     [False, False, False, False, False, False, False]],

                    [[False,  True,  True,  True,  True,  True,  True],
                     [False, False,  True,  True,  True,  True,  True],
                     [False, False, False,  True,  True,  True,  True],
                     [False, False, False, False,  True,  True,  True],
                     [False, False, False, False, False,  True,  True],
                     [False, False, False, False, False, False,  True],
                     [False, False, False, False, False, False, False]]])
        """

        batch_size, seq_length = inputs.size()
        subsequent_mask = torch.triu(torch.ones((seq_length, seq_length), device=inputs.device, dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)  # BxTxT

        return subsequent_mask.bool()

    return get_attn_key_pad_mask(seq_k, seq_q, pad_id) | get_subsequent_mask(seq_k)


def get_attn_pad_mask(inputs, input_lengths, expand_length):
    """
    mask position is set to 1

    Examples::
        >>> get_attn_pad_mask(inputs, input_lengths, expand_length)
        tensor([[[False, False, False, False, False,  True,  True,  True,  True],
                 [False, False, False, False, False,  True,  True,  True,  True],
                 [False, False, False, False, False,  True,  True,  True,  True]],

                [[False, False, False, False, False, False,  True,  True,  True],
                 [False, False, False, False, False, False,  True,  True,  True],
                 [False, False, False, False, False, False,  True,  True,  True]],

                [[False, False, False, False, False, False, False, False,  True],
                 [False, False, False, False, False, False, False, False,  True],
                 [False, False, False, False, False, False, False, False,  True]]])

    """
    # N x Ti x 1
    non_pad_mask = get_pad_mask(inputs, input_lengths=input_lengths).eq(False)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask.squeeze(-1).eq(False)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask
