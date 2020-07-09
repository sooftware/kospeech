import torch
import numpy as np
from torch import Tensor


def pad_masking(x: Tensor, target_length: int, pad_id: int) -> Tensor:
    batch_size, seq_length = x.size()
    pad_indices = x == pad_id
    return pad_indices.unsqueeze(1).expand(batch_size, target_length, seq_length)


def subsequent_masking(x: Tensor) -> Tensor:
    """
    Makes subsequent masking like following:

    Examples::
        >>> subsequent_mask(x)
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
    subsequent_mask = np.triu(np.ones(shape=(seq_length, seq_length)), k=1).astype('bool')  # make lower triangle
    subsequent_mask = torch.tensor(subsequent_mask).to(x.device)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, seq_length, seq_length)
    return subsequent_mask
