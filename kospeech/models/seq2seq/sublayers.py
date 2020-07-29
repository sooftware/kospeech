import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Tuple, Optional, Any
from torch import Tensor, BoolTensor

from kospeech.models.modules import LayerNorm


class BaseRNN(nn.Module):
    """
    Applies a multi-layer RNN to an input sequence.

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        input_size (int): size of input
        hidden_dim (int): dimension of RNN`s hidden state vector
        num_layers (int, optional): number of RNN layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional RNN (defulat: False)
        rnn_type (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability (default: 0)
        device (torch.device): device - 'cuda' or 'cpu'

    Attributes:
          supported_rnns = Dictionary of supported rnns
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN
    }

    def __init__(self,
                 input_size: int,                       # size of input
                 hidden_dim: int = 512,                 # dimension of RNN`s hidden state vector
                 num_layers: int = 1,                   # number of recurrent layers
                 rnn_type: str = 'lstm',                # number of RNN layers
                 dropout_p: float = 0.3,                # dropout probability
                 bidirectional: bool = True,            # if True, becomes a bidirectional rnn
                 device: str = 'cuda') -> None:         # device - 'cuda' or 'cpu'
        super(BaseRNN, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_size, hidden_dim, num_layers, True, True, dropout_p, bidirectional)
        self.hidden_dim = hidden_dim
        self.device = device
        self._init_params()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _init_params(self):
        for name, param in self.rnn.named_parameters():
            if name.startswith('weight_ih'):
                init.xavier_uniform_(param)
            elif name.startswith('weight_hh'):
                init.xavier_uniform_(param)
            elif name.startswith('bias'):
                init.zeros_(param)


class MaskConv(nn.Module):
    """
    Masking Convolutional Neural Network

    Adds padding to the output of the module based on the given lengths.
    This is to ensure that the results of the model do not change when batch sizes change during inference.
    Input needs to be in the shape of (batch_size, channel, hidden_dim, seq_len)

    Refer to https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
    Copyright (c) 2017 Sean Naren
    MIT License

    Args:
        sequential (torch.nn): sequential list of convolution layer

    Inputs: inputs, seq_lengths
        - **inputs** (torch.FloatTensor): The input of size BxCxHxT
        - **seq_lengths** (torch.IntTensor): The actual length of each sequence in the batch

    Returns: output, seq_lengths
        - **output**: Masked output from the sequential
        - **seq_lengths**: Sequence length of output from the sequential
    """
    def __init__(self, sequential: nn.Sequential) -> None:
        super(MaskConv, self).__init__()
        self.sequential = sequential

    def forward(self, inputs: Tensor, seq_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        output = None

        for module in self.sequential:
            output = module(inputs)
            mask = BoolTensor(output.size()).fill_(0)

            if output.is_cuda:
                mask = mask.cuda()

            seq_lengths = self.get_sequence_lengths(module, seq_lengths)

            for idx, length in enumerate(seq_lengths):
                length = length.item()

                if (mask[idx].size(2) - length) > 0:
                    mask[idx].narrow(dim=2, start=length, length=mask[idx].size(2) - length).fill_(1)

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lengths

    def get_sequence_lengths(self, module: nn.Module, seq_lengths: Tensor) -> Tensor:
        """
        Calculate convolutional neural network receptive formula

        Args:
            module (torch.nn.Module): module of CNN
            seq_lengths (torch.IntTensor): The actual length of each sequence in the batch

        Returns: seq_lengths
            - **seq_lengths**: Sequence length of output from the module
        """
        if isinstance(module, nn.Conv2d):
            numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lengths = numerator / module.stride[1] + 1

        elif isinstance(module, nn.MaxPool2d):
            seq_lengths >>= 1

        return seq_lengths.int()


class CNNExtractor(nn.Module):
    """
    Provides inteface of convolutional extractor.

    Note:
        Do not use this class directly, use one of the sub classes.
    """
    supported_activations = {
        'hardtanh': nn.Hardtanh(0, 20, inplace=True),
        'relu': nn.ReLU(inplace=True),
        'elu': nn.ELU(inplace=True),
        'leaky_relu': nn.LeakyReLU(inplace=True),
        'gelu': nn.GELU()
    }

    def __init__(self, activation: str = 'hardtanh') -> None:
        super(CNNExtractor, self).__init__()
        self.activation = CNNExtractor.supported_activations[activation]

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Optional[Any]:
        raise NotImplementedError


class VGGExtractor(CNNExtractor):
    """
    VGG extractor for automatic speech recognition described in
    "Advances in Joint CTC-Attention based End-to-End Speech Recognition with a Deep CNN Encoder and RNN-LM" paper
    - https://arxiv.org/pdf/1706.02737.pdf
    """
    def __init__(self, activation: str, mask_conv: bool):
        super(VGGExtractor, self).__init__(activation)
        self.mask_conv = mask_conv
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            self.activation,
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            self.activation,
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            self.activation,
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            self.activation,
            nn.MaxPool2d(2, stride=2)
        )
        if mask_conv:
            self.conv = MaskConv(self.conv)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Optional[Any]:
        if self.mask_conv:
            conv_feat, seq_lengths = self.conv(inputs, input_lengths)
            output = conv_feat, seq_lengths
        else:
            conv_feat = self.conv(inputs)
            output = conv_feat

        return output


class DeepSpeech2Extractor(CNNExtractor):
    """
    DeepSpeech2 extractor for automatic speech recognition described in
    "Deep Speech 2: End-to-End Speech Recognition in English and Mandarin" paper
    - https://arxiv.org/abs/1512.02595
    """
    def __init__(self, activation: str = 'hardtanh', mask_conv: bool = False) -> None:
        super(DeepSpeech2Extractor, self).__init__(activation)
        self.mask_conv = mask_conv
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(32),
            self.activation,
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(32),
            self.activation
        )
        if mask_conv:
            self.conv = MaskConv(self.conv)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Optional[Any]:
        if self.mask_conv:
            conv_feat, seq_lengths = self.conv(inputs, input_lengths)
            output = conv_feat, seq_lengths
        else:
            conv_feat = self.conv(inputs)
            output = conv_feat

        return output
