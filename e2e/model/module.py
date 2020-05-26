import torch
import torch.nn as nn


class BaseRNN(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        input_size (int): size of input
        hidden_dim (int): the number of features in the hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: False)
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

    def __init__(self, input_size, hidden_dim, num_layers, rnn_type, dropout_p, bidirectional, device):
        super(BaseRNN, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_size, hidden_dim, num_layers, True, True, dropout_p, bidirectional)
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class MaskCNN(nn.Module):
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
        - **inputs** (torch.FloatTensor): The input of size BxCxHxS
        - **seq_lengths** (torch.IntTensor): The actual length of each sequence in the batch

    Returns: output, seq_lengths
        - **output**: Masked output from the sequential
        - **seq_lengths**: Sequence length of output from the sequential
    """
    def __init__(self, sequential):
        super(MaskCNN, self).__init__()
        self.sequential = sequential

    def forward(self, inputs, seq_lengths):
        output = None

        for module in self.sequential:
            output = module(inputs)
            mask = torch.BoolTensor(output.size()).fill_(0)

            if output.is_cuda:
                mask = mask.cuda()

            seq_lengths = self.get_seq_lengths(module, seq_lengths)

            for i, length in enumerate(seq_lengths):
                length = length.item()

                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(dim=2, start=length, length=mask[i].size(2) - length).fill_(1)

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lengths

    def get_seq_lengths(self, module, seq_lengths):
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


class VGGExtractor(nn.Module):
    """
    VGG extractor for automatic speech recognition described in
    "Advances in Joint CTC-Attention based End-to-End Speech Recognition with "a Deep CNN Encoder and RNN-LM" paper
    - https://arxiv.org/pdf/1706.02737.pdf
    """
    def __init__(self, in_channel=1):
        super(VGGExtractor).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Hardtanh(0, 20, inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Hardtanh(0, 20, inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, inputs, input_lengths):
        inputs = inputs.unsqueeze(1).permute(0, 1, 3, 2)
        conv_feat, seq_lengths = self.extractor(inputs, input_lengths)
        return conv_feat, seq_lengths
