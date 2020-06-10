import torch
import torch.nn as nn


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


class ConvolutionalExtractor(nn.Module):
    """
    Provides inteface of extractor.

    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, activation='hardtanh'):
        super(ConvolutionalExtractor, self).__init__()
        if activation.lower() == 'hardtanh':
            self.activation = nn.Hardtanh(0, 20, inplace=True)
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation.lower() == 'leacky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError("Unsupported activation function : {0}".format(activation))

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class VGGExtractor(ConvolutionalExtractor):
    """
    VGG extractor for automatic speech recognition described in
    "Advances in Joint CTC-Attention based End-to-End Speech Recognition with a Deep CNN Encoder and RNN-LM" paper
    - https://arxiv.org/pdf/1706.02737.pdf
    """
    def __init__(self, in_channels=1, activation='hardtanh'):
        super(VGGExtractor, self).__init__(activation)
        self.cnn = MaskCNN(
            nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
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
        )

    def forward(self, inputs, input_lengths):
        conv_feat, seq_lengths = self.cnn(inputs, input_lengths)
        return conv_feat, seq_lengths


class DeepSpeech2Extractor(ConvolutionalExtractor):
    """
    DeepSpeech2 extractor for automatic speech recognition described in
    "Deep Speech 2: End-to-End Speech Recognition in English and Mandarin" paper
    - https://arxiv.org/abs/1512.02595
    """

    def __init__(self, in_channels=1, activation='hardtanh'):
        super(DeepSpeech2Extractor, self).__init__(activation)
        self.cnn = MaskCNN(
            nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                self.activation,
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                self.activation
            )
        )

    def forward(self, inputs, input_lengths):
        conv_feat, seq_lengths = self.cnn(inputs, input_lengths)
        return conv_feat, seq_lengths
