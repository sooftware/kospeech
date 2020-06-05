import torch.nn as nn
from e2e.model.sub_layers.maskCNN import MaskCNN


class Extractor(nn.Module):
    """
    Provides inteface of extractor.

    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, activation='elu'):
        super(Extractor, self).__init__()
        if activation.lower() == 'hardtanh':
            self.activation = nn.Hardtanh(0, 20, inplace=True)
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation.lower() == 'leacky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError("Unsupported activation function : {0}".format(activation))

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class VGGExtractor(Extractor):
    """
    VGG extractor for automatic speech recognition described in
    "Advances in Joint CTC-Attention based End-to-End Speech Recognition with a Deep CNN Encoder and RNN-LM" paper
    - https://arxiv.org/pdf/1706.02737.pdf
    """
    def __init__(self, in_channels=1, activation='elu'):
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


class DeepSpeech2Extractor(Extractor):
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
