import torch.nn as nn
from e2e.model.sub_layers.maskCNN import MaskCNN


class VGGExtractor(nn.Module):
    """
    VGG extractor for automatic speech recognition described in
    "Advances in Joint CTC-Attention based End-to-End Speech Recognition with "a Deep CNN Encoder and RNN-LM" paper
    - https://arxiv.org/pdf/1706.02737.pdf
    """
    def __init__(self, in_channels=1):
        super(VGGExtractor, self).__init__()
        self.cnn = MaskCNN(
            nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
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
        )

    def forward(self, inputs, input_lengths):
        conv_feat, seq_lengths = self.cnn(inputs, input_lengths)
        return conv_feat, seq_lengths


class DeepSpeech2Extractor(nn.Module):
    """
    VGG extractor for automatic speech recognition described in
    "Advances in Joint CTC-Attention based End-to-End Speech Recognition with "a Deep CNN Encoder and RNN-LM" paper
    - https://arxiv.org/pdf/1706.02737.pdf
    """

    def __init__(self, in_channels=1):
        super(DeepSpeech2Extractor, self).__init__()
        self.cnn = MaskCNN(
            nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True)
            )
        )

    def forward(self, inputs, input_lengths):
        conv_feat, seq_lengths = self.cnn(inputs, input_lengths)
        return conv_feat, seq_lengths
