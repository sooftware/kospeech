# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


def same_padding(kernel):
    pad_val = (kernel - 1) / 2

    if kernel % 2 == 0:
        out = (int(pad_val - 0.5), int(pad_val + 0.5))
    else:
        out = int(pad_val)

    return out


class ResnetBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_kernels1: tuple,
            num_kernels2: int
    ):
        super(ResnetBlock, self).__init__()

        padding = same_padding(num_kernels1[0])
        self.zero_pad = nn.ZeroPad2d((0, 0, padding[0], padding[1]))
        self.conv1 = nn.Conv2d(in_channels, out_channels, (num_kernels1[0], num_kernels2))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, num_kernels1[1], padding=same_padding(num_kernels1[1]))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, num_kernels1[2], padding=same_padding(num_kernels1[2]))
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, (1, num_kernels2))
        self.bn_shortcut = nn.BatchNorm2d(out_channels)
        self.out_block = nn.ReLU()

    def forward(self, inputs):
        x = self.zero_pad(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.shortcut(inputs)
        shortcut = self.bn_shortcut(shortcut)
        x = torch.add(x, shortcut)
        out_block = self.out_block(x)

        return out_block


class ResnetVADModel(nn.Module):
    """
    Resnet VAD Model. This Model just for inference.
    Please use our pre-train model.
    Refer to : https://github.com/sooftware/KoSpeech
    """
    def __init__(self):
        super(ResnetVADModel, self).__init__()

        self.block1 = ResnetBlock(
            in_channels=1,
            out_channels=32,
            num_kernels1=(8, 5, 3),
            num_kernels2=16,
        )
        self.block2 = ResnetBlock(
            in_channels=32,
            out_channels=64,
            num_kernels1=(8, 5, 3),
            num_kernels2=1,
        )
        self.block3 = ResnetBlock(
            in_channels=64,
            out_channels=128,
            num_kernels1=(8, 5, 3),
            num_kernels2=1,
        )
        self.block4 = ResnetBlock(
            in_channels=128,
            out_channels=128,
            num_kernels1=(8, 5, 3),
            num_kernels2=1,
        )

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(128 * 65, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2048, 2)

    def forward(self, inputs):
        output = self.block1(inputs)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)

        output = self.flat(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output
