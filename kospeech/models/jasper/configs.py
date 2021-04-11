# Copyright (c) 2021, Soohwan Kim. All rights reserved.
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


class Jasper10x5Config:
    def __init__(self, num_classes: int, num_blocks: int, num_sub_blocks: int) -> None:
        super(Jasper10x5Config, self).__init__()
        self.num_blocks = num_blocks
        self.num_sub_blocks = num_sub_blocks
        self.preprocess_block = {
            'in_channels': 80,
            'out_channels': 256,
            'kernel_size': 11,
            'stride': 2,
            'dilation': 1,
            'dropout_p': 0.2,
        }
        self.block = {
            'in_channels': (256, 256, 256, 384, 384, 512, 512, 640, 640, 768),
            'out_channels': (256, 256, 384, 384, 512, 512, 640, 640, 768, 768),
            'kernel_size': (11, 11, 13, 13, 17, 17, 21, 21, 25, 25),
            'dilation': [1] * 10,
            'dropout_p': (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3),
        }
        self.postprocess_block = {
            'in_channels': (768, 896, 1024),
            'out_channels': (896, 1024, num_classes),
            'kernel_size': (29, 1, 1),
            'dilation': (2, 1, 1),
            'dropout_p': (0.4, 0.4, 0.0),
        }


class Jasper5x3Config:
    def __init__(self, num_classes: int, num_blocks: int, num_sub_blocks: int) -> None:
        super(Jasper5x3Config, self).__init__()
        self.num_blocks = num_blocks
        self.num_sub_blocks = num_sub_blocks
        self.preprocess_block = {
            'in_channels': 80,
            'out_channels': 256,
            'kernel_size': 11,
            'stride': 2,
            'dilation': 1,
            'dropout_p': 0.2,
        }
        self.block = {
            'in_channels': (256, 256, 384, 512, 640),
            'out_channels': (256, 384, 512, 640, 768),
            'kernel_size': (11, 13, 17, 21, 25),
            'dilation': [1] * 5,
            'dropout_p': (0.2, 0.2, 0.2, 0.3, 0.3),
        }
        self.postprocess_block = {
            'in_channels': (768, 896, 1024),
            'out_channels': (896, 1024, num_classes),
            'kernel_size': (29, 1, 1),
            'dilation': (2, 1, 1),
            'dropout_p': (0.4, 0.4, 0.0),
        }
