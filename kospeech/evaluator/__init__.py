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

from dataclasses import dataclass


@dataclass
class EvalConfig:
    dataset: str = 'kspon'
    dataset_path: str = ''
    transcripts_path: str = '../../../data/eval_transcript.txt'
    model_path: str = ''
    output_unit: str = 'character'
    batch_size: int = 32
    num_workers: int = 4
    print_every: int = 20
    decode: str = 'greedy'
    k: int = 3
    use_cuda: bool = True