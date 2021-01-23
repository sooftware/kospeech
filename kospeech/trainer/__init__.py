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
from kospeech.trainer.supervised_trainer import SupervisedTrainer


@dataclass
class TrainConfig:
    dataset: str = "kspon"
    dataset_path: str = "/home/sanghoon/KoSpeech/dataset/kspon/original/"
    transcripts_path: str = "/home/sanghoon/KoSpeech/data/transcripts/transcripts.txt"
    output_unit: str = "character"

    batch_size: int = 16
    save_result_every: int = 1000
    checkpoint_every: int = 5000
    print_every: int = 10
    mode: str = "train"

    num_workers: int = 40
    use_cuda: bool = True

    init_lr_scale: float = 0.01
    final_lr_scale: float = 0.05
    max_grad_norm: int = 400
    weight_decay: float = 1e-05

    seed: int = 777
    resume: bool = False


@dataclass
class ListenAttendSpellTrainConfig(TrainConfig):
    optimizer: str = "adam"
    init_lr: float = 1e-06
    final_lr: float = 1e-06
    peak_lr: float = 1e-04
    warmup_steps: int = 400
    num_epochs: int = 20
    reduction: str = "mean"
    label_smoothing: float = 0.1


@dataclass
class DeepSpeech2TrainConfig(TrainConfig):
    optimizer: str = "adam"
    init_lr: float = 1e-06
    final_lr: float = 1e-06
    peak_lr: float = 1e-04
    warmup_steps: int = 1000
    num_epochs: int = 70
    reduction: str = "mean"


@dataclass
class TransformerTrainConfig(TrainConfig):
    optimizer: str = "adam"
    init_lr: float = 1e-06
    final_lr: float = 1e-06
    peak_lr: float = 1e-04
    warmup_steps: int = 4000
    num_epochs: int = 40
    reduction: str = "mean"
    label_smoothing: float = 0.1


@dataclass
class JasperTrainConfig(TrainConfig):
    optimizer: str = "novograd"
    reduction: str = "sum"
    init_lr: float = 1e-3
    final_lr: float = 1e-4
    peak_lr: float = 1e-3
    weight_decay: float = 1e-3
    warmup_steps: int = 0
    num_epochs: int = 10
