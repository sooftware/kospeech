# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset: str = "kspon"
    dataset_path: str = "???"
    transcripts_path: str = "../../../data/transcripts.txt"
    output_unit: str = "character"

    num_epochs: int = 20
    batch_size: int = 32
    save_result_every: int = 1000
    checkpoint_every: int = 5000
    print_every: int = 10
    mode: str = "train"

    num_workers: int = 4
    use_cuda: bool = True

    init_lr_scale: float = 0.01
    final_lr_scale: float = 0.05
    max_grad_norm: int = 400
    weight_decay: float = 1e-05
    reduction: str = "mean"

    seed: int = 777
    resume: bool = False


@dataclass
class ListenAttendSpellTrainConfig(TrainConfig):
    optimizer: str = "adam"
    init_lr: float = 1e-06
    final_lr: float = 1e-06
    peak_lr: float = 1e-04
    warmup_steps: int = 400


@dataclass
class DeepSpeech2TrainConfig(TrainConfig):
    optimizer: str = "adam"
    init_lr: float = 1e-06
    final_lr: float = 1e-06
    peak_lr: float = 1e-04
    warmup_steps: int = 1000


@dataclass
class TransformerTrainConfig(TrainConfig):
    optimizer: str = "adam"
    init_lr: float = 1e-06
    final_lr: float = 1e-06
    peak_lr: float = 1e-07
    warmup_steps: int = 4000
