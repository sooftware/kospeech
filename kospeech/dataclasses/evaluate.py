# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class EvalConfig:
    dataset: str = 'kspon'
    dataset_path: str = ''
    transcript_path: str = '../../../data/eval_transcript.txt'
    model_path: str = ''
    output_unit: str = 'character'
    batch_size: int = 32
    num_workers: int = 4
    print_every: int = 20
    decode: str = 'greedy'
    k: int = 3
    use_cuda: bool = True
