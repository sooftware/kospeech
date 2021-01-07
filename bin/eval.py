# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import hydra
import warnings
sys.path.append('..')

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from kospeech.dataclass import EvalConfig, FilterBankConfig
from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.vocabs.librispeech import LibriSpeechVocabulary
from kospeech.data.label_loader import load_dataset
from kospeech.data.data_loader import SpectrogramDataset
from kospeech.evaluator.evaluator import Evaluator
from kospeech.utils import check_envirionment, logger
from kospeech.model_builder import load_test_model


def inference(config: DictConfig):
    device = check_envirionment(config.eval.use_cuda)
    model = load_test_model(config.eval, device)

    if config.eval.dataset == 'kspon':
        vocab = KsponSpeechVocabulary(
            f'../../../data/vocab/aihub_{config.eval.output_unit}_vocabs.csv', output_unit=config.eval.output_unit
        )
    elif config.eval.dataset == 'libri':
        vocab = LibriSpeechVocabulary('../../../data/vocab/tokenizer.vocab', 'data/vocab/tokenizer.model')
    else:
        raise ValueError("Unsupported Dataset : {0}".format(config.eval.dataset))

    audio_paths, transcripts = load_dataset(config.eval.transcripts_path)

    testset = SpectrogramDataset(audio_paths=audio_paths, transcripts=transcripts,
                                 sos_id=vocab.sos_id, eos_id=vocab.eos_id,
                                 dataset_path=config.eval.dataset_path,  config=config, spec_augment=False)

    evaluator = Evaluator(
        dataset=testset,
        vocab=vocab,
        batch_size=config.eval.batch_size,
        device=device,
        num_workers=config.eval.num_workers,
        print_every=config.eval.print_every,
        decode=config.eval.decode,
        beam_size=config.eval.k
    )
    evaluator.evaluate(model)


cs = ConfigStore.instance()
cs.store(group="eval", name="default", node=EvalConfig, package="eval")
cs.store(group="audio", name="fbank", node=FilterBankConfig, package="audio")


@hydra.main(config_path=os.path.join('..', "configs"), config_name="eval")
def main(config: DictConfig) -> None:
    warnings.filterwarnings('ignore')
    logger.info(OmegaConf.to_yaml(config))
    inference(config)


if __name__ == '__main__':
    main()
