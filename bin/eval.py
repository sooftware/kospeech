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

import os
import hydra
import warnings
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from kospeech.evaluator import EvalConfig
from kospeech.data.audio import FilterBankConfig
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
        beam_size=config.eval.k,
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
