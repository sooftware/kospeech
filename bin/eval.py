# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import sys
import argparse
import warnings
sys.path.append('..')
from kospeech.data.label_loader import load_dataset
from kospeech.data.data_loader import SpectrogramDataset
from kospeech.evaluator.evaluator import Evaluator
from kospeech.utils import check_envirionment, EOS_token, SOS_token
from kospeech.model_builder import load_test_model
from kospeech.opts import build_eval_opts, build_preprocess_opts, print_opts


def inference(opt):
    device = check_envirionment(opt.use_cuda)
    model = load_test_model(opt, device)

    audio_paths, transcripts = load_dataset(opt.transcripts_path)

    testset = SpectrogramDataset(audio_paths=audio_paths, transcripts=transcripts,
                                 sos_id=SOS_token, eos_id=EOS_token,
                                 dataset_path=opt.dataset_path,  opt=opt, spec_augment=False)

    evaluator = Evaluator(testset, opt.batch_size, device, opt.num_workers, opt.print_every, opt.decode, opt.k)
    evaluator.evaluate(model)


def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser(description='KoSpeech')
    parser.add_argument('--mode', type=str, default='eval')

    build_preprocess_opts(parser)
    build_eval_opts(parser)

    return parser


def main():
    warnings.filterwarnings('ignore')
    parser = _get_parser()
    opt = parser.parse_args()
    print_opts(opt, opt.mode)

    inference(opt)


if __name__ == '__main__':
    main()
