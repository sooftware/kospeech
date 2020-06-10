"""
-*- coding: utf-8 -*-

@github{
  title = {KoSpeech},
  author = {Soohwan Kim, Seyoung Bae, Cheolhwang Won},
  link = {https://github.com/sooftware/KoSpeech},
  year = {2020}
}
"""
import sys
import argparse
import warnings
sys.path.append('..')
from kospeech.data.data_loader import load_data_list, SpeechDataset
from kospeech.data.label_loader import load_targets
from kospeech.evaluator.evaluator import Evaluator
from kospeech.utils import check_envirionment, EOS_token, SOS_token
from kospeech.model_builder import load_test_model
from kospeech.opts import build_eval_opts, build_preprocess_opts, print_opts


def inference(opt):
    device = check_envirionment(opt.use_cuda)
    model = load_test_model(opt, device)

    audio_paths, script_paths = load_data_list(opt.data_list_path, opt.dataset_path)
    target_dict = load_targets(script_paths)

    testset = SpeechDataset(
        audio_paths=audio_paths,
        script_paths=script_paths,
        sos_id=SOS_token,
        eos_id=EOS_token,
        target_dict=target_dict,
        opt=opt,
        spec_augment=False,
        noise_augment=False
    )

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
