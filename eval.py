"""
-*- coding: utf-8 -*-

@source_code{
  title={End-to-end Speech Recognition},
  author={Soohwan Kim, Seyoung Bae, Cheolhwang Won},
  link={https://github.com/sooftware/End-to-end-Speech-Recognition},
  year={2020}
}
"""
import argparse
import warnings
from e2e.data.data_loader import load_data_list, SpectrogramDataset
from e2e.data.label_loader import load_targets
from e2e.solver.evaluator import Evaluator
from e2e.utils import check_envirionment, EOS_token, SOS_token
from e2e.model_builder import load_test_model
from e2e.opts import build_eval_opts, build_preprocess_opts, print_opts


def inference(opt):
    device = check_envirionment(opt.use_cuda)
    model = load_test_model(opt, device)

    audio_paths, script_paths = load_data_list(opt.data_list_path, opt.dataset_path)
    target_dict = load_targets(script_paths)

    testset = SpectrogramDataset(
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
    parser = argparse.ArgumentParser(description='End-to-end Speech Recognition')
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
