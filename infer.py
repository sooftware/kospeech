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
from e2e.data_loader.data_loader import load_data_list, SpectrogramDataset
from e2e.data_loader.label_loader import load_targets
from e2e.evaluator.evaluator import Evaluator
from e2e.modules.global_var import EOS_token, SOS_token
from e2e.modules.model_builder import load_test_model
from e2e.modules.opts import inference_opts, preprocess_opts, print_opts
from e2e.modules.utils import check_envirionment


def inference(opt):
    device = check_envirionment(opt.use_cuda)
    model = load_test_model(opt, device, use_beamsearch=opt.use_beam_search)

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

    evaluator = Evaluator(testset, opt.batch_size, device, opt.num_workers, opt.print_every)
    evaluator.evaluate(model)


def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser(description='End-to-end Speech Recognition')
    parser.add_argument('--mode', type=str, default='infer')

    preprocess_opts(parser)
    inference_opts(parser)

    return parser


def main():
    warnings.filterwarnings('ignore')
    parser = _get_parser()
    opt = parser.parse_args()
    print_opts(opt, opt.mode)

    inference(opt)


if __name__ == '__main__':
    main()
