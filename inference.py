import argparse
import warnings
from e2e.evaluator.evaluator import Evaluator
from e2e.modules.definition import TEST_LIST_PATH, DATASET_PATH
from e2e.modules.model_builder import load_test_model
from e2e.modules.opts import inference_opts, preprocess_opts, print_opts
from e2e.modules.utils import check_envirionment


def evaluate(opt):
    device = check_envirionment(opt)
    model = load_test_model(opt, device, use_beamsearch=opt.use_beam_search)

    evaluator = Evaluator(batch_size=opt.batch_size, device=device)
    evaluator.evaluate(model, opt, TEST_LIST_PATH, DATASET_PATH)


def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser(description='End-to-end Speech Recognition')
    parser.add_argument('--mode', type=str, default='test')

    preprocess_opts(parser)
    inference_opts(parser)

    return parser


def main():
    warnings.filterwarnings('ignore')
    parser = _get_parser()
    opt = parser.parse_args()
    print_opts(opt, opt.mode)

    evaluate(opt)


if __name__ == '__main__':
    main()
