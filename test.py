"""
    -*- coding: utf-8 -*-

    @source_code{
      title={Character-unit based End-to-End Korean Speech Recognition},
      author={Soohwan Kim, Seyoung Bae, Cheolhwang Won},
      link={https://github.com/sooftware/End-to-End-Korean-Speech-Recognition},
      year={2020}
    }
"""

import queue
import torch
import warnings
import argparse
from data_loader import SpectrogramDataset, AudioDataLoader, load_data_list
from definition import logger, id2char, EOS_token, SAMPLE_LIST_PATH, SAMPLE_DATASET_PATH, SOS_token, TEST_LIST_PATH, \
    DATASET_PATH
from label_loader import load_targets, load_label
from model.topk_decoder import TopKDecoder
from utils import get_distance


def test(model, queue, device, args):
    """ Test for Model Performance """
    logger.info('test() start')
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    time_step = 0

    model.eval()

    with torch.no_grad():
        while True:
            inputs, targets, input_lengths, target_lengths = queue.get()
            if inputs.shape[0] == 0:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            scripts = targets[:, 1:]

            output = model(inputs, input_lengths, teacher_forcing_ratio=0.0)

            logit = torch.stack(output, dim=1).to(device)
            hypothesis = logit.max(-1)[1]

            dist, length = get_distance(scripts, hypothesis, id2char, EOS_token)
            total_dist += dist
            total_length += length
            total_sent_num += scripts.size(0)

            if time_step % args.print_every == 0:
                logger.info('cer: {:.2f}'.format(dist / length))

            time_step += 1

    logger.info('test() completed')
    return total_dist / total_length


parser = argparse.ArgumentParser(description='End-to-end Speech Recognition')
parser.add_argument('--batch_size', type=int, default=1, help='batch size in performance test (default: 1)')
parser.add_argument('--load_model', action='store_true', default=True)
parser.add_argument('--use_cuda', action='store_true', default=True)
parser.add_argument('--model_path', type=str, default='./data/weight_file/epoch4.pt', help='Location to load models (default: None')
parser.add_argument('--k', type=int, default=5, help='size of beam (default: 5)')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers in dataset loader (default: 1)')
parser.add_argument('--sr', type=int, default=16000, help='sample rate (default: 16000)')
parser.add_argument('--window_size', type=int, default=20, help='Window size for spectrogram (default: 20ms)')
parser.add_argument('--stride', type=int, default=10, help='Window stride for spectrogram (default: 10ms)')
parser.add_argument('--n_mels', type=int, default=80, help='number of mel filter (default: 80)')
parser.add_argument('--normalize', action='store_true', default=True)
parser.add_argument('--del_silence', action='store_true', default=True)
parser.add_argument('--input_reverse', action='store_true', default=True)
parser.add_argument('--print_every', type=int, default=10,
                    help='to determine whether to store training progress every N timesteps (default: 10')
parser.add_argument('--feature_extract_by', type=str, default='librosa',
                    help='which library to use for feature extraction: [librosa, torchaudio] (default: librosa)')
parser.add_argument('--max_len', type=int, default=151, help='maximum characters of sentence (default: 151)')
args = parser.parse_args()


def main():
    warnings.filterwarnings('ignore')
    cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    if str(device) == 'cuda':
        for idx in range(torch.cuda.device_count()):
            logger.info("device : %s" % torch.cuda.get_device_name(idx))
        logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
        logger.info("CUDA version : %s" % torch.version.cuda)
        logger.info("PyTorch version : %s" % torch.__version__)

    model = torch.load(args.model_path)
    model.module.speller.device = device
    model.module.listener.device = device

    topk_decoder = TopKDecoder(model.module.speller, args.k)
    model.module.set_speller(topk_decoder)

    audio_paths, label_paths = load_data_list(data_list_path=TEST_LIST_PATH, dataset_path=DATASET_PATH)
    target_dict = load_targets(label_paths)

    testset = SpectrogramDataset(
        audio_paths=audio_paths,
        label_paths=label_paths,
        sos_id=SOS_token,
        eos_id=EOS_token,
        target_dict=target_dict,
        args=args,
        use_augment=False
    )

    test_queue = queue.Queue(args.num_workers << 1)
    test_loader = AudioDataLoader(testset, test_queue, args.batch_size, 0)
    test_loader.start()

    cer = test(model, test_queue, device, args)
    logger.info('20h Test Set CER : %s' % cer)
    test_loader.join()


if __name__ == '__main__':
    main()
