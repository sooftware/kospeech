"""
    -*- coding: utf-8 -*-

    @source_code{
      title={Character-unit based End-to-End Korean Speech Recognition},
      author={Soohwan Kim, Seyoung Bae, Cheolhwang Won},
      link={https://github.com/sooftware/End-to-End-Korean-Speech-Recognition},
      year={2020}
    }
"""

import os
import queue
import torch
import warnings
from utils.dataset import SpectrogramDataset
from utils.definition import *
from utils.config import Config
from utils.loader import AudioDataLoader, load_data_list, load_targets
from utils.util import get_distance


def test(model, queue, device):
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

            model.flatten_parameters()
            y_hat, _ = model(inputs, targets, teacher_forcing_ratio=0.0, use_beam_search=True)

            dist, length = get_distance(scripts, y_hat, id2char, char2id, EOS_token)
            total_dist += dist
            total_length += length
            total_sent_num += scripts.size(0)

            #  if time_step % 10 == 0:
            logger.info('cer: {:.2f}'.format(dist / length))

            time_step += 1

    logger.info('test() completed')
    return total_dist / total_length


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    config = Config(batch_size=4)
    cuda = config.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    if device == 'cuda':
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # if you use Multi-GPU, delete this line
        logger.info("device : %s" % torch.cuda.get_device_name(0))
        logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
        logger.info("CUDA version : %s" % torch.version.cuda)
        logger.info("PyTorch version : %s" % torch.__version__)

    model = torch.load('./data/weight_file/epoch4.pt', map_location=torch.device('cpu')).module
    model.listener.device = 'cpu'
    model.speller.device = 'cpu'
    model.set_beam_size(k=3)

    audio_paths, label_paths = load_data_list(data_list_path=SAMPLE_LIST_PATH, dataset_path=SAMPLE_DATASET_PATH)
    target_dict = load_targets(label_paths)

    testset = SpectrogramDataset(
        audio_paths=audio_paths,
        label_paths=label_paths,
        sos_id=SOS_token,
        eos_id=EOS_token,
        target_dict=target_dict,
        config=config,
        use_augment=False
    )

    test_queue = queue.Queue(config.worker_num << 1)
    test_loader = AudioDataLoader(testset, test_queue, config.batch_size, 0)
    test_loader.start()

    cer = test(model, test_queue, device)
    logger.info('20h Test Set CER : %s' % cer)
    test_loader.join()
