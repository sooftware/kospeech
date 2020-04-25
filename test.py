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
from model.listenAttendSpell import ListenAttendSpell
from model.listener import Listener
from model.speller import Speller
from package.dataset import CustomDataset
from package.definition import *
from package.config import Config
from package.loader import CustomDataLoader, load_data_list, load_targets
from package.utils import get_distance


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
            target = targets[:, 1:]

            model.flatten_parameters()
            y_hat, _ = model(inputs, targets, teacher_forcing_ratio=0.0, use_beam_search=True)
            dist, length = get_distance(target, y_hat, id2char, EOS_token)
            total_dist += dist
            total_length += length
            total_sent_num += target.size(0)

            if time_step % 10 == 0:
                logger.info('cer: {:.2f}'.format(dist / length))

            time_step += 1

    logger.info('test() completed')

    return total_dist / total_length


if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    logger.info("device : %s" % torch.cuda.get_device_name(0))
    logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
    logger.info("CUDA version : %s" % torch.version.cuda)
    logger.info("PyTorch version : %s" % torch.__version__)

    config = Config()
    cuda = config.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    listener = Listener(
        in_features=80,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout,
        n_layers=config.listener_layer_size,
        bidirectional=config.use_bidirectional,
        rnn_type='gru',
        use_pyramidal=False,
        device=device
    )
    speller = Speller(
        n_class=len(char2id),
        max_length=config.max_len,
        k=1,
        hidden_dim=config.hidden_dim << (1 if config.use_bidirectional else 0),
        sos_id=SOS_token,
        eos_id=EOS_token,
        n_layers=config.speller_layer_size,
        rnn_type='gru',
        dropout_p=config.dropout,
        device=device
    )
    model = ListenAttendSpell(listener, speller)

    load_model = torch.load('weight_path')
    model.set_beam_size(k=5)

    audio_paths, label_paths = load_data_list(data_list_path=SAMPLE_LIST_PATH, dataset_path=SAMPLE_DATASET_PATH)
    target_dict = load_targets(label_paths)

    test_dataset = CustomDataset(
        audio_paths=audio_paths,
        label_paths=label_paths,
        sos_id=SOS_token,
        eos_id=EOS_token,
        target_dict=target_dict,
        input_reverse=config.input_reverse,
        use_augment=False
    )

    test_queue = queue.Queue(config.worker_num << 1)
    test_loader = CustomDataLoader(test_dataset, test_queue, config.batch_size, 0)
    test_loader.start()

    cer = test(model, test_queue, device)
    logger.info('20h Test Set CER : %s' % cer)
    test_loader.join()
