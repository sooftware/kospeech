# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import queue
import torch
from kospeech.utils import logger
from kospeech.data.data_loader import AudioDataLoader
from kospeech.decode.search import (
    GreedySearch,
    BeamSearch
)


class Evaluator(object):
    """
    Class to evaluate models with given datasets.

    Args:
        dataset (kospeech.data.data_loader.SpectrogramDataset): dataset for spectrogram & script matching
        batch_size (int): size of batch. recommended batch size is 1.
        device (torch.device): device - 'cuda' or 'cpu'
        num_workers (int): the number of cpu cores used
        print_every (int): to determine whether to store training progress every N timesteps (default: 10)
    """

    def __init__(self, dataset, batch_size=1, device=None, num_workers=1, print_every=100, decode='greedy', beam_size=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.print_every = print_every

        if decode == 'greedy':
            self.decoder = GreedySearch()

        elif decode == 'beam':
            assert beam_size > 1, "beam_size should be greater than 1. You can choose `greedy` search"
            self.decoder = BeamSearch(beam_size)

        else:
            raise ValueError("Unsupported decode : {0}".format(decode))

    def evaluate(self, model):
        """ Evaluate a model on given dataset and return performance. """
        logger.info('evaluate() start')

        eval_queue = queue.Queue(self.num_workers << 1)
        eval_loader = AudioDataLoader(self.dataset, eval_queue, self.batch_size, 0)
        eval_loader.start()

        cer = self.decoder.search(model, eval_queue, self.device, self.print_every)
        self.decoder.save_result('../data/train_result/%s.csv' % type(self.decoder).__name__)

        logger.info('Evaluate CER: %s' % cer)
        logger.info('evaluate() completed')
        eval_loader.join()
