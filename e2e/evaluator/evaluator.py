import queue
import torch
from e2e.data.data_loader import AudioDataLoader
from e2e.decode.search import GreedySearch, BeamSearch
from e2e.utils import logger


class Evaluator(object):
    """
    Class to evaluate models with given datasets.

    Args:
        dataset (e2e.data_loader.SpectrogramDataset): dataset for spectrogram & script matching
        batch_size (int): size of batch. recommended batch size is 1.
        device (torch.device): device - 'cuda' or 'cpu'
        num_workers (int): the number of cpu cores used
        print_every (int): to determine whether to store training progress every N timesteps (default: 10)
    """

    def __init__(self, dataset, batch_size=1, device=None, num_workers=1, print_every=100, decode='greedy', k=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.print_every = print_every

        if decode == 'greedy':
            self.decoder = GreedySearch()
            self.save_result_path = './data/train_result/greedy_search.csv'
        elif decode == 'beam':
            assert k is not None, "In beam search mode, k should has value."
            self.decoder = BeamSearch(k)
            self.save_result_path = './data/train_result/beam_search.csv'
        else:
            raise ValueError("Unsupported decode : {0}".format(decode))

    def evaluate(self, model):
        """ Evaluate a model on given dataset and return performance. """
        logger.info('evaluate() start')

        eval_queue = queue.Queue(self.num_workers << 1)
        eval_loader = AudioDataLoader(self.dataset, eval_queue, self.batch_size, 0)
        eval_loader.start()

        cer = self.decoder.search(model, eval_queue, self.device, self.print_every)
        self.decoder.save_result(self.save_result_path)
        logger.info('Evaluate CER: %s' % cer)
        logger.info('evaluate() completed')
        eval_loader.join()
