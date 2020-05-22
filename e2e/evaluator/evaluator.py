import queue
import torch
from e2e.data_loader.data_loader import AudioDataLoader
from e2e.modules.utils import get_distance
#from e2e.modules.global_var import id2char, EOS_token, logger


class Evaluator:
    """
    Class to evaluate models with given datasets.

    Args:
        dataset (e2e.data_loader.SpectrogramDataset): dataset for spectrogram & script matching
        batch_size (int): size of batch. recommended batch size is 1.
        device (torch.device): device - 'cuda' or 'cpu'
        num_workers (int): the number of cpu cores used
        print_every (int): to determine whether to store training progress every N timesteps (default: 10)
    """

    def __init__(self, dataset, batch_size=1, device=None, num_workers=1, print_every=100):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.print_every = print_every

    def evaluate(self, model):
        """ Evaluate a model on given dataset and return performance. """
        eval_queue = queue.Queue(self.num_workers << 1)
        eval_loader = AudioDataLoader(self.dataset, eval_queue, self.batch_size, 0)
        eval_loader.start()

        cer = self.predict(model, eval_queue)
        logger.info('Evaluate CER: %s' % cer)
        eval_loader.join()

    def predict(self, model, queue):
        """ Make prediction given testset as input. """
        logger.info('evaluate() start')
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

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                scripts = targets[:, 1:]

                output = model(inputs, input_lengths, teacher_forcing_ratio=0.0)

                logit = torch.stack(output, dim=1).to(self.device)
                hypothesis = logit.max(-1)[1]

                dist, length = get_distance(scripts, hypothesis, id2char, EOS_token)
                total_dist += dist
                total_length += length
                total_sent_num += scripts.size(0)

                if time_step % self.print_every == 0:
                    logger.info('cer: {:.2f}'.format(dist / length))

                time_step += 1

        logger.info('evaluate() completed')
        return total_dist / total_length
