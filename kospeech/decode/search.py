import torch
import torch.nn as nn
import pandas as pd
from queue import Queue
from kospeech.metrics import CharacterErrorRate
from kospeech.models.acoustic.seq2seq.decoder import SpeechTopKDecoder
from kospeech.models.acoustic.seq2seq.seq2seq import SpeechSeq2seq
from kospeech.utils import id2char, EOS_token, logger, label_to_string


class GreedySearch(object):
    """ Provides greedy search and save result to csv format """
    def __init__(self):
        self.target_list = list()
        self.predict_list = list()
        self.metric = CharacterErrorRate(id2char, EOS_token)

    def search(self, model: nn.Module, queue: Queue, device: str, print_every: int) -> float:
        cer = 0
        total_sent_num = 0
        timestep = 0

        model.eval()

        with torch.no_grad():
            while True:
                inputs, targets, input_lengths, target_lengths = queue.get()
                if inputs.shape[0] == 0:
                    break

                inputs = inputs.to(device)
                targets = targets.to(device)

                output = model(inputs, input_lengths, teacher_forcing_ratio=0.0, return_decode_dict=False)
                logit = torch.stack(output, dim=1).to(device)
                pred = logit.max(-1)[1]

                for idx in range(targets.size(0)):
                    self.target_list.append(label_to_string(targets[idx], id2char, EOS_token))
                    self.predict_list.append(label_to_string(pred[idx].cpu().detach().numpy(), id2char, EOS_token))

                cer = self.metric(targets[:, 1:], pred)
                total_sent_num += targets.size(0)

                if timestep % print_every == 0:
                    logger.info('cer: {:.2f}'.format(cer))

                timestep += 1

        return cer

    def save_result(self, save_path: str) -> None:
        results = {
            'targets': self.target_list,
            'predictions': self.predict_list
        }
        results = pd.DataFrame(results)
        results.to_csv(save_path, index=False, encoding='cp949')


class BeamSearch(GreedySearch):
    """ Provides beam search decoding. """
    def __init__(self, k):
        super(BeamSearch, self).__init__()
        self.k = k

    def search(self, model: SpeechSeq2seq, queue: Queue, device: str, print_every: int) -> float:
        if isinstance(model, nn.DataParallel):
            topk_decoder = SpeechTopKDecoder(model.module.decoder, self.k)
            model.module.set_decoder(topk_decoder)
        else:
            topk_decoder = SpeechTopKDecoder(model.decoder, self.k)
            model.set_decoder(topk_decoder)
        return super(BeamSearch, self).search(model, queue, device, print_every)
