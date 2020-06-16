import torch
import torch.nn as nn
import pandas as pd
from kospeech.metrics import CharacterErrorRate
from kospeech.model.beam_search import BeamSearchDecoder
from kospeech.model_builder import load_language_model
from kospeech.utils import id2char, EOS_token, logger, label_to_string


class GreedySearch(object):
    """
    Provides some functions : search, save result to csv format.

    Note:
        You can use this class directly and you can use one of the sub classes.
    """
    def __init__(self):
        self.target_list = list()
        self.hypothesis_list = list()
        self.metric = CharacterErrorRate(id2char, EOS_token)
        # self.language_model = load_language_model('lm_path', 'cuda')

    def search(self, model, queue, device, print_every):
        cer = 0
        total_sent_num = 0
        timestep = 0

        model.eval()

        with torch.no_grad():
            while True:
                inputs, scripts, input_lengths, target_lengths = queue.get()
                if inputs.shape[0] == 0:
                    break

                inputs = inputs.to(device)
                scripts = scripts.to(device)
                targets = scripts[:, 1:]

                output, _ = model(inputs, input_lengths, teacher_forcing_ratio=0.0)  # language_model=self.language_model

                logit = torch.stack(output, dim=1).to(device)
                hypothesis = logit.max(-1)[1]

                for idx in range(targets.size(0)):
                    self.target_list.append(label_to_string(scripts[idx], id2char, EOS_token))
                    self.hypothesis_list.append(label_to_string(hypothesis[idx].cpu().detach().numpy(), id2char, EOS_token))

                cer = self.metric(targets, hypothesis)
                total_sent_num += scripts.size(0)

                if timestep % print_every == 0:
                    logger.info('cer: {:.2f}'.format(cer))

                timestep += 1

        return cer

    def save_result(self, save_path):
        results = {
            'original': self.target_list,
            'hypothesis': self.hypothesis_list
        }
        results = pd.DataFrame(results)
        results.to_csv(save_path, index=False, encoding='cp949')


class BeamSearch(GreedySearch):
    """ Provides beam search decoding. """
    def __init__(self, k):
        super(BeamSearch, self).__init__()
        self.k = k

    def search(self, model, queue, device, print_every):
        topk_decoder = BeamSearchDecoder(model.module.speller, self.k)
        if isinstance(model, nn.DataParallel):
            model.module.set_speller(topk_decoder)
        else:
            model.set_speller(topk_decoder)
        return super(BeamSearch, self).search(model, queue, device, print_every)


class EnsembleSearch(GreedySearch):
    """ Provides ensemble search decoding. """
    def __init__(self, method='basic'):
        super(EnsembleSearch, self).__init__()
        self.method = method

    def search(self, models, queue, device, print_every):
        # TODO : IMPLEMENTS ENSEMBLE SEARCH
        return super(EnsembleSearch, self).search(models, queue, device, print_every)
