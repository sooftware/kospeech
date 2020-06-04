import torch
import pandas as pd
from e2e.model.topk_decoder import TopKDecoder
from e2e.metric import CharacterErrorRate
from e2e.utils import id2char, EOS_token, logger, label_to_string, char2id


class Search(object):
    """
    Provides inteface of search algorithm.

    Note:
        Do not use this class directly, use one of the sub classes.

    Method:
        - **search()**: abstract method. you have to override this method.
        - **save_result()**: abstract method. you have to override this method.
    """
    def __init__(self):
        self.target_list = list()
        self.hypothesis_list = list()
        self.metric = CharacterErrorRate(id2char, EOS_token, ignore_id=char2id(' '))

    def search(self, model, queue, device, print_every):
        raise NotImplementedError

    def save_result(self, save_path):
        results = {
            'original': self.target_list,
            'hypothesis': self.hypothesis_list
        }
        results = pd.DataFrame(results)
        results.to_csv(save_path, index=False, encoding='utf-8')


class GreedySearch(Search):

    def __init__(self):
        super(GreedySearch, self).__init__()

    def search(self, model, queue, device, print_every):
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
                scripts = scripts.to(device)
                targets = scripts[:, 1:]

                output, _ = model(inputs, input_lengths, teacher_forcing_ratio=0.0)

                logit = torch.stack(output, dim=1).to(device)
                hypothesis = logit.max(-1)[1]

                for idx in range(targets.size(0)):
                    self.target_list.append(label_to_string(scripts[idx]))
                    self.hypothesis_list.append(label_to_string(hypothesis[idx]))

                cer = self.metric(targets, hypothesis)
                total_sent_num += scripts.size(0)

                if timestep % print_every == 0:
                    logger.info('cer: {:.2f}'.format(cer))

                timestep += 1

        return cer


class BeamSearch(Search):
    def __init__(self, k):
        super(BeamSearch, self).__init__()
        self.k = k

    def search(self, model, queue, device, print_every):
        topk_decoder = TopKDecoder(model.module.speller, self.k)
        try:
            model.module.set_speller(topk_decoder)
        except AttributeError:
            model.set_speller(topk_decoder)

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
                scripts = scripts.to(device)
                targets = scripts[:, 1:]

                output, _ = model(inputs, input_lengths, teacher_forcing_ratio=0.0)

                logit = torch.stack(output, dim=1).to(device)
                hypothesis = logit.max(-1)[1]

                for idx in range(targets.size(0)):
                    self.target_list.append(label_to_string(scripts[idx]))
                    self.hypothesis_list.append(label_to_string(hypothesis[idx]))

                cer = self.metric(targets, hypothesis)
                total_sent_num += scripts.size(0)

                if timestep % print_every == 0:
                    logger.info('cer: {:.2f}'.format(cer))

                timestep += 1

        return cer


class EnsembleSearch(Search):

    def __init__(self, method='basic'):
        super(EnsembleSearch, self).__init__()
        self.method = method

    def search(self, ensemble, queue, device, print_every):
        cer = 0
        total_sent_num = 0
        timestep = 0

        with torch.no_grad():
            while True:
                inputs, targets, input_lengths, target_lengths = queue.get()
                if inputs.shape[0] == 0:
                    break

                inputs = inputs.to(device)
                scripts = scripts.to(device)
                targets = scripts[:, 1:]

                output = ensemble(inputs, input_lengths)

                logit = torch.stack(output, dim=1).to(device)
                hypothesis = logit.max(-1)[1]

                for idx in range(targets.size(0)):
                    self.target_list.append(label_to_string(scripts[idx]))
                    self.hypothesis_list.append(label_to_string(hypothesis[idx]))

                cer = self.metric(targets, hypothesis)
                total_sent_num += scripts.size(0)

                if timestep % print_every == 0:
                    logger.info('cer: {:.2f}'.format(cer))

                timestep += 1

        return cer
