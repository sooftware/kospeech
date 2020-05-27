import Levenshtein as Lev
from e2e.modules.global_ import logger


def label_to_string(labels, id2char, eos_id):
    """
    Converts label to string (number => Hangeul)

    Args:
        labels (numpy.ndarray): number label
        id2char (dict): id2char[id] = ch
        eos_id (int): identification of <end of sequence>

    Returns: sentence
        - **sentence** (str or list): symbol of labels
    """
    if len(labels.shape) == 1:
        sentence = str()
        for label in labels:
            if label.item() == eos_id:
                break
            sentence += id2char[label.item()]
        return sentence

    elif len(labels.shape) == 2:
        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == eos_id:
                    break
                sentence += id2char[label.item()]
            sentences.append(sentence)
        return sentences

    else:
        logger.info("Unsupported shape : {0}".format(str(labels.shape)))


class CharacterErrorRater(object):

    def __init__(self, id2char, eos_id):
        self.total_dist = 0.0
        self.total_length = 0.0
        self.id2char = id2char
        self.eos_id = eos_id

    def calc_error_rate(self, targets, hypothesis):
        dist, length = self._get_distance(targets, hypothesis)
        self.total_dist += dist
        self.total_length += length
        return self.total_dist / self.total_length

    @staticmethod
    def _char_distance(target, y_hat):
        """
        Calculating charater distance between target & y_hat

        Args:
            target: sequence of target
            y_hat: sequence of y_Hat

        Returns: distance, length
            - **dist**: distance between target & y_hat
            - **length**: length of target sequence
        """
        target = target.replace(' ', '')
        y_hat = y_hat.replace(' ', '')

        dist = Lev.distance(y_hat, target)
        length = len(target.replace(' ', ''))

        return dist, length

    def _get_distance(self, targets, y_hats):
        """
        Provides total character distance between targets & y_hats

        Args:
            targets (torch.Tensor): set of ground truth
            y_hats (torch.Tensor): predicted y values (y_hat) by the model

        Returns: total_dist, total_length
            - **total_dist**: total distance between targets & y_hats
            - **total_length**: total length of targets sequence
        """
        total_dist = 0
        total_length = 0

        for (target, y_hat) in zip(targets, y_hats):
            script = label_to_string(target, self.id2char, self.eos_id)
            pred = label_to_string(y_hat, self.id2char, self.eos_id)

            dist, length = self._char_distance(script, pred)

            total_dist += dist
            total_length += length

        return total_dist, total_length
