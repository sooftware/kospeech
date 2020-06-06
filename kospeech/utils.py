import sys
import logging
import torch
import platform
from kospeech.data.label_loader import load_label


class Logger(object):
    """
    Print log message in format.
    FORMAT: [%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s
    """
    def __init__(self):
        self.logger = logging.getLogger('root')
        FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
        self.logger.setLevel(logging.INFO)

    def info(self, message=''):
        """ Print log message for information """
        self.logger.info(message)

    def debug(self, message=''):
        """ Print log message for debugging """
        self.logger.debug(message)


logger = Logger()
char2id, id2char = load_label('./data/label/aihub_labels.csv', encoding='utf-8')  # char labels

SOS_token = int(char2id['<s>'])
EOS_token = int(char2id['</s>'])
PAD_token = int(char2id['_'])


def check_envirionment(use_cuda):
    """
    Check execution envirionment.
    OS, Processor, CUDA version, Pytorch version, ... etc.
    """
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    logger.info("Operating System : %s %s" % (platform.system(), platform.release()))
    logger.info("Processor : %s" % platform.processor())

    if str(device) == 'cuda':
        for idx in range(torch.cuda.device_count()):
            logger.info("device : %s" % torch.cuda.get_device_name(idx))
        logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
        logger.info("CUDA version : %s" % torch.version.cuda)
        logger.info("PyTorch version : %s" % torch.__version__)

    else:
        logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
        logger.info("PyTorch version : %s" % torch.__version__)

    return device


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
