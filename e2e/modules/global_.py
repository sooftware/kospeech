"""
This file provides logger, char2id, id2char, SOS_token, EOS_token, PAD_token.
Global variables are defined in this file.
"""
import torch
import platform
from e2e.modules.logger import Logger
from e2e.data_loader.label_loader import load_label

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
