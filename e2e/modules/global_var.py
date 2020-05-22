"""
This file provides logger, char2id, id2char, SOS_token, EOS_token, PAD_token.
Global variables are defined in this file.
"""

from e2e.modules.logger import Logger
#from e2e.data_loader.label_loader import load_label

logger = Logger()
# char2id, id2char = load_label('./data/label/aihub_labels.csv', encoding='utf-8')  # char labels

# SOS_token = int(char2id['<s>'])
# EOS_token = int(char2id['</s>'])
# PAD_token = int(char2id['_'])
