"""
Copyright 2020- Kai.Lib
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import logging
import sys
import queue
import torch.nn as nn
import torch.optim as optim
import random
import torch
import time
import math
import pandas as pd
import threading
import Levenshtein as Lev
import csv
from label.label_func import load_label

char2index, index2char = load_label('./label/kai_labels.csv')
SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']
DATASET_PATH = "E:/한국어 음성데이터/KaiSpeech/"
logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)
TRAIN_LIST_PATH = "./data/train_list.csv"
TEST_LIST_PATH = "./data/test_list.csv"