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

import logging, sys
logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)
from utils.label import load_label
char2index, index2char = load_label('./data/label/train_labels.csv', encoding='utf-8')
test_char2index, test_index2char = load_label('./data/label/test_labels.csv', encoding='utf-8')
SOS_token = int(char2index['<s>'])
EOS_token = int(char2index['</s>'])
PAD_token = int(char2index['_'])
DATASET_PATH = "G:/한국어 음성데이터/KaiSpeech/"
SAMPLE_DATASET_PATH = "./data/sample/"
TRAIN_LIST_PATH = "./data/data_list/train_list.csv"
TEST_LIST_PATH = "./data/data_list/test_list.csv"
SAMPLE_LIST_PATH = "./data/data_list/sample_list.csv"
TARGET_DICT_PATH = "./data/pickle/target_dict.bin"
SAVE_WEIGHT_PATH = "./data/weight_file/epoch%s"
TRAIN_RESULT_PATH = "./data/train_result/train_result.csv"
VALID_RESULT_PATH = "./data/train_result/eval_result.csv"
TRAIN_STEP_RESULT_PATH = "./data/train_result/train_step_result.csv"
TRAIN_DATASET_PICKLE_PATH = "./data/pickle/train_dataset.txt"
VALID_DATASET_PICKLE_PATH = "./data/pickle/valid_dataset.txt"
ENCODING = "cp949"
train_dict = {'loss': [], 'cer': []}
valid_dict = {'loss': [], 'cer': []}