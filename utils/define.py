"""
Copyright 2020- Kai.Lib

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# If you haven't read https://sh951011.github.io/Korean-Speech-Recognition/notes/Preparation.html
# please read it first before setting define.py

DATASET_PATH = "G:/한국어 음성데이터/KaiSpeech/" # set by your data path
SAMPLE_DATASET_PATH = "./data/sample/"
TRAIN_LIST_PATH = "./data/data_list/train_list.csv"
TEST_LIST_PATH = "./data/data_list/test_list.csv"
SAMPLE_LIST_PATH = "./data/data_list/sample_list.csv"
TARGET_DICT_PATH = "./data/pickle/target_dict.bin"
TRAIN_RESULT_PATH = "./data/train_result/train_result.csv"
VALID_RESULT_PATH = "./data/train_result/eval_result.csv"
TRAIN_STEP_RESULT_PATH = "./data/train_result/train_step_result.csv"
import logging, sys
logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)
from utils.load import load_label
char2id, id2char = load_label('./data/label/train_labels.csv', encoding='utf-8')
# => char2id, id2char = load_label('./data/label/test_labels.csv', encoding='utf-8')
SOS_TOKEN = int(char2id['<s>'])
EOS_TOKEN = int(char2id['</s>'])
PAD_TOKEN = int(char2id['_'])
train_dict = {'loss': [], 'cer': []}
valid_dict = {'loss': [], 'cer': []}