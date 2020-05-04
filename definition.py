import logging
import sys
import csv
from label_loader import load_label

# If you haven't read https://sh951011.github.io/End-to-End-Korean-Speech-Recognition/notes/Preparation.html
# please read it first before setting define.py

DATASET_PATH = "/data1/"  # set by your data path
SAMPLE_DATASET_PATH = "./data/sample/"
TRAIN_LIST_PATH = "./data/data_list/train_list.csv"
TEST_LIST_PATH = "./data/data_list/test_list.csv"
SAMPLE_LIST_PATH = "./data/data_list/sample_list.csv"
DEBUG_LIST_PATH = "./data/data_list/debug_list.csv"
TARGET_DICT_PATH = "./data/pickle/new_target_dict.bin"
TRAIN_RESULT_PATH = "./data/train_result/train_result.csv"
VALID_RESULT_PATH = "./data/train_result/eval_result.csv"
TRAIN_STEP_RESULT_PATH = "./data/train_result/train_step_result.csv"
logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)
char2id, id2char = load_label('./data/label/train_labels.csv', encoding='utf-8')  # 2,040
# if you want to use total character label
# change => char2id, id2char = load_label('./data/label/test_labels.csv', encoding='utf-8') # 2,337
SOS_token = int(char2id['<s>'])
EOS_token = int(char2id['</s>'])
PAD_token = int(char2id['_'])
train_dict = {'loss': [], 'cer': []}
valid_dict = {'loss': [], 'cer': []}
