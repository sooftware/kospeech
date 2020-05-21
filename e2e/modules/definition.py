import logging
import sys
from e2e.data.label_loader import load_label

# If you haven't read https://sooftware.github.io/End-to-end-Speech-Recognition/notes/Preparation.html
# please read it first before setting definition.py

DATASET_PATH = "/data1/"  # set your data path
SAMPLE_DATASET_PATH = "./data/sample/"
TRAIN_LIST_PATH = "./data/data_list/filter_train_list.csv"   # set your train list
TEST_LIST_PATH = "./data/data_list/filter_test_list.csv"     # set your test list
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

char2id, id2char = load_label('./data/label/aihub_labels.csv', encoding='utf-8')  # 2,040

SOS_token = int(char2id['<s>'])
EOS_token = int(char2id['</s>'])
PAD_token = int(char2id['_'])

train_dict = {'loss': [], 'cer': []}
valid_dict = {'loss': [], 'cer': []}
