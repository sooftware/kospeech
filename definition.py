from loader.label_loader import load_label
import logging
import sys

char2index, index2char = load_label('./kai_labels.csv')
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