import os
import queue
from data.baseDataset import BaseDataset
from definition import *
import torch
import pickle
import torch.nn as nn
from hyperParams import HyperParams
from loader.baseLoader import BaseDataLoader
from loader.loader import load_data_list
from train.evaluate import evaluate

if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    logger.info("device : %s" % torch.cuda.get_device_name(0))
    logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
    logger.info("CUDA version : %s" % (torch.version.cuda))
    logger.info("PyTorch version : %s" % (torch.__version__))
    device = torch.device('cuda')

    hparams = HyperParams()
    hparams.log_hparams()

    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

    def load_model(filepath):
        logger.info("Load model..")
        model = torch.load(filepath)
        model.eval()
        logger.info("Load model Succesfuuly completely !!")

    model = load_model("./weight_file/epoch2.pt")

    audio_paths, label_paths = load_data_list(data_list_path=TEST_LIST_PATH, dataset_path=DATASET_PATH)

    logger.info("load all target_dict using pickle")
    with open("./pickle/target_dict_test.txt", "rb") as f:
        target_dict = pickle.load(f)
    logger.info("load all target_dict using pickle complete !!")

    logger.info('start')

    test_dataset = BaseDataset(audio_paths=audio_paths[:],
                               label_paths=label_paths[:],
                               bos_id=SOS_token, eos_id=EOS_token, target_dict=target_dict,
                               input_reverse=hparams.input_reverse, use_augmentation=False)

    test_queue = queue.Queue(hparams.worker_num * 2)
    test_loader = BaseDataLoader(test_dataset, test_queue, hparams.batch_size, 0)
    test_loader.start()

    test_loss, test_cer = evaluate(model, test_queue, criterion, device)
    logger.info('200h Test Set CER : %s' % test_cer)