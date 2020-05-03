"""
    -*- coding: utf-8 -*-

    @source_code{
      title={Character-unit based End-to-End Korean Speech Recognition},
      author={Soohwan Kim, Seyoung Bae, Cheolhwang Won},
      link={https://github.com/sooftware/End-to-End-Korean-Speech-Recognition},
      year={2020}
    }
"""

import queue
import torch.nn as nn
import torch.optim as optim
import random
import torch
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.speller import Speller
from model.listener import Listener
from model.listenAttendSpell import ListenAttendSpell
from utils.dataset import split_dataset
from utils.definition import *
from utils.evaluator import evaluate
from utils.args import Arguments
from utils.loader import load_data_list, load_targets, load_pickle, MultiLoader, AudioDataLoader
from utils.loss import LabelSmoothingLoss
from utils.trainer import supervised_train
from utils.util import save_epoch_result


if __name__ == '__main__':
    args = Arguments(
        use_bidirectional=True,
        input_reverse=True,
        use_augment=True,
        use_pickle=False,
        use_cuda=True,
        augment_num=1,
        hidden_dim=256,
        dropout=0.3,
        num_head=4,
        attn_dim=128,
        label_smoothing=0.1,
        listener_layer_size=5,
        speller_layer_size=3,
        rnn_type='gru',
        batch_size=32,
        worker_num=1,
        max_epochs=20,
        lr=3e-4,
        teacher_forcing_ratio=0.99,
        valid_ratio=0.01,
        sr=16000,
        window_size=20,
        stride=10,
        n_mels=80,
        normalize=True,
        del_silence=True,
        feature_extract_by='librosa',  # you can choose librosa or torchaudio
        time_mask_para=50,
        freq_mask_para=12,
        time_mask_num=2,
        freq_mask_num=2,
        save_result_every=1000,
        save_model_every=10000,
        print_every=10,
        seed=7,
        max_len=151,
        load_model=False,
        model_path=None,
        run_by_sample=True
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    if str(device) == 'cuda':
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # if you use Multi-GPU, delete this line
        logger.info("device : %s" % torch.cuda.get_device_name(0))
        logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
        logger.info("CUDA version : %s" % torch.version.cuda)
        logger.info("PyTorch version : %s" % torch.__version__)

    if args.load_model:
        model = torch.load(args.model_path).to(device)
    else:
        listener = Listener(
            in_features=args.n_mels,
            hidden_dim=args.hidden_dim,
            dropout_p=args.dropout,
            num_layers=args.listener_layer_size,
            bidirectional=args.use_bidirectional,
            rnn_type=args.rnn_type,
            device=device
        )
        speller = Speller(
            num_class=len(char2id),
            max_length=args.max_len,
            k=5,
            hidden_dim=args.hidden_dim << (1 if args.use_bidirectional else 0),
            sos_id=SOS_token,
            eos_id=EOS_token,
            num_layers=args.speller_layer_size,
            rnn_type=args.rnn_type,
            dropout_p=args.dropout,
            num_head=args.num_head,
            attn_dim=args.attn_dim,
            device=device
        )
        model = ListenAttendSpell(listener, speller)
        model.flatten_parameters()
        model = nn.DataParallel(model).to(device)

        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)

    optimizer = optim.Adam(model.module.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)

    if args.label_smoothing == 0.0:
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)
    else:
        criterion = LabelSmoothingLoss(len(char2id), PAD_token, args.label_smoothing, dim=-1).to(device)

    if args.run_by_sample:
        audio_paths, label_paths = load_data_list(data_list_path=SAMPLE_LIST_PATH, dataset_path=SAMPLE_DATASET_PATH)
    else:
        audio_paths, label_paths = load_data_list(data_list_path=TRAIN_LIST_PATH, dataset_path=DATASET_PATH)

    if args.use_pickle:
        target_dict = load_pickle(TARGET_DICT_PATH, "load all target_dict using pickle complete !!")
    else:
        target_dict = load_targets(label_paths)

    total_time_step, trainset_list, validset = split_dataset(
        args=args,
        audio_paths=audio_paths,
        label_paths=label_paths,
        valid_ratio=args.valid_ratio,
        target_dict=target_dict,
    )

    logger.info('start')
    train_begin = time.time()

    for epoch in range(args.max_epochs):
        train_queue = queue.Queue(args.worker_num << 1)
        for trainset in trainset_list:
            trainset.shuffle()

        # Training
        train_loader = MultiLoader(trainset_list, train_queue, args.batch_size, args.worker_num)
        train_loader.start()
        train_loss, train_cer = supervised_train(
            model=model,
            total_time_step=total_time_step,
            args=args,
            queue=train_queue,
            criterion=criterion,
            epoch=epoch,
            optimizer=optimizer,
            device=device,
            train_begin=train_begin,
            worker_num=args.worker_num,
            teacher_forcing_ratio=args.teacher_forcing_ratio
        )
        train_loader.join()

        torch.save(model, "./data/weight_file/epoch%s.pt" % str(epoch))
        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

        # Validation
        valid_queue = queue.Queue(args.worker_num << 1)
        valid_loader = AudioDataLoader(validset, valid_queue, args.batch_size, 0)
        valid_loader.start()

        valid_loss, valid_cer = evaluate(model, valid_queue, criterion, device)
        valid_loader.join()

        scheduler.step(valid_loss)

        logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, valid_loss, valid_cer))
        save_epoch_result(train_result=[train_dict, train_loss, train_cer],
                          valid_result=[valid_dict, valid_loss, valid_cer])
        logger.info('Epoch %d Training result saved as a csv file complete !!' % epoch)
