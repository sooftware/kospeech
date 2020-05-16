import time
import torch
import torch.nn as nn
import queue
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import load_data_list, load_pickle, split_dataset, MultiDataLoader, AudioDataLoader
#from definition import logger, id2char, EOS_token, char2id, PAD_token, TARGET_DICT_PATH, valid_dict, train_dict
from label_loader import load_targets
from loss import LabelSmoothingLoss
from utils import get_distance, save_step_result, save_epoch_result

train_step_result = {'loss': [], 'cer': []}


class SupervisedTrainer:
    """
    The SupervisedTrainer class helps in setting up training framework in a supervised setting.

    Args:
        model (torch.nn.Module):
        args (argparse.ArgumentParser):
        device (torch.device): device - 'cuda' or 'cpu'
    """
    def __init__(self, model, args, device):
        self.model = model
        self.args = args
        self.device = device
        self.optimizer = optim.Adam(model.module.parameters(), lr=args.lr)
        self.lr_scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            patience=args.lr_patience,
            factor=args.lr_factor,
            verbose=True,
            min_lr=args.min_lr
        )

        if args.label_smoothing == 0.0:
            self.criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)
        else:
            self.criterion = LabelSmoothingLoss(len(char2id), PAD_token, args.label_smoothing, dim=-1).to(device)

    def train(self, data_list_path, dataset_path, start_epoch):
        audio_paths, label_paths = load_data_list(data_list_path, dataset_path)

        if self.args.use_pickle:
            target_dict = load_pickle(TARGET_DICT_PATH, "load all target_dict using pickle complete !!")
        else:
            target_dict = load_targets(label_paths)

        total_time_step, trainset_list, validset = split_dataset(self.args, audio_paths, label_paths, target_dict)

        logger.info('start')
        train_begin = time.time()

        for epoch in range(start_epoch, self.args.num_epochs):
            train_queue = queue.Queue(self.args.num_workers << 1)
            for trainset in trainset_list:
                trainset.shuffle()

            # Training
            train_loader = MultiDataLoader(trainset_list, train_queue, self.args.batch_size, self.args.num_workers)
            train_loader.start()
            train_loss, train_cer = self.train_epoches(epoch, total_time_step, train_begin, train_queue)
            train_loader.join()

            torch.save(self.model, "./data/weight_file/epoch%s.pt" % str(epoch))
            logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

            # Validation
            valid_queue = queue.Queue(self.args.num_workers << 1)
            valid_loader = AudioDataLoader(validset, valid_queue, self.args.batch_size, 0)
            valid_loader.start()

            valid_loss, valid_cer = self.validate(valid_queue)
            valid_loader.join()

            self.lr_scheduler.step(valid_loss)

            logger.info('Epoch %d (Validate) Loss %0.4f CER %0.4f' % (epoch, valid_loss, valid_cer))
            save_epoch_result(train_result=[train_dict, train_loss, train_cer],
                              valid_result=[valid_dict, valid_loss, valid_cer])
            logger.info('Epoch %d Training result saved as a csv file complete !!' % epoch)

    def train_epoches(self, epoch, epoch_time_step, train_begin, queue):
        epoch_loss_total = 0.
        total_num = 0
        total_dist = 0
        total_length = 0
        time_step = 0
        max_norm = 400

        self.model.train()
        begin = epoch_begin = time.time()

        while True:
            inputs, scripts, input_lengths, target_lengths = queue.get()

            if inputs.shape[0] == 0:
                # empty feats means closing one loader
                self.args.num_workers -= 1
                logger.debug('left train_loader: %d' % self.args.num_workers)

                if self.args.num_workers == 0:
                    break
                else:
                    continue

            inputs = inputs.to(self.device)
            scripts = scripts.to(self.device)
            targets = scripts[:, 1:]

            self.model.module.flatten_parameters()
            output = self.model(inputs, input_lengths, scripts, teacher_forcing_ratio=self.args.teacher_forcing_ratio)

            logit = torch.stack(output, dim=1).to(self.device)
            hypothesis = logit.max(-1)[1]

            loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), targets.contiguous().view(-1))
            epoch_loss_total += loss.item()

            dist, length = get_distance(targets, hypothesis, id2char, EOS_token)

            total_num += int(input_lengths.sum())
            total_dist += dist
            total_length += length

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.optimizer.step()

            time_step += 1
            torch.cuda.empty_cache()

            if time_step % self.args.print_every == 0:
                current = time.time()
                elapsed = current - begin
                epoch_elapsed = (current - epoch_begin) / 60.0
                train_elapsed = (current - train_begin) / 3600.0

                logger.info('timestep: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'.format(
                    time_step,
                    epoch_time_step,
                    epoch_loss_total / total_num,
                    total_dist / total_length,
                    elapsed, epoch_elapsed, train_elapsed)
                )
                begin = time.time()

            if time_step % self.args.save_result_every == 0:
                save_step_result(train_step_result, epoch_loss_total / total_num, total_dist / total_length)

            if time_step % self.args.save_model_every == 0:
                torch.save(self.model, "./data/weight_file/epoch_%s_step_%s.pt" % (str(epoch), str(time_step)))

        logger.info('train() completed')
        return epoch_loss_total / total_num, total_dist / total_length

    def validate(self, queue):
        logger.info('validate() start')

        total_loss = 0.
        total_num = 0
        total_dist = 0
        total_length = 0

        self.model.eval()

        with torch.no_grad():
            while True:
                inputs, scripts, input_lengths, script_lengths = queue.get()

                if inputs.shape[0] == 0:
                    break

                inputs = inputs.to(self.device)
                scripts = scripts.to(self.device)
                targets = scripts[:, 1:]

                self.model.module.flatten_parameters()
                output = self.model(inputs, input_lengths, teacher_forcing_ratio=0.0)

                logit = torch.stack(output, dim=1).to(self.device)
                hypothesis = logit.max(-1)[1]

                loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), targets.contiguous().view(-1))
                total_loss += loss.item()
                total_num += sum(input_lengths)

                dist, length = get_distance(targets, hypothesis, id2char, EOS_token)
                total_dist += dist
                total_length += length

        logger.info('validate() completed')
        return total_loss / total_num, total_dist / total_length
