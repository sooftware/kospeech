import time
import torch
from package.definition import logger, id2char, EOS_TOKEN
from package.utils import get_distance, save_step_result

train_step_result = {'loss': [], 'cer': []}

def supervised_train(model, hparams, epoch, total_time_step, queue,
          criterion, optimizer, device, train_begin, worker_num,
          print_time_step=10, teacher_forcing_ratio=0.90):
    """
    Args:
        model (torch.nn.Module): Model to be trained
        optimizer (torch.optim): optimizer for training
        teacher_forcing_ratio (float):  The probability that teacher forcing will be used (default: 0.90)
        print_time_step (int): Parameters to determine how many steps to output
        queue (Queue.queue): queue for threading
        criterion (torch.nn): one of PyTorch’s loss function. Refer to http://pytorch.org/docs/master/nn.html#loss-functions for a list of them.
        device (torch.cuda): device used ('cuda' or 'cpu')
        worker_num (int): the number of cpu cores used

    Returns: loss, cer
        - **loss** (float): loss of present epoch
        - **cer** (float): character error rate
    """
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    time_step = 0

    model.train()
    begin = epoch_begin = time.time()

    while True:
        if hparams.use_multistep_lr and epoch == 0 and time_step < 1000:
            ramp_up(optimizer, time_step, hparams)
        if hparams.use_multistep_lr and epoch == 1:
            exp_decay(optimizer, total_time_step, hparams)
        feats, targets, feat_lens, target_lens = queue.get()
        if feats.shape[0] == 0:
            # empty feats means closing one loader
            worker_num -= 1
            logger.debug('left train_loader: %d' % (worker_num))

            if worker_num == 0:
                break
            else:
                continue
        optimizer.zero_grad()

        inputs = feats.to(device)
        targets = targets.to(device)
        target = targets[:, 1:]
        model.module.flatten_parameters()

        y_hat, logit = model(inputs, targets, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))

        total_loss += loss.item()
        total_num += sum(feat_lens)
        dist, length = get_distance(target, y_hat, id2char, EOS_TOKEN)
        total_dist += dist
        total_length += length
        total_sent_num += target.size(0)
        loss.backward()
        optimizer.step()

        if time_step % print_time_step == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('timestep: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'.format(
                time_step,
                total_time_step,
                total_loss / total_num,
                total_dist / total_length,
                elapsed, epoch_elapsed, train_elapsed)
            )
            begin = time.time()

        if time_step % 1000 == 0:
            save_step_result(train_step_result, total_loss / total_num, total_dist / total_length)

        if time_step % 10000 == 0:
            torch.save(model, "model.pt")
            torch.save(model, "./data/weight_file/epoch_%s_step_%s.pt" % (str(epoch), str(time_step)))

        time_step += 1
        supervised_train.cumulative_batch_count += 1
        torch.cuda.empty_cache()

    logger.info('train() completed')
    return total_loss / total_num, total_dist / total_length

supervised_train.cumulative_batch_count = 0

def ramp_up(optimizer, time_step, hparams):
    """
    Steps to gradually increase the learing rate

    Reference:
        「SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition」Google Brain Team. 2019.
        https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_pytorch.py
    """
    power = 3
    lr = hparams.high_plateau_lr * (time_step / 1000) ** power
    set_lr(optimizer, lr)

def exp_decay(optimizer, total_time_step, hparams):
    """
    a gradual decrease in learning rates

    Reference:
        「SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition」Google Brain Team. 2019.
        https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_pytorch.py
    """
    decay_rate = hparams.low_plateau_lr / hparams.high_plateau_lr
    decay_speed = decay_rate ** (1/total_time_step)
    lr = get_lr(optimizer)
    set_lr(optimizer, lr * decay_speed)

def set_lr(optimizer, lr):
    """ set learning rate """
    for g in optimizer.param_groups:
        g['lr'] = lr

def get_lr(optimizer):
    """ get learning rate """
    for g in optimizer.param_groups:
        return g['lr']