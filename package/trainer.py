import time
import torch
from package.definition import logger, id2char, EOS_TOKEN
from package.utils import get_distance, save_step_result

train_step_result = {'loss': [], 'cer': []}

def supervised_train(model, config, epoch, total_time_step, queue,
                     criterion, optimizer, device, train_begin, worker_num,
                     print_time_step=10, teacher_forcing_ratio=0.90):
    r"""
    Args:
        model (torch.nn.Module): Model to be trained
        optimizer (torch.optim): optimizer for training
        teacher_forcing_ratio (float):  The probability that teacher forcing will be used (default: 0.90)
        print_time_step (int): Parameters to determine how many steps to output
        queue (Queue.queue): queue for threading
        criterion (torch.nn): one of PyTorch’s loss function.
          Refer to http://pytorch.org/docs/master/nn.html#loss-functions for a list of them.
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
    time_step = 0
    decay_speed = 1.0

    RAMPUP_POWER = 3
    RANMPUP_PERIOD = 3000
    EXP_DECAY_PERIOD = total_time_step * 3

    model.train()
    begin = epoch_begin = time.time()

    while True:
        # LR Wamp-Up
        if config.use_multistep_lr and epoch == 0 and time_step < RANMPUP_PERIOD:
            set_lr(optimizer, lr=config.high_plateau_lr * ((time_step + 1) / RANMPUP_PERIOD) ** RAMPUP_POWER)

        # LR Exponential-Decay
        if config.use_multistep_lr and (epoch == 1 or epoch == 2 or epoch == 3):
            decay_rate = config.low_plateau_lr / config.high_plateau_lr
            decay_speed *= decay_rate ** (1 / EXP_DECAY_PERIOD)
            set_lr(optimizer, config.high_plateau_lr * decay_speed)

        feats, scripts, feat_lens, target_lens = queue.get()

        if feats.shape[0] == 0:
            # empty feats means closing one loader
            worker_num -= 1
            logger.debug('left train_loader: %d' % (worker_num))

            if worker_num == 0:
                break
            else:
                continue


        inputs = feats.to(device)
        scripts = scripts.to(device)
        targets = scripts[:, 1:]

        model.module.flatten_parameters()
        y_hat, logit = model(inputs, scripts, teacher_forcing_ratio=teacher_forcing_ratio)

        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), targets.contiguous().view(-1))
        total_loss += loss.item()

        total_num += sum(feat_lens)
        dist, length = get_distance(targets, y_hat, id2char, EOS_TOKEN)
        total_dist += dist
        total_length += length

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_step += 1
        torch.cuda.empty_cache()

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
            torch.save(model, "./data/weight_file/epoch_%s_step_%s.pt" % (str(epoch), str(time_step)))

    logger.info('train() completed')

    return total_loss / total_num, total_dist / total_length


def set_lr(optimizer, lr):
    """ set learning rate """
    for g in optimizer.param_groups:
        g['lr'] = lr


def get_lr(optimizer):
    """ get learning rate """
    for g in optimizer.param_groups:
        return g['lr']