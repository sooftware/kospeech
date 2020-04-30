import time
import torch
from utils.definition import logger, id2char, EOS_token, char2id
from utils.util import get_distance, save_step_result

train_step_result = {'loss': [], 'cer': []}


def supervised_train(model, config, epoch, total_time_step, queue,
                     criterion, optimizer, device, train_begin, worker_num,
                     print_every=10, teacher_forcing_ratio=0.90):
    r"""
    Args:
        train_begin: train begin time
        total_time_step: total time step in epoch
        epoch (int): present epoch
        config (Config): configuration
        model (torch.nn.Module): Model to be trained
        optimizer (torch.optim): optimizer for training
        teacher_forcing_ratio (float):  The probability that teacher forcing will be used (default: 0.90)
        print_every (int): Parameters to determine how many steps to output
        queue (Queue.queue): queue for threading
        criterion (torch.nn): one of PyTorch’s loss function.
          Refer to http://pytorch.org/docs/master/nn.html#loss-functions for a list of them.
        device (torch.cuda): device used ('cuda' or 'cpu')
        worker_num (int): the number of cpu cores used

    Returns: loss, cer
        - **loss** (float): loss of present epoch
        - **cer** (float): character error rate
    """
    epoch_loss_total = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    time_step = 0
    max_norm = 400
    save_model_every = 10000
    save_result_every = 1000

    model.train()
    begin = epoch_begin = time.time()

    while True:
        inputs, scripts, input_lengths, target_lengths = queue.get()

        if inputs.shape[0] == 0:
            # empty feats means closing one loader
            worker_num -= 1
            logger.debug('left train_loader: %d' % worker_num)

            if worker_num == 0:
                break
            else:
                continue

        inputs = inputs.to(device)
        scripts = scripts.to(device)
        targets = scripts[:, 1:]

        model.module.flatten_parameters()
        y_hat, logit = model(inputs, scripts, teacher_forcing_ratio=teacher_forcing_ratio)

        print(logit.size())
        print(targets.size())
        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), targets.contiguous().view(-1))
        epoch_loss_total += loss.item()

        dist, length = get_distance(targets, y_hat, id2char, char2id, EOS_token)

        total_num += sum(input_lengths)
        total_dist += dist
        total_length += length

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        time_step += 1
        torch.cuda.empty_cache()

        if time_step % print_every == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('timestep: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'.format(
                time_step,
                total_time_step,
                epoch_loss_total / total_num,
                total_dist / total_length,
                elapsed, epoch_elapsed, train_elapsed)
            )
            begin = time.time()

        if time_step % save_result_every == 0:
            save_step_result(train_step_result, epoch_loss_total / total_num, total_dist / total_length)

        if time_step % save_model_every == 0:
            torch.save(model, "./data/weight_file/epoch_%s_step_%s.pt" % (str(epoch), str(time_step)))

    logger.info('train() completed')
    return epoch_loss_total / total_num, total_dist / total_length


def set_lr(optimizer, lr):
    """ set learning rate """
    for g in optimizer.param_groups:
        g['lr'] = lr


def get_lr(optimizer):
    """ get learning rate """
    for g in optimizer.param_groups:
        return g['lr']
