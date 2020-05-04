import time
import torch
from definition import logger, id2char, EOS_token, char2id
from utils import get_distance, save_step_result

train_step_result = {'loss': [], 'cer': []}


def supervised_train(model, args, epoch, total_time_step, queue, criterion, optimizer, device, train_begin):
    r"""
    Args:
        train_begin: train begin time
        total_time_step: total time step in epoch
        epoch (int): present epoch
        args (Arguments): set of Argugments
        model (torch.nn.Module): Model to be trained
        optimizer (torch.optim): optimizer for training
        queue (Queue.queue): queue for threading
        criterion (torch.nn): one of PyTorch’s loss function.
          Refer to http://pytorch.org/docs/master/nn.html#loss-functions for a list of them.
        device (torch.cuda): device used ('cuda' or 'cpu')

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

    model.train()
    begin = epoch_begin = time.time()

    while True:
        inputs, scripts, input_lengths, target_lengths = queue.get()

        if inputs.shape[0] == 0:
            # empty feats means closing one loader
            args.worker_num -= 1
            logger.debug('left train_loader: %d' % args.worker_num)

            if args.worker_num == 0:
                break
            else:
                continue

        inputs = inputs.to(device)
        scripts = scripts.to(device)
        targets = scripts[:, 1:]

        model.module.flatten_parameters()
        hypothesis, logit = model(inputs, input_lengths, scripts, teacher_forcing_ratio=args.teacher_forcing_ratio)

        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), targets.contiguous().view(-1))
        epoch_loss_total += loss.item()

        dist, length = get_distance(targets, hypothesis, id2char, char2id, EOS_token)

        total_num += int(input_lengths.sum())
        total_dist += dist
        total_length += length

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        time_step += 1
        torch.cuda.empty_cache()

        if time_step % args.print_every == 0:
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

        if time_step % args.save_result_every == 0:
            save_step_result(train_step_result, epoch_loss_total / total_num, total_dist / total_length)

        if time_step % args.save_model_every == 0:
            torch.save(model, "./data/weight_file/epoch_%s_step_%s.pt" % (str(epoch), str(time_step)))

    logger.info('train() completed')
    return epoch_loss_total / total_num, total_dist / total_length


def evaluate(model, queue, criterion, device):
    r"""
    Args:
        model (torch.nn.Module): Model to be evaluated
        queue (queue): queue for threading
        criterion (torch.nn): one of PyTorch’s loss function.
            Refer to http://pytorch.org/docs/master/nn.html#loss-functions for a list of them.
        device (torch.cuda): device used ('cuda' or 'cpu')

    Returns: loss, cer
        - **loss** (float): loss of evalution
        - **cer** (float): character error rate
    """
    logger.info('evaluate() start')

    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0

    model.eval()

    with torch.no_grad():
        while True:
            inputs, scripts, input_lengths, script_lengths = queue.get()

            if inputs.shape[0] == 0:
                break

            inputs = inputs.to(device)
            scripts = scripts.to(device)
            targets = scripts[:, 1:]

            model.module.flatten_parameters()
            hypothesis, logit = model(inputs, teacher_forcing_ratio=0.0, use_beam_search=False)

            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), targets.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(input_lengths)

            dist, length = get_distance(targets, hypothesis, id2char, char2id, EOS_token)
            total_dist += dist
            total_length += length

    logger.info('evaluate() completed')
    return total_loss / total_num, total_dist / total_length


def set_lr(optimizer, lr):
    """ set learning rate """
    for g in optimizer.param_groups:
        g['lr'] = lr


def get_lr(optimizer):
    """ get learning rate """
    for g in optimizer.param_groups:
        return g['lr']
