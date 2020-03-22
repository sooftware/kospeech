import torch
from package.definition import logger, id2char, EOS_TOKEN
from package.utils import get_distance


def evaluate(model, queue, criterion, device):
    r"""
    Args:
        model (torch.nn.Module): Model to be evaluated
        queue (queue): queue for threading
        criterion (torch.nn): one of PyTorch’s loss function. Refer to http://pytorch.org/docs/master/nn.html#loss-functions for a list of them.
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
            feats, scripts, feat_lengths, script_lengths = queue.get()

            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            scripts = scripts.to(device)
            targets = scripts[:, 1:]

            model.module.flatten_parameters()
            y_hat, logit = model(feats, scripts, teacher_forcing_ratio=0.0, use_beam_search = False)

            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), targets.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            dist, length = get_distance(targets, y_hat, id2char, EOS_TOKEN)
            total_dist += dist
            total_length += length

    logger.info('evaluate() completed')

    return total_loss / total_num, total_dist / total_length