import Levenshtein as Lev
from utils.label import label_to_string

def char_distance(target, y_hat):
    """
    Calculating charater distance between target & y_hat

    Args:
        target: sequence of target
        y_hat: sequence of y_Hat

    Returns:
        - **distance**: distance between target & y_hat
        - **length**: length of target sequence
    """
    target = target.replace(' ', '')
    y_hat = y_hat.replace(' ', '')
    distance = Lev.distance(y_hat, target)
    length = len(target.replace(' ', ''))

    return distance, length

def get_distance(targets, y_hats, id2char, eos_id):
    """
    Provides total character distance between targets & y_hats

    Args:
        targets: set of target
        y_hats: set of y_hat
        id2char: id2char[id] = ch
        eos_id: identification of <end of sequence>

    Returns:
        - **total_distance**: total distance between targets & y_hats
        - **total_length**: total length of targets sequence
    """
    total_distance = 0
    total_length = 0

    for i in range(len(targets)):
        target = label_to_string(targets[i], id2char, eos_id)
        y_hat = label_to_string(y_hats[i], id2char, eos_id)
        distance, length = char_distance(target, y_hat)
        total_distance += distance
        total_length += length
    return total_distance, total_length