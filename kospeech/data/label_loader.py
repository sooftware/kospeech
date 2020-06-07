import csv
from tqdm import trange


def load_label(label_path, encoding='utf-8'):
    """
    Provides char2id, id2char

    Args:
        label_path (str): csv file with character labels
        encoding (str): encoding method

    Returns: char2id, id2char
        - **char2id** (dict): char2id[ch] = id
        - **id2char** (dict): id2char[id] = ch
    """
    char2id = dict()
    id2char = dict()

    try:
        with open(label_path, 'r', encoding=encoding) as f:
            labels = csv.reader(f, delimiter=',')
            next(labels)

            for row in labels:
                char2id[row[1]] = row[0]
                id2char[int(row[0])] = row[1]

        return char2id, id2char
    except IOError:
        raise IOError("Character label file (csv format) doesn`t exist : {0}".format(label_path))


def load_targets(label_paths):
    """
    Provides dictionary of filename and labels

    Args:
        label_paths (list): set of label paths

    Returns: target_dict
        - **target_dict** (dict): dictionary of filename and labels
    """
    target_dict = dict()

    for idx in trange(len(label_paths)):
        label_txt = label_paths[idx]

        try:
            with open(file=label_txt, mode="r") as f:
                label = f.readline()
                file_num = label_txt.split('/')[-1].split('.')[0].split('_')[-1]
                target_dict['KsponScript_%s' % file_num] = label
        except IOError:
            raise IOError("label paths file (csv format) doesn`t exist : {0}".format(label_paths))

    return target_dict
