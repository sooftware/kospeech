import os
import hgtk
import pandas as pd


def sentence_to_target(sentence, grpm2id):
    target = str()

    for grapheme in sentence:
        target += (str(grpm2id[grapheme]) + ' ')

    return target[:-1]


def load_label(filepath):
    grpm2id = dict()
    id2grpm = dict()

    grapheme_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = grapheme_labels["id"]
    grapheme_list = grapheme_labels["grapheme"]

    for (idx, grapheme) in zip(id_list, grapheme_list):
        grpm2id[grapheme] = idx
        id2grpm[idx] = grapheme

    return grpm2id, id2grpm


def character_to_grapheme(dataset_path, grapheme_save_path):
    for folder in os.listdir(dataset_path):
        if not folder.startswith('KsponSpeech'):
            continue
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                if file.endswith('.txt'):
                    with open(os.path.join(path, file), "r", encoding='cp949') as f:
                        sentence = f.read()

                    with open(os.path.join(grapheme_save_path, file), "w", encoding='cp949') as f:
                        f.write(hgtk.text.decompose(sentence).replace('ᴥ', ''))


def generate_grapheme_labels(dataset_path, labels_dest):
    print('create_grapheme_labels started..')

    grapheme_list = list()
    grapheme_freq = list()

    for folder in os.listdir(dataset_path):
        if not folder.startswith('KsponSpeech'):
            continue
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                if file.endswith('txt'):
                    with open(os.path.join(path, file), "r", encoding='cp949') as f:
                        sentence = f.read()

                        for grapheme in sentence:
                            if grapheme not in grapheme_list:
                                grapheme_list.append(grapheme)
                                grapheme_freq.append(1)
                            else:
                                grapheme_freq[grapheme_list.index(grapheme)] += 1

    # sort together Using zip
    grapheme_freq, grapheme_list = zip(*sorted(zip(grapheme_freq, grapheme_list), reverse=True))
    grapheme_dict = {'id': [0, 1, 2], 'char': ['<pad>', '<sos>', '<eos>'], 'freq': [0, 0, 0]}

    for idx, (grapheme, freq) in enumerate(zip(grapheme_list, grapheme_freq)):
        grapheme_dict['id'].append(idx + 3)
        grapheme_dict['grapheme'].append(grapheme)
        grapheme_dict['freq'].append(freq)

    grapheme_df = pd.DataFrame(grapheme_dict)
    grapheme_df.to_csv(os.path.join(labels_dest, 'grapheme_labels.csv'), encoding='utf-8')


def generate_grapheme_script(dataset_path, new_path, script_prefix, labels_dest):
    print('create_grapheme_script started..')
    grpm2id, id2grpm = load_label(os.path.join(labels_dest, 'grapheme_labels.csv'))

    for folder in os.listdir(dataset_path):
        if not folder.startswith('KsponSpeech'):
            continue
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                if file.endswith('.txt'):
                    with open(os.path.join(path, file), "r", encoding='cp949') as f:
                        sentence = f.read()

                    with open(os.path.join(new_path, script_prefix + file[12:]), "w", encoding='cp949') as f:
                        target = sentence_to_target(sentence, grpm2id)
                        f.write(target)
