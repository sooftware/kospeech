import os
import pandas as pd
import shutil
from .functional import sentence_filter, load_label, sentence_to_target, percent_process


def preprocess(dataset_path, new_path, mode, filenum_adjust):
    # files which has "%" in their sentence
    percent_files = ['087797', '215401', '284574', '397184', '501006', '502173', '542363', '581483']

    # adjust file number; maybe one file has deleted in this case, so filenums should reduced by 1.
    adjust_val = [0, -1, -1, -1, -1, -1, -1, -1]

    # True for "퍼센트", False for "프로"
    milestone = [True, True, True, True, False, False, False, True]

    if filenum_adjust:
        for idx in range(len(percent_files)):
            percent_files[idx] = str(int(percent_files[idx]) + adjust_val[idx])
            if len(percent_files[idx]) < 6:
                percent_files[idx] = "0" * (6 - len(percent_files[idx])) + percent_files[idx]

    print('preprocess started..')

    for file in os.listdir(dataset_path):
        if "Script" in file:
            continue

        if file.endswith('.txt'):

            with open(os.path.join(dataset_path, file), "r") as f:
                raw_sentence = f.read()
                new_sentence = sentence_filter(raw_sentence, mode)

                # handle "%" to "퍼센트" or "프로"
                filenum = file[-10:-4]
                if filenum in percent_files:
                    if milestone[percent_files.index(filenum)]:
                        percent_process(new_sentence, 'long')
                    else:
                        percent_process(new_sentence, 'short')

            with open(os.path.join(new_path, file), "w") as f:
                f.write(new_sentence)

        else:
            continue


def create_char_labels(dataset_path, label_dest):
    print('create_char_labels started..')

    label_list = list()
    label_freq = list()

    for file in os.listdir(dataset_path):
        if "Script" in file: continue
        if file.endswith('txt'):
            with open(os.path.join(dataset_path, file), "r") as f:
                sentence = f.read()

                for ch in sentence:
                    if ch not in label_list:
                        label_list.append(ch)
                        label_freq.append(1)
                    else:
                        label_freq[label_list.index(ch)] += 1
        else:
            continue

    # sort together Using zip
    label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
    label = {'id': [0, 1, 2], 'char': ['<pad>', '<sos>', '<eos>'], 'freq': [0, 0, 0]}

    for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
        label['id'].append(idx + 3)
        label['char'].append(ch)
        label['freq'].append(freq)

    # save to csv
    label_df = pd.DataFrame(label)
    label_df.to_csv(label_dest+"aihub_labels.csv", encoding="utf-8", index=False)


def create_script(dataset_path, new_path, script_prefix):
    print('create_script started..')
    char2id, id2char = load_label(new_path+'aihub_labels.csv')

    for file in os.listdir(dataset_path):
        if "Script" in file: continue
        if file.endswith('.txt'):

            with open(os.path.join(dataset_path, file), "r") as f:
                sentence = f.read()

            with open(os.path.join(new_path, script_prefix + file[12:]), "w") as f:
                target = sentence_to_target(sentence, char2id)
                f.write(target)


# not use this time, not changed
def gather_files(dataset_path, new_path, script_prefix):
    print('gather_files started...')
    for folder in os.listdir(dataset_path):
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        for subfolder in os.listdir(folder):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                if (file.endswith('.txt') and file.startswith(script_prefix)) or file.endswith('.pcm'):
                    shutil.move(os.path.join(path, file), os.path.join(new_path, file))
