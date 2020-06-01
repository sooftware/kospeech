import pandas as pd
import re


def load_label(filepath):
    char2id = dict()
    id2char = dict()
    ch_labels = pd.read_csv(filepath, encoding="cp949")
    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    freq_list = ch_labels["freq"]

    for (id, char, freq) in zip(id_list, char_list, freq_list):
        char2id[char] = id
        id2char[id] = char
    return char2id, id2char


def bracket_filter(sentence):
    new_sentence = str()
    flag = False

    for ch in sentence:
        if ch == '(' and flag is False:
            flag = True
            continue
        if ch == '(' and flag is True:
            flag = False
            continue
        if ch != ')' and flag is False:
            new_sentence += ch
    return new_sentence


def special_filter(sentence):
    SENTENCE_MARK = ['?', '!']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', '.', ',']

    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            # o/, n/ 등 처리
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue

        if ch == '#':
            new_sentence += '샾'

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence):
    return special_filter(bracket_filter(raw_sentence))


def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        target += (str(char2id[ch]) + ' ')

    return target[:-1]
