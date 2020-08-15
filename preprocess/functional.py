import re
import pandas as pd


def load_label(filepath):
    char2id = dict()
    id2char = dict()
    ch_labels = pd.read_csv(filepath, encoding="cp949")
    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    freq_list = ch_labels["freq"]

    for (id_, char, freq) in zip(id_list, char_list, freq_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def bracket_filter(sentence, mode):
    new_sentence = str()

    if mode == 'phonetic':
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

    elif mode == 'numeric':
        update = True

        for ch in sentence:
            if ch == '(':
                continue
            if ch == ')':
                if update is True:
                    update = False
                    continue
                else:
                    update = True
                    continue
            if ch != ')' and update is True:
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


def sentence_filter(raw_sentence, mode):
    return special_filter(bracket_filter(raw_sentence, mode))


def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        target += (str(char2id[ch]) + ' ')

    return target[:-1]


def percent_process(sentence, option):
    if option == 'short':
        return sentence.replace("%", "프로")
    elif option == 'long':
        return sentence.replace("%", "퍼센트")
