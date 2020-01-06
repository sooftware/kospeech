import csv

#-*- coding: utf-8 -*-

def load_label(label_path):
    char2index = dict() # [ch] = id
    index2char = dict() # [id] = ch
    f = open(label_path, 'r', encoding="UTF-8")
    labels = csv.reader(f, delimiter=',')
    header = next(labels)

    for row in labels:
        char2index[row[1]] = row[0]
        index2char[row[0]] = row[1]

    return char2index, index2char
