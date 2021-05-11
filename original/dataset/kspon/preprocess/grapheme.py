# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd
import unicodedata


def load_label(filepath):
    grpm2id = dict()
    id2grpm = dict()

    vocab_data_frame = pd.read_csv(filepath, encoding="utf-8")

    id_list = vocab_data_frame["id"]
    grpm_list = vocab_data_frame["grpm"]

    for _id, grpm in zip(id_list, grpm_list):
        grpm2id[grpm] = _id
        id2grpm[_id] = grpm
    return grpm2id, id2grpm


def sentence_to_target(transcript, grpm2id):
    target = str()

    for grpm in transcript:
        target += (str(grpm2id[grpm]) + ' ')

    return target[:-1]


def sentence_to_grapheme(audio_paths, transcripts, vocab_dest: str = './data'):
    grapheme_transcripts = list()

    if not os.path.exists(vocab_dest):
        os.mkdir(vocab_dest)

    for transcript in transcripts:
        grapheme_transcripts.append(" ".join(unicodedata.normalize('NFKD', transcript).replace(' ', '|')).upper())

    generate_grapheme_labels(grapheme_transcripts, vocab_dest)

    print('create_script started..')
    grpm2id, id2grpm = load_label(os.path.join(vocab_dest, "aihub_labels.csv"))

    with open(os.path.join(f"{vocab_dest}/transcripts.txt"), "w") as f:
        for audio_path, transcript, grapheme_transcript in zip(audio_paths, transcripts, grapheme_transcripts):
            audio_path = audio_path.replace('txt', 'pcm')
            grpm_id_transcript = sentence_to_target(grapheme_transcript.split(), grpm2id)
            f.write(f'{audio_path}\t{transcript}\t{grpm_id_transcript}\n')


def generate_grapheme_labels(grapheme_transcripts, vocab_dest: str = './data'):
    vocab_list = list()
    vocab_freq = list()

    for grapheme_transcript in grapheme_transcripts:
        graphemes = grapheme_transcript.split()
        for grapheme in graphemes:
            if grapheme not in vocab_list:
                vocab_list.append(grapheme)
                vocab_freq.append(1)
            else:
                vocab_freq[vocab_list.index(grapheme)] += 1

    vocab_freq, vocab_list = zip(*sorted(zip(vocab_freq, vocab_list), reverse=True))
    vocab_dict = {
        'id': [0, 1, 2],
        'grpm': ['<pad>', '<sos>', '<eos>'],
        'freq': [0, 0, 0]
    }

    for idx, (grpm, freq) in enumerate(zip(vocab_list, vocab_freq)):
        vocab_dict['id'].append(idx + 3)
        vocab_dict['grpm'].append(grpm)
        vocab_dict['freq'].append(freq)

    label_df = pd.DataFrame(vocab_dict)
    label_df.to_csv(os.path.join(vocab_dest, "aihub_labels.csv"), encoding="utf-8", index=False)
