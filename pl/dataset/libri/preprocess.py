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
import sentencepiece as spm

LIBRI_SPEECH_DATASETS = [
    'train_960', 'dev-clean', 'dev-other', 'test-clean', 'test-other'
]


def collect_transcripts(dataset_path):
    transcripts_collection = list()

    for dataset in LIBRI_SPEECH_DATASETS:
        dataset_transcripts = list()

        for subfolder1 in os.listdir(os.path.join(dataset_path, dataset)):
            for subfolder2 in os.listdir(os.path.join(dataset_path, dataset, subfolder1)):
                for file in os.listdir(os.path.join(dataset_path, dataset, subfolder1, subfolder2)):
                    if file.endswith('txt'):
                        with open(os.path.join(dataset_path, dataset, subfolder1, subfolder2, file)) as f:
                            for line in f.readlines():
                                tokens = line.split()
                                audio = '%s.flac' % os.path.join(dataset, subfolder1, subfolder2, tokens[0])
                                transcript = " ".join(tokens[1:])
                                dataset_transcripts.append('%s|%s' % (audio, transcript))

                    else:
                        continue

        transcripts_collection.append(dataset_transcripts)

    return transcripts_collection


def prepare_tokenizer(train_transcripts, vocab_size):
    input_file = 'spm_input.txt'
    model_name = 'tokenizer'
    model_type = 'unigram'

    with open(input_file, 'w') as f:
        for transcript in train_transcripts:
            f.write('{}\n'.format(transcript.split('|')[-1]))

    input_args = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s'
    cmd = input_args % (input_file, model_name, vocab_size, model_type)
    spm.SentencePieceTrainer.Train(cmd)


def generate_transcript_file(dataset_name, transcripts):
    sp = spm.SentencePieceProcessor()
    sp.Load("tokenizer.model")

    with open('../../data/%s-transcript.txt' % dataset_name, 'w') as f:
        for transcript in transcripts:
            audio, transcript = transcript.split('|')
            text = " ".join(sp.EncodeAsPieces(transcript))
            label = " ".join([str(item) for item in sp.EncodeAsIds(transcript)])

            f.write('%s\t%s\t%s\n' % (audio, text, label))


def merge_train_dev_transcript_file():
    merge_list = ['train_960', 'dev-clean', 'dev-other']

    lines = list()

    for dataset in merge_list:
        with open('../../data/%s-transcript.txt' % dataset) as f:
            for line in f.readlines():
                lines.append(line)

    with open('../../data/train.txt', 'w') as f:
        for line in lines:
            f.write('%s' % line)
