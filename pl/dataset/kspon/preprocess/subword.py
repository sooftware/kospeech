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


def train_sentencepiece(transcripts, datapath: str = './data', vocab_size: int = 5000):
    print('generate_sentencepiece_input..')

    if not os.path.exists(datapath):
        os.mkdir(datapath)

    with open(f'{datapath}/sentencepiece_input.txt', 'w') as f:
        for transcript in transcripts:
            transcript = transcript.upper()
            f.write(f'{transcript}\n')

    spm.SentencePieceTrainer.Train(
        f'--input={datapath}/sentencepiece_input.txt '
        '--model_prefix=kspon_sentencepiece '
        f'--vocab_size={vocab_size} '
        '--model_type=bpe '
        '--max_sentence_length=9999 '
        '--hard_vocab_limit=false'
        '--pad_id=0'
        '--bos_id=1'
        '--eos_id=2'
    )


def sentence_to_subwords(audio_paths: list, transcripts: list, datapath: str = './data'):
    subwords = list()

    print('sentence_to_subwords...')

    sp = spm.SentencePieceProcessor()
    vocab_file = "kspon_sentencepiece.model"
    sp.load(vocab_file)

    with open(f'{datapath}/transcripts.txt', 'w') as f:
        for audio_path, transcript in zip(audio_paths, transcripts):
            audio_path = audio_path.replace('txt', 'pcm')
            subword_transcript = " ".join(sp.EncodeAsPieces(transcript))
            subword_id_transcript = " ".join([str(item) for item in sp.EncodeAsIds(transcript)])
            f.write(f'{audio_path}\t{subword_transcript}\t{subword_id_transcript}\n')

    return subwords
