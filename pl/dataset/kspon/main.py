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

import argparse
from preprocess.grapheme import sentence_to_grapheme
from preprocess.preprocess import preprocess
from preprocess.character import generate_character_labels, generate_character_script
from preprocess.subword import train_sentencepiece, sentence_to_subwords


def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser(description='KsponSpeech Preprocess')
    parser.add_argument('--dataset_path', type=str,
                        default='E:/KsponSpeech/original',
                        help='path of original dataset')
    parser.add_argument('--vocab_dest', type=str,
                        default='E:/KsponSpeech',
                        help='destination to save character / subword labels file')
    parser.add_argument('--output_unit', type=str,
                        default='character',
                        help='character or subword or grapheme')
    parser.add_argument('--savepath', type=str,
                        default='./data',
                        help='path of data')
    parser.add_argument('--preprocess_mode', type=str,
                        default='phonetic',
                        help='Ex) (70%)/(칠 십 퍼센트) 확률이라니 (뭐 뭔)/(모 몬) 소리야 진짜 (100%)/(백 프로)가 왜 안돼?'
                             'phonetic: 칠 십 퍼센트 확률이라니 모 몬 소리야 진짜 백 프로가 왜 안돼?'
                             'spelling: 70% 확률이라니 뭐 뭔 소리야 진짜 100%가 왜 안돼?')
    parser.add_argument('--vocab_size', type=int,
                        default=5000,
                        help='size of vocab (default: 5000)')

    return parser


def log_info(opt):
    print("Dataset Path : %s" % opt.dataset_path)
    print("Vocab Destination : %s" % opt.vocab_dest)
    print("Save Path : %s" % opt.savepath)
    print("Output-Unit : %s" % opt.output_unit)
    print("Preprocess Mode : %s" % opt.preprocess_mode)


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    log_info(opt)

    audio_paths, transcripts = preprocess(opt.dataset_path, opt.preprocess_mode)

    if opt.output_unit == 'character':
        generate_character_labels(transcripts, opt.vocab_dest)
        generate_character_script(audio_paths, transcripts, opt.vocab_dest)

    elif opt.output_unit == 'subword':
        train_sentencepiece(transcripts, opt.savepath, opt.vocab_size)
        sentence_to_subwords(audio_paths, transcripts, opt.savepath)

    elif opt.output_unit == 'grapheme':
        sentence_to_grapheme(audio_paths, transcripts, opt.vocab_dest)

    else:
        raise ValueError("Unsupported preprocess method : {0}".format(opt.output_unit))


if __name__ == '__main__':
    main()
