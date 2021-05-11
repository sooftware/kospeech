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

from kospeech.vocabs import Vocabulary


class LibriSpeechVocabulary(Vocabulary):
    def __init__(self, vocab_path, model_path):
        super(LibriSpeechVocabulary, self).__init__()
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("Please install sentencepiece: `pip install sentencepiece`")
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2

        self.vocab_path = vocab_path

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def __len__(self):
        count = 0
        with open(self.vocab_path, encoding='utf-8') as f:
            for _ in f.readlines():
                count += 1

        return count

    def label_to_string(self, labels):
        if len(labels.shape) == 1:
            return self.sp.DecodeIds([l for l in labels])

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                sentence = self.sp.DecodeIds([l for l in label])
            sentences.append(sentence)
        return sentences
