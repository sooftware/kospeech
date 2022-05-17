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


class Vocabulary(object):
    """
    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, *args, **kwargs):
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None
        self.blank_id = None

    def label_to_string(self, labels):
        raise NotImplementedError


from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.vocabs.librispeech import LibriSpeechVocabulary
