"""
Copyright 2020- Kai.Lib
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import Levenshtein as Lev
from label.label_func import label_to_string
from definition import logger, test_index2char
from definition import index2char, EOS_token

def char_distance(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')
    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length

def get_distance(ref_labels, hyp_labels, display=False):
    total_dist = 0
    total_length = 0

    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i], index2char, EOS_token)
        hyp = label_to_string(hyp_labels[i], index2char, EOS_token)
        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length
        if display:
            cer = total_dist / total_length
            logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
    return total_dist, total_length