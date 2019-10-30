"""
Copyright 2019-present NAVER Corp.
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

import os
from nsml.constants import DATASET_PATH

def feed_infer(output_file, infer_func):

    filepath = os.path.join(DATASET_PATH, 'test', 'test_data', 'test_list.csv')

    with open(output_file, 'w') as of:

        with open(filepath, 'r') as f:

            for no, line in enumerate(f):

                # line : "abc.wav"

                wav_path = line.strip()
                wav_path = os.path.join(DATASET_PATH, 'test', 'test_data', wav_path)
                pred = infer_func(wav_path)

                of.write('%s,%s\n' % (wav_path, pred))
                print(wav_path, pred)

