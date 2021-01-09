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

import pandas as pd
import matplotlib.pyplot as plt

RGB = (0.4157, 0.2784, 0.3333)
TRAIN_RESULT_PATH = '../data/train_result/train_step_result.csv'


train_result = pd.read_csv(TRAIN_RESULT_PATH, delimiter=',', encoding='cp949')
losses = train_result['loss']
cers = train_result['cer']


plt.title('Visualization of training (loss)')
plt.plot(losses, color=RGB, label='losses')
plt.xlabel('step (unit : 1000)', fontsize='x-large')
plt.ylabel('L2norm', fontsize='x-large')
plt.show()
