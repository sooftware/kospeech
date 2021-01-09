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

from kospeech.models.las.encoder import Listener
from kospeech.models.las.decoder import Speller
from kospeech.models.las.model import ListenAttendSpell
from kospeech.models.transformer.model import SpeechTransformer


encoder = Listener(80, 512, 'cpu')
decoder = Speller(2038, 151, 1024, 1, 2)
model = ListenAttendSpell(encoder, decoder)

print(model)


model = SpeechTransformer(num_classes=2038, num_encoder_layers=3, num_decoder_layers=3)
print(model)
