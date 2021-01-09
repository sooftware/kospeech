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

# Note! This code is not available !!

import torch
import torch.nn as nn


class Ensemble(nn.Module):
    """
    Ensemble decoding.
    Decodes using multiple models simultaneously,

    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        self.num_models = len(models)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class BasicEnsemble(Ensemble):
    """
    Basic ensemble decoding.

    Decodes using multiple models simultaneously,
    combining their prediction distributions by adding.
    All models in the ensemble must share a target characters.
    """
    def __init__(self, models):
        super(BasicEnsemble, self).__init__(models)

    def forward(self, inputs, input_lengths):
        y_hats = None

        with torch.no_grad():
            for model in self.models:
                if y_hats is None:
                    y_hats = model(inputs, input_lengths, teacher_forcing_ratio=0.0)
                else:
                    y_hats += model(inputs, input_lengths, teacher_forcing_ratio=0.0)

        return y_hats


class WeightedEnsemble(Ensemble):
    """
    Weighted ensemble decoding.

    Decodes using multiple models simultaneously,
    combining their prediction distributions by weighted sum.
    All models in the ensemble must share a target characters.
    """
    def __init__(self, models, dim=128):
        super(WeightedEnsemble, self).__init__(models)
        self.meta_classifier = nn.Sequential(
            nn.Linear(self.num_models, dim),
            nn.ELU(inplace=True),
            nn.Linear(dim, self.num_models)
        )

    def forward(self, inputs, input_lengths):
        y_hats, outputs = None, list()
        weights = torch.FloatTensor([1.] * self.num_models)

        # model`s parameters are fixed
        with torch.no_grad():
            for model in self.models:
                outputs.append(model(inputs, input_lengths, teacher_forcing_ratio=0.0))

        weights = self.meta_classifier(weights)

        for (output, weight) in zip(outputs, weights):
            if y_hats is None:
                y_hats = output * weight
            else:
                y_hats += output * weight

        return y_hats
