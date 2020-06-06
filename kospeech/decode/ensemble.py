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
        hypothesis = None

        with torch.no_grad():
            for model in self.models:
                if hypothesis is None:
                    hypothesis = model(inputs, input_lengths, teacher_forcing_ratio=0.0)
                else:
                    hypothesis += model(inputs, input_lengths, teacher_forcing_ratio=0.0)

        return hypothesis


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
        hypothesis = None
        outputs = list()
        weights = torch.FloatTensor([1.] * self.num_models)

        # model`s parameters are fixed
        with torch.no_grad():
            for model in self.models:
                outputs.append(model(inputs, input_lengths, teacher_forcing_ratio=0.0))

        weights = self.meta_classifier(weights)

        for (output, weight) in zip(outputs, weights):
            if hypothesis is None:
                hypothesis = output * weight
            else:
                hypothesis += output * weight

        return hypothesis
