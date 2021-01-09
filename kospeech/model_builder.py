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

import torch
import torch.nn as nn

from omegaconf import DictConfig
from astropy.modeling import ParameterError
from kospeech.models.modules import BaseRNN
from kospeech.vocabs import Vocabulary
from kospeech.decode.ensemble import (
    BasicEnsemble,
    WeightedEnsemble,
)
from kospeech.models import (
    ListenAttendSpell,
    Listener,
    Speller,
    DeepSpeech2,
    SpeechTransformer,
)


def build_model(
        config: DictConfig,
        vocab: Vocabulary,
        device: torch.device,
) -> nn.DataParallel:
    """ Various model dispatcher function. """
    if config.audio.transform_method.lower() == 'spect':
        if config.audio.feature_extract_by == 'kaldi':
            input_size = 257
        else:
            input_size = (config.audio.frame_length << 3) + 1
    else:
        input_size = config.audio.n_mels

    if config.model.architecture.lower() == 'las':
        model = build_las(input_size, config, vocab, device)

    elif config.model.architecture.lower() == 'transformer':
        model = build_transformer(
            num_classes=len(vocab),
            pad_id=vocab.pad_id,
            input_size=input_size,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            eos_id=vocab.eos_id,
            num_encoder_layers=config.model.num_encoder_layers,
            num_decoder_layers=config.model.num_decoder_layers,
            dropout_p=config.model.dropout,
            ffnet_style=config.model.ffnet_style,
            device=device,
            joint_ctc_attention=config.model.joint_ctc_attention,
        )

    elif config.model.architecture.lower() == 'deepspeech2':
        model = build_deepspeech2(
            input_size=input_size,
            num_classes=len(vocab),
            rnn_type=config.model.rnn_type,
            num_rnn_layers=config.model.num_encoder_layers,
            rnn_hidden_dim=config.model.hidden_dim,
            dropout_p=config.model.dropout,
            bidirectional=config.model.use_bidirectional,
            activation=config.model.activation,
            device=device,
        )

    else:
        raise ValueError('Unsupported model: {0}'.format(config.model.architecture))

    return model


def build_deepspeech2(input_size: int, num_classes: int, rnn_type: str, num_rnn_layers: int,
                      rnn_hidden_dim: int, dropout_p: float,
                      bidirectional: bool, activation: str, device: torch.device) -> nn.DataParallel:
    if dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if input_size < 0:
        raise ParameterError("input_size should be greater than 0")
    if rnn_hidden_dim < 0:
        raise ParameterError("hidden_dim should be greater than 0")
    if num_rnn_layers < 0:
        raise ParameterError("num_layers should be greater than 0")
    if rnn_type.lower() not in BaseRNN.supported_rnns.keys():
        raise ParameterError("Unsupported RNN Cell: {0}".format(rnn_type))

    return nn.DataParallel(DeepSpeech2(
            input_size=input_size,
            num_classes=num_classes,
            rnn_type=rnn_type,
            num_rnn_layers=num_rnn_layers,
            rnn_hidden_dim=rnn_hidden_dim,
            dropout_p=dropout_p,
            bidirectional=bidirectional,
            activation=activation,
            device=device,
    )).to(device)


def build_transformer(num_classes: int, pad_id: int, d_model: int, num_heads: int, input_size: int,
                      num_encoder_layers: int, num_decoder_layers: int,
                      dropout_p: float, ffnet_style: str, device: torch.device, eos_id: int,
                      joint_ctc_attention: bool) -> nn.DataParallel:
    if ffnet_style not in {'ff', 'conv'}:
        raise ParameterError("Unsupported ffnet_style: {0}".format(ffnet_style))

    return nn.DataParallel(SpeechTransformer(
            num_classes=num_classes,
            pad_id=pad_id,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout_p=dropout_p,
            ffnet_style=ffnet_style,
            input_dim=input_size,
            eos_id=eos_id,
            joint_ctc_attention=joint_ctc_attention,
    )).to(device)


def build_las(
        input_size: int,
        config: DictConfig,
        vocab: Vocabulary,
        device: torch.device
) -> nn.DataParallel:
    """ Various Listen, Attend and Spell dispatcher function. """
    listenr = build_listener(
            input_size=input_size,
            num_classes=len(vocab),
            hidden_dim=config.model.hidden_dim,
            dropout_p=config.model.dropout,
            num_layers=config.model.num_encoder_layers,
            bidirectional=config.model.use_bidirectional,
            extractor=config.model.extractor,
            activation=config.model.activation,
            rnn_type=config.model.rnn_type,
            device=device,
            mask_conv=config.model.mask_conv,
            joint_ctc_attention=config.model.joint_ctc_attention,
    )
    speller = build_speller(
            num_classes=len(vocab),
            max_len=config.model.max_len,
            pad_id=vocab.pad_id,
            sos_id=vocab.sos_id,
            eos_id=vocab.eos_id,
            hidden_dim=config.model.hidden_dim << (1 if config.model.use_bidirectional else 0),
            num_layers=config.model.num_decoder_layers,
            rnn_type=config.model.rnn_type,
            dropout_p=config.model.dropout,
            num_heads=config.model.num_heads,
            attn_mechanism=config.model.attn_mechanism,
            device=device,
    )

    model = ListenAttendSpell(listenr, speller)
    model.flatten_parameters()

    return nn.DataParallel(model).to(device)


def build_listener(
        input_size: int = 80,
        num_classes: int = None,
        hidden_dim: int = 512,
        dropout_p: float = 0.2,
        num_layers: int = 3,
        bidirectional: bool = True,
        rnn_type: str = 'lstm',
        extractor: str = 'vgg',
        activation: str = 'hardtanh',
        device: torch.device = 'cuda',
        mask_conv: bool = False,
        joint_ctc_attention: bool = False,
) -> Listener:
    """ Various encoder dispatcher function. """
    if dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if input_size < 0:
        raise ParameterError("input_size should be greater than 0")
    if hidden_dim < 0:
        raise ParameterError("hidden_dim should be greater than 0")
    if num_layers < 0:
        raise ParameterError("num_layers should be greater than 0")
    if extractor.lower() not in {'vgg', 'ds2'}:
        raise ParameterError("Unsupported extractor".format(extractor))
    if rnn_type.lower() not in BaseRNN.supported_rnns.keys():
        raise ParameterError("Unsupported RNN Cell: {0}".format(rnn_type))

    return Listener(
        input_size=input_size,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        num_layers=num_layers,
        mask_conv=mask_conv,
        bidirectional=bidirectional,
        rnn_type=rnn_type,
        extractor=extractor,
        device=device,
        activation=activation,
        joint_ctc_attention=joint_ctc_attention,
    )


def build_speller(
        num_classes: int,
        max_len: int,
        hidden_dim: int,
        sos_id: int,
        eos_id: int,
        pad_id: int,
        attn_mechanism: str,
        num_layers: int,
        rnn_type: str,
        dropout_p: float,
        num_heads: int,
        device: torch.device,
) -> Speller:
    """ Various decoder dispatcher function. """
    if hidden_dim % num_heads != 0:
        raise ParameterError("{0} % {1} should be zero".format(hidden_dim, num_heads))
    if dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if num_heads < 0:
        raise ParameterError("num_heads should be greater than 0")
    if hidden_dim < 0:
        raise ParameterError("hidden_dim should be greater than 0")
    if num_layers < 0:
        raise ParameterError("num_layers should be greater than 0")
    if max_len < 0:
        raise ParameterError("max_len should be greater than 0")
    if num_classes < 0:
        raise ParameterError("num_classes should be greater than 0")
    if rnn_type.lower() not in BaseRNN.supported_rnns.keys():
        raise ParameterError("Unsupported RNN Cell: {0}".format(rnn_type))
    if device is None:
        raise ParameterError("device is None")

    return Speller(
        num_classes=num_classes,
        max_length=max_len,
        hidden_dim=hidden_dim,
        pad_id=pad_id,
        sos_id=sos_id,
        eos_id=eos_id,
        attn_mechanism=attn_mechanism,
        num_heads=num_heads,
        num_layers=num_layers,
        rnn_type=rnn_type,
        dropout_p=dropout_p,
        device=device,
    )


def load_test_model(config: DictConfig, device: torch.device):
    model = torch.load(config.model_path, map_location=lambda storage, loc: storage).to(device)

    if isinstance(model, nn.DataParallel):
        model.module.decoder.device = device
        model.module.encoder.device = device

    else:
        model.encoder.device = device
        model.decoder.device = device

    return model


def load_language_model(path: str, device: torch.device):
    model = torch.load(path, map_location=lambda storage, loc: storage).to(device)

    if isinstance(model, nn.DataParallel):
        model = model.module

    model.device = device

    return model


def build_ensemble(model_paths: list, method: str, device: torch.device):
    models = list()

    for model_path in model_paths:
        models.append(torch.load(model_path, map_location=lambda storage, loc: storage))

    if method == 'basic':
        ensemble = BasicEnsemble(models).to(device)
    elif method == 'weight':
        ensemble = WeightedEnsemble(models).to(device)
    else:
        raise ValueError("Unsupported ensemble method : {0}".format(method))

    return ensemble
