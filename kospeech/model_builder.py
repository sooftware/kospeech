# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from astropy.modeling import ParameterError
from kospeech.decode.ensemble import BasicEnsemble, WeightedEnsemble
from kospeech.models.modules import BaseRNN
from kospeech.models.las.las import ListenAttendSpell
from kospeech.models.las.encoder import Listener
from kospeech.models.las.decoder import Speller
from kospeech.models.transformer.transformer import SpeechTransformer
from kospeech.utils import char2id, EOS_token, SOS_token, PAD_token


def build_model(opt, device):
    """ Various model dispatcher function. """
    if opt.transform_method.lower() == 'spect':
        if opt.feature_extract_by == 'kaldi':
            input_size = 257
        else:
            input_size = (opt.frame_length << 3) + 1
    else:
        input_size = opt.n_mels

    if opt.architecture.lower() == 'las':
        model = build_las(input_size, opt, device)

    elif opt.architecture.lower() == 'transformer':
        model = build_transformer(
            num_classes=opt.num_classes,
            pad_id=PAD_token,
            input_size=input_size,
            d_model=opt.d_model,
            num_heads=opt.num_heads,
            eos_id=EOS_token,
            num_encoder_layers=opt.num_encoder_layers,
            num_decoder_layers=opt.num_decoder_layers,
            dropout_p=opt.dropout,
            ffnet_style=opt.ffnet_style,
            device=device
        )

    else:
        raise ValueError('Unsupported architecture: {0}'.format(opt.architecture))

    return model


def build_transformer(num_classes: int, pad_id: int, d_model: int, num_heads: int, input_size: int,
                      num_encoder_layers: int, num_decoder_layers: int,
                      dropout_p: float, ffnet_style: str, device: str, eos_id: int) -> nn.DataParallel:
    if ffnet_style not in {'ff', 'conv'}:
        raise ParameterError("Unsupported ffnet_style: {0}".format(ffnet_style))

    return nn.DataParallel(
        SpeechTransformer(
            num_classes=num_classes,
            pad_id=pad_id,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout_p=dropout_p,
            ffnet_style=ffnet_style,
            input_dim=input_size,
            eos_id=eos_id
        )
    ).to(device)


def build_las(input_size, opt, device):
    """ Various Listen, Attend and Spell dispatcher function. """
    model = ListenAttendSpell(
        build_listener(
            input_size=input_size,
            hidden_dim=opt.hidden_dim,
            dropout_p=opt.dropout,
            num_layers=opt.num_encoder_layers,
            bidirectional=opt.use_bidirectional,
            extractor=opt.extractor,
            activation=opt.activation,
            rnn_type=opt.rnn_type,
            device=device,
            mask_conv=opt.mask_conv
        ),
        build_speller(
            num_classes=len(char2id),
            max_len=opt.max_len,
            pad_id=PAD_token,
            sos_id=SOS_token,
            eos_id=EOS_token,
            hidden_dim=opt.hidden_dim << (1 if opt.use_bidirectional else 0),
            num_layers=opt.num_decoder_layers,
            rnn_type=opt.rnn_type,
            dropout_p=opt.dropout,
            num_heads=opt.num_heads,
            attn_mechanism=opt.attn_mechanism,
            device=device
        )
    )
    model.flatten_parameters()

    return nn.DataParallel(model).to(device)


def build_listener(
        input_size: int = 80,
        hidden_dim: int = 512,
        dropout_p: float = 0.2,
        num_layers: int = 3,
        bidirectional: bool = True,
        rnn_type: str = 'lstm',
        extractor: str = 'vgg',
        activation: str = 'hardtanh',
        device: str = 'cuda',
        mask_conv: bool = False
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
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        num_layers=num_layers,
        mask_conv=mask_conv,
        bidirectional=bidirectional,
        rnn_type=rnn_type,
        extractor=extractor,
        device=device,
        activation=activation
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
        device: str
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
        device=device
    )


def load_test_model(opt, device):
    model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(device)

    if isinstance(model, nn.DataParallel):
        model.module.decoder.device = device
        model.module.encoder.device = device

    else:
        model.encoder.device = device
        model.decoder.device = device

    return model


def load_language_model(path, device):
    model = torch.load(path, map_location=lambda storage, loc: storage).to(device)

    if isinstance(model, nn.DataParallel):
        model = model.module

    model.device = device

    return model


def build_ensemble(model_paths, method, device):
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
