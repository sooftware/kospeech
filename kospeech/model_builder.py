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

from kospeech.models.conformer import Conformer
from kospeech.vocabs import Vocabulary
from kospeech.models.las import EncoderRNN
from kospeech.decode.ensemble import (
    BasicEnsemble,
    WeightedEnsemble,
)
from kospeech.models import (
    ListenAttendSpell,
    DeepSpeech2,
    SpeechTransformer,
    Jasper,
    RNNTransducer,
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
            input_dim=input_size,
            d_model=config.model.d_model,
            d_ff=config.model.d_ff,
            num_heads=config.model.num_heads,
            pad_id=vocab.pad_id,
            sos_id=vocab.sos_id,
            eos_id=vocab.eos_id,
            max_length=config.model.max_len,
            num_encoder_layers=config.model.num_encoder_layers,
            num_decoder_layers=config.model.num_decoder_layers,
            dropout_p=config.model.dropout,
            device=device,
            joint_ctc_attention=config.model.joint_ctc_attention,
            extractor=config.model.extractor,
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

    elif config.model.architecture.lower() == 'jasper':
        model = build_jasper(
            num_classes=len(vocab),
            version=config.model.version,
            device=device,
        )

    elif config.model.architecture.lower() == 'conformer':
        model = build_conformer(
            num_classes=len(vocab),
            input_size=input_size,
            encoder_dim=config.model.encoder_dim,
            decoder_dim=config.model.decoder_dim,
            num_encoder_layers=config.model.num_encoder_layers,
            num_decoder_layers=config.model.num_decoder_layers,
            decoder_rnn_type=config.model.decoder_rnn_type,
            num_attention_heads=config.model.num_attention_heads,
            feed_forward_expansion_factor=config.model.feed_forward_expansion_factor,
            conv_expansion_factor=config.model.conv_expansion_factor,
            input_dropout_p=config.model.input_dropout_p,
            feed_forward_dropout_p=config.model.feed_forward_dropout_p,
            attention_dropout_p=config.model.attention_dropout_p,
            conv_dropout_p=config.model.conv_dropout_p,
            decoder_dropout_p=config.model.decoder_dropout_p,
            conv_kernel_size=config.model.conv_kernel_size,
            half_step_residual=config.model.half_step_residual,
            device=device,
            decoder=config.model.decoder,
        )

    elif config.model.architecture.lower() == 'rnnt':
        model = build_rnnt(
            num_classes=len(vocab),
            input_dim=input_size,
            num_encoder_layers=config.model.num_encoder_layers,
            num_decoder_layers=config.model.num_decoder_layers,
            encoder_hidden_state_dim=config.model.encoder_hidden_state_dim,
            decoder_hidden_state_dim=config.model.decoder_hidden_state_dim,
            output_dim=config.model.output_dim,
            rnn_type=config.model.rnn_type,
            bidirectional=config.model.bidirectional,
            encoder_dropout_p=config.model.encoder_dropout_p,
            decoder_dropout_p=config.model.decoder_dropout_p,
            sos_id=vocab.sos_id,
            eos_id=vocab.eos_id,
        )

    else:
        raise ValueError('Unsupported model: {0}'.format(config.model.architecture))

    print(model)

    return model


def build_rnnt(
        num_classes: int,
        input_dim: int,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 1,
        encoder_hidden_state_dim: int = 320,
        decoder_hidden_state_dim: int = 512,
        output_dim: int = 512,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        encoder_dropout_p: float = 0.2,
        decoder_dropout_p: float = 0.2,
        sos_id: int = 1,
        eos_id: int = 2,
) -> nn.DataParallel:
    return nn.DataParallel(RNNTransducer(
        num_classes=num_classes,
        input_dim=input_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        encoder_hidden_state_dim=encoder_hidden_state_dim,
        decoder_hidden_state_dim=decoder_hidden_state_dim,
        output_dim=output_dim,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        encoder_dropout_p=encoder_dropout_p,
        decoder_dropout_p=decoder_dropout_p,
        sos_id=sos_id,
        eos_id=eos_id,
    ))


def build_conformer(
        num_classes: int,
        input_size: int,
        encoder_dim: int,
        decoder_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        decoder_rnn_type: str,
        num_attention_heads: int,
        feed_forward_expansion_factor: int,
        conv_expansion_factor: int,
        input_dropout_p: float,
        feed_forward_dropout_p: float,
        attention_dropout_p: float,
        conv_dropout_p: float,
        decoder_dropout_p: float,
        conv_kernel_size: int,
        half_step_residual: bool,
        device: torch.device,
        decoder: str,
) -> nn.DataParallel:
    if input_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if feed_forward_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if attention_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if conv_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if input_size < 0:
        raise ParameterError("input_size should be greater than 0")
    assert conv_expansion_factor == 2, "currently, conformer conv expansion factor only supports 2"

    return nn.DataParallel(Conformer(
        num_classes=num_classes,
        input_dim=input_size,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        decoder_rnn_type=decoder_rnn_type,
        num_attention_heads=num_attention_heads,
        feed_forward_expansion_factor=feed_forward_expansion_factor,
        conv_expansion_factor=conv_expansion_factor,
        input_dropout_p=input_dropout_p,
        feed_forward_dropout_p=feed_forward_dropout_p,
        attention_dropout_p=attention_dropout_p,
        conv_dropout_p=conv_dropout_p,
        decoder_dropout_p=decoder_dropout_p,
        conv_kernel_size=conv_kernel_size,
        half_step_residual=half_step_residual,
        device=device,
        decoder=decoder,
    )).to(device)


def build_deepspeech2(
        input_size: int,
        num_classes: int,
        rnn_type: str,
        num_rnn_layers: int,
        rnn_hidden_dim: int,
        dropout_p: float,
        bidirectional: bool,
        activation: str,
        device: torch.device,
) -> nn.DataParallel:
    if dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if input_size < 0:
        raise ParameterError("input_size should be greater than 0")
    if rnn_hidden_dim < 0:
        raise ParameterError("hidden_dim should be greater than 0")
    if num_rnn_layers < 0:
        raise ParameterError("num_layers should be greater than 0")
    if rnn_type.lower() not in EncoderRNN.supported_rnns.keys():
        raise ParameterError("Unsupported RNN Cell: {0}".format(rnn_type))

    return nn.DataParallel(DeepSpeech2(
        input_dim=input_size,
        num_classes=num_classes,
        rnn_type=rnn_type,
        num_rnn_layers=num_rnn_layers,
        rnn_hidden_dim=rnn_hidden_dim,
        dropout_p=dropout_p,
        bidirectional=bidirectional,
        activation=activation,
        device=device,
    )).to(device)


def build_transformer(
        num_classes: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        input_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        extractor: str,
        dropout_p: float,
        device: torch.device,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        joint_ctc_attention: bool = False,
        max_length: int = 400,
) -> nn.DataParallel:
    if dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if input_dim < 0:
        raise ParameterError("input_size should be greater than 0")
    if num_encoder_layers < 0:
        raise ParameterError("num_layers should be greater than 0")
    if num_decoder_layers < 0:
        raise ParameterError("num_layers should be greater than 0")
    return nn.DataParallel(SpeechTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        extractor=extractor,
        d_model=d_model,
        d_ff=d_ff,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        encoder_dropout_p=dropout_p,
        decoder_dropout_p=dropout_p,
        pad_id=pad_id,
        sos_id=sos_id,
        eos_id=eos_id,
        max_length=max_length,
        joint_ctc_attention=joint_ctc_attention,
    )).to(device)


def build_las(
        input_size: int,
        config: DictConfig,
        vocab: Vocabulary,
        device: torch.device,
) -> nn.DataParallel:
    model = ListenAttendSpell(
        input_dim=input_size,
        num_classes=len(vocab),
        encoder_hidden_state_dim=config.model.hidden_dim,
        decoder_hidden_state_dim=config.model.hidden_dim << (1 if config.model.use_bidirectional else 0),
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        bidirectional=config.model.use_bidirectional,
        extractor=config.model.extractor,
        activation=config.model.activation,
        rnn_type=config.model.rnn_type,
        max_length=config.model.max_len,
        pad_id=vocab.pad_id,
        sos_id=vocab.sos_id,
        eos_id=vocab.eos_id,
        attn_mechanism=config.model.attn_mechanism,
        num_heads=config.model.num_heads,
        encoder_dropout_p=config.model.dropout,
        decoder_dropout_p=config.model.dropout,
        joint_ctc_attention=config.model.joint_ctc_attention,
    )
    model.flatten_parameters()

    return nn.DataParallel(model).to(device)


def build_jasper(
    version: str,
    num_classes: int,
    device: torch.device,
) -> nn.DataParallel:
    assert version.lower() in ["10x5", "5x3"], "Unsupported Version: {}".format(version)
    return nn.DataParallel(Jasper(
        num_classes=num_classes,
        version=version,
        device=device,
    ))


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
