import torch
import torch.nn as nn
from astropy.modeling import ParameterError
from kospeech.decode.ensemble import BasicEnsemble, WeightedEnsemble
from kospeech.model.encoder import BaseRNN
from kospeech.model.seq2seq import ListenAttendSpell
from kospeech.model.encoder import Listener
from kospeech.model.decoder import Speller
from kospeech.utils import char2id, EOS_token, SOS_token


def build_model(opt, device):
    """ Various model dispatcher function. """
    listener = build_listener(input_size=opt.n_mels, hidden_dim=opt.hidden_dim, dropout_p=opt.dropout,
                              num_layers=opt.listener_layer_size, bidirectional=opt.use_bidirectional,
                              extractor=opt.extractor, activation=opt.activation, rnn_type=opt.rnn_type, device=device)
    speller = build_speller(num_classes=len(char2id), max_len=opt.max_len, sos_id=SOS_token, eos_id=EOS_token,
                            hidden_dim=opt.hidden_dim << (1 if opt.use_bidirectional else 0),
                            num_layers=opt.speller_layer_size, rnn_type=opt.rnn_type, dropout_p=opt.dropout,
                            num_heads=opt.num_heads, attn_mechanism=opt.attn_mechanism, device=device)

    return build_las(listener, speller, device, opt.init_uniform)


def build_las(listener, speller, device, init_uniform=True):
    """ Various Listen, Attend and Spell dispatcher function. """
    if listener is None:
        raise ParameterError("listener should not be None")
    if speller is None:
        raise ParameterError("speller should not be None")

    model = ListenAttendSpell(listener, speller)
    model.flatten_parameters()
    model = nn.DataParallel(model).to(device)

    if init_uniform:
        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)

    return model


def build_listener(input_size, hidden_dim, dropout_p, num_layers, bidirectional, rnn_type, extractor, activation, device):
    """ Various encoder dispatcher function. """
    if not isinstance(input_size, int):
        raise ParameterError("input_size should be inteager type")
    if not isinstance(hidden_dim, int):
        raise ParameterError("hidden_dim should be inteager type")
    if not isinstance(num_layers, int):
        raise ParameterError("num_layers should be inteager type")
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

    return Listener(input_size=input_size, hidden_dim=hidden_dim,
                    dropout_p=dropout_p, num_layers=num_layers,
                    bidirectional=bidirectional, rnn_type=rnn_type,
                    extractor=extractor, device=device, activation=activation)


def build_speller(num_classes, max_len, hidden_dim, sos_id, eos_id, attn_mechanism,
                  num_layers, rnn_type, dropout_p, num_heads, device):
    """ Various decoder dispatcher function. """
    if not isinstance(num_classes, int):
        raise ParameterError("num_classes should be inteager type")
    if not isinstance(num_layers, int):
        raise ParameterError("num_layers should be inteager type")
    if not isinstance(hidden_dim, int):
        raise ParameterError("hidden_dim should be inteager type")
    if not isinstance(sos_id, int):
        raise ParameterError("sos_id should be inteager type")
    if not isinstance(eos_id, int):
        raise ParameterError("eos_id should be inteager type")
    if not isinstance(num_heads, int):
        raise ParameterError("num_heads should be inteager type")
    if not isinstance(max_len, int):
        raise ParameterError("max_len should be inteager type")
    if not isinstance(dropout_p, float):
        raise ParameterError("dropout_p should be float type")
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

    return Speller(num_classes=num_classes, max_length=max_len,
                   hidden_dim=hidden_dim, sos_id=sos_id, eos_id=eos_id,
                   attn_mechanism=attn_mechanism, num_heads=num_heads,
                   num_layers=num_layers, rnn_type=rnn_type,
                   dropout_p=dropout_p, device=device)


def load_test_model(opt, device):
    model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(device)

    if isinstance(model, nn.DataParallel):
        model.module.speller.device = device
        model.module.listener.device = device

    else:
        model.speller.device = device
        model.listener.device = device

    return model


def load_language_model(path, device):
    model = torch.load(path, map_location=lambda storage, loc: storage).to(device)

    if isinstance(model, nn.DataParallel):
        model = model.module

    model.device = device

    return model


def build_ensemble(model_paths, method, device):
    ensemble = None
    models = list()

    for idx in range(len(model_paths)):
        models.append(torch.load(model_paths[idx], map_location=lambda storage, loc: storage))

    if method == 'basic':
        ensemble = BasicEnsemble(models).to(device)
    elif method == 'weight':
        ensemble = WeightedEnsemble(models).to(device)
    else:
        raise ValueError("Unsupported ensemble method : {0}".format(method))

    return ensemble
