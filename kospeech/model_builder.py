import torch
import torch.nn as nn
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
    assert listener is not None, "listener is None"
    assert speller is not None, "speller is None"

    model = ListenAttendSpell(listener, speller)
    model.flatten_parameters()
    model = nn.DataParallel(model).to(device)

    if init_uniform:
        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)

    return model


def build_listener(input_size, hidden_dim, dropout_p, num_layers, bidirectional, rnn_type, extractor, activation, device):
    """ Various encoder dispatcher function. """
    assert isinstance(input_size, int), "input_size should be inteager type"
    assert isinstance(hidden_dim, int), "hidden_dim should be inteager type"
    assert isinstance(num_layers, int), "num_layers should be inteager type"
    assert dropout_p >= 0.0, "dropout probability should be positive"
    assert input_size > 0, "input_size should be greater than 0"
    assert hidden_dim > 0, "hidden_dim should be greater than 0"
    assert num_layers > 0, "num_layers should be greater than 0"
    assert extractor in {'vgg', 'ds2'}, "Unsupported extractor"
    assert rnn_type.lower() in BaseRNN.supported_rnns.keys(), "Unsupported RNN Cell: {0}".format(rnn_type)

    return Listener(input_size=input_size, hidden_dim=hidden_dim,
                    dropout_p=dropout_p, num_layers=num_layers,
                    bidirectional=bidirectional, rnn_type=rnn_type,
                    extractor=extractor, device=device, activation=activation)


def build_speller(num_classes, max_len, hidden_dim, sos_id, eos_id, attn_mechanism,
                  num_layers, rnn_type, dropout_p, num_heads, device):
    """ Various decoder dispatcher function. """
    assert isinstance(num_classes, int), "num_classes should be inteager type"
    assert isinstance(num_layers, int), "num_layers should be inteager type"
    assert isinstance(hidden_dim, int), "hidden_dim should be inteager type"
    assert isinstance(sos_id, int), "sos_id should be inteager type"
    assert isinstance(eos_id, int), "eos_id should be inteager type"
    assert isinstance(num_heads, int), "num_heads should be inteager type"
    assert isinstance(max_len, int), "max_len should be inteager type"
    assert isinstance(dropout_p, float), "dropout_p should be inteager type"
    assert hidden_dim % num_heads == 0, "{0} % {1} should be zero".format(hidden_dim, num_heads)
    assert dropout_p >= 0.0, "dropout probability should be positive"
    assert num_heads > 0, "num_heads should be greater than 0"
    assert hidden_dim > 0, "hidden_dim should be greater than 0"
    assert num_layers > 0, "num_layers should be greater than 0"
    assert max_len > 0, "max_len should be greater than 0"
    assert num_classes > 0, "num_classes should be greater than 0"
    assert rnn_type.lower() in BaseRNN.supported_rnns.keys(), "Unsupported RNN Cell: {0}".format(rnn_type)
    assert device is not None, "device is None"

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
