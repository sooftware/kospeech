import torch
import torch.nn as nn
from e2e.modules.definition import char2id, EOS_token, SOS_token
from e2e.las.las import ListenAttendSpell
from e2e.las.listener import Listener
from e2e.las.speller import Speller
from e2e.las.topk_decoder import TopKDecoder


def build_model(opt, device):
    """ build base model """
    listener = build_listener(
        input_size=opt.n_mels,
        hidden_dim=opt.hidden_dim,
        dropout_p=opt.dropout,
        num_layers=opt.listener_layer_size,
        bidirectional=opt.use_bidirectional,
        rnn_type=opt.rnn_type,
        device=device,
    )
    speller = build_speller(
        num_classes=len(char2id),
        max_len=opt.max_len,
        hidden_dim=opt.hidden_dim << (1 if opt.use_bidirectional else 0),
        sos_id=SOS_token,
        eos_id=EOS_token,
        num_layers=opt.speller_layer_size,
        rnn_type=opt.rnn_type,
        dropout_p=opt.dropout,
        num_heads=opt.num_heads,
        device=device
    )

    return build_las(listener, speller, device, opt.use_multi_gpu, opt.init_uniform)


def build_las(listener, speller, device, use_multi_gpu=True, init_uniform=True):
    """ build las model & validate parameters """
    assert listener is not None, "listener is None"
    assert speller is not None, "speller is None"

    model = ListenAttendSpell(listener, speller)
    model.flatten_parameters()

    if use_multi_gpu:
        model = nn.DataParallel(model).to(device)

    if init_uniform:
        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)

    return model


def build_listener(input_size, hidden_dim, dropout_p, num_layers, bidirectional, rnn_type, device):
    """ build listener & validate parameters """
    assert isinstance(input_size, int), "input_size should be inteager type"
    assert isinstance(hidden_dim, int), "hidden_dim should be inteager type"
    assert isinstance(num_layers, int), "num_layers should be inteager type"
    assert dropout_p >= 0.0, "dropout probability should be positive"
    assert input_size > 0, "input_size should be greater than 0"
    assert hidden_dim > 0, "hidden_dim should be greater than 0"
    assert num_layers > 0, "num_layers should be greater than 0"
    assert rnn_type.lower() in Listener.supported_rnns.keys(), "Unsupported RNN Cell: {0}".format(rnn_type)

    return Listener(input_size, hidden_dim, device, dropout_p, num_layers, bidirectional, rnn_type)


def build_speller(num_classes, max_len, hidden_dim, sos_id, eos_id, num_layers, rnn_type, dropout_p, num_heads, device):
    """ build speller & validate parameters """
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
    assert rnn_type.lower() in Speller.supported_rnns.keys(), "Unsupported RNN Cell: {0}".format(rnn_type)
    assert device is not None, "device is None"

    return Speller(num_classes, max_len, hidden_dim, sos_id, eos_id, num_heads, num_layers, rnn_type, dropout_p, device)


def load_test_model(opt, device, use_beamsearch=True):
    """ load model for performance test """
    model = torch.load(opt.model_path, map_location=lambda storage, loc: storage)

    if isinstance(model, nn.DataParallel):
        model.module.speller.device = device
        model.module.listener.device = device

        if use_beamsearch:
            topk_decoder = TopKDecoder(model.module.speller, opt.k)
            model.module.set_speller(topk_decoder)

    else:
        model.speller.device = device
        model.listener.device = device

        if use_beamsearch:
            topk_decoder = TopKDecoder(model.speller, opt.k)
            model.set_speller(topk_decoder)

    return model
