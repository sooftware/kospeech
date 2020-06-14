import random
import torch.nn as nn
import torch.nn.functional as F
from kospeech.model.encoder import BaseRNN


class RNNLM(BaseRNN):
    def __init__(self, num_classes, num_layers, rnn_type, hidden_dim,
                 dropout_p, max_length, sos_id, eos_id, device):
        super(RNNLM, self).__init__(hidden_dim, hidden_dim, num_layers, rnn_type, dropout_p, False, device)
        self.rnn_cell = self.supported_rnns[rnn_type]
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward_step(self, input_var, hidden):
        """ forward one time step """
        batch_size = input_var.size(0)
        output_length = input_var.size(1)

        embedded = self.embedding(input_var).to(self.device)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(embedded, hidden)

        predicted_softmax = F.log_softmax(self.fc(output.contiguous().view(-1, self.hidden_dim)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, output_length, -1)

        return predicted_softmax, hidden

    def forward(self, inputs, teacher_forcing_ratio=1.0):
        batch_size = inputs.size(0)
        max_length = inputs.size(1) - 1  # minus the start of sequence symbol
        hidden = None

        decode_outputs = list()
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            inputs = inputs[inputs != self.eos_id].view(batch_size, -1)
            predicted_softmax, hidden = self.forward_step(inputs, hidden)

            for di in range(predicted_softmax.size(1)):
                step_output = predicted_softmax[:, di, :]
                decode_outputs.append(step_output)

        else:
            input_var = inputs[:, 0].unsqueeze(1)

            for di in range(max_length):
                predicted_softmax, hidden = self.forward_step(input_var, hidden)

                step_output = predicted_softmax.squeeze(1)
                decode_outputs.append(step_output)
                input_var = decode_outputs[-1].topk(1)[1]

        return decode_outputs

    def flatten_parameters(self):
        self.rnn.flatten_parameters()
