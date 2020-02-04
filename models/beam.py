import torch

class Beam:
    def __init__(self, eos_id, beam_size, speller_input, embedding, input_dropout, rnn,
                 use_attention, attention, out, hidden_size, listener_outputs, function, speller_hidden, batch_size):
        self.eos_id = eos_id
        self.symbols = torch.FloatTensor([[0] * beam_size] * batch_size)
        self.probs = torch.FloatTensor([[0] * beam_size] * batch_size)
        self.beam_size = beam_size
        self.speller_input = speller_input
        self.embedding = embedding
        self.input_dropout = input_dropout
        self.rnn = rnn
        self.use_attention = use_attention
        self.attention = attention
        self.out= out
        self.hidden_size = hidden_size
        self.candidate_probs = None
        self.candidate_symbols = None
        self.listener_outputs = listener_outputs
        self.function = function
        self.speller_hidden = speller_hidden

    def search(self):
        """
        :param speller_input: labels (except </s>)
        :param speller_hidden: hidden state of speller
        :param listener_outputs: output of listener
        :param function: decode function
        """
        batch_size = self.speller_input.size(0)   # speller_input.size(0) : batch_size
        output_size = self.speller_input.size(1)  # speller_input.size(1) : seq_len
        embedded = self.embedding(self.speller_input)
        embedded = self.input_dropout(embedded)
        speller_output, hidden = self.rnn(embedded, self.speller_hidden) # speller output
        attn = None
        if self.use_attention:
            output, attn = self.attention(decoder_output=speller_output, encoder_output=self.listener_outputs)
        else: output = speller_output
        # torch.view()에서 -1이면 나머지 알아서 맞춰줌
        predicted_softmax = self.function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        step_output = predicted_softmax.squeeze(1)
        self.candidate_probs, self.candidate_symbols = step_output.topk(self.beam_size)

        return self.probs + self.candidate_probs # log probability

    def forward(self, candidate_idx):
        """
        (+) length_penalty 추가해줘야함

        """
        symbol = self.candidate_symbols[candidate_idx]
        self.symbols.append(symbol)
        self.probs += self.candidate_probs[candidate_idx]
        self.speller_input = symbol
        return symbol == self.eos_id

    def reset_beam(self, symbols, probs, candidate_symbols, candidate_probs):
        self.symbols = symbols
        self.probs = probs
        self.candidate_symbols = candidate_symbols
        self.candidate_probs = candidate_probs