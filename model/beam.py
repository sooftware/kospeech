import torch
import numpy as np

class Beam:
    r"""
    Applying Beam-Search during decoding process.

    Args:
        k (int) : size of beam
        decoder_hidden (torch.Tensor) : hidden state of decoder
        batch_size (int) : mini-batch size during infer
        max_len (int) :  a maximum allowed length for the sequence to be processed
        function (torch.nn) : A function used to generate symbols from RNN hidden state (default : torch.nn.functional.log_softmax)
        decoder (torch.nn) : get pointer of decoder object to get multiple parameters at once
        beams (torch.Tensor) : ongoing beams for decoding
        beam_scores (torch.Tensor) : score of beams (cumulative probability)
        done_beams (list) : store beams which met <eos> token and terminated decoding process.
        done_beam_scores (list) : score of done_beams

    Inputs: decoder_input, encoder_outputs
        - **decoder_input** (torch.Tensor): initial input of decoder - <sos>
        - **encoder_outputs** (torch.Tensor): tensor with containing the outputs of the encoder.

    Returns: y_hats
        - **y_hats** (batch, seq_len): predicted y values (y_hat) by the model

    Examples::

        >>> beam = Beam(k, decoder_hidden, decoder, batch_size, max_len, F.log_softmax)
        >>> y_hats = beam.search(inputs, encoder_outputs)
    """

    def __init__(self, k, decoder_hidden, decoder, batch_size, max_len, function, device):
        assert k > 1, "beam size (k) should be bigger than 1"
        self.k = k
        self.decoder_hidden = decoder_hidden
        self.batch_size = batch_size
        self.max_len = max_len
        self.function = function
        self.rnn = decoder.rnn
        self.embedding = decoder.embedding
        self.input_dropout = decoder.input_dropout
        self.use_attention = decoder.use_attention
        self.attention = decoder.attention
        self.hidden_size = decoder.hidden_size
        self.out = decoder.out
        self.eos_id = decoder.eos_id
        self.beams = None
        self.beam_scores = None
        self.done_beams = [[] for _ in range(self.batch_size)]
        self.done_beam_scores = [[] for _ in range(self.batch_size)]
        self.device = device

    def search(self, decoder_input, encoder_outputs):
        """ Beam-Search Decoding (Top-K Decoding) """
        # Comment Notation
        # B : batch size
        # K : beam size
        # C : classfication number
        # S : sequence length
        # get class classfication distribution (shape: BxC)
        step_outputs = self._forward_step(decoder_input, encoder_outputs).squeeze(1)
        # get top K probability & index (shape: BxK)
        self.beam_scores, self.beams = step_outputs.topk(self.k)
        decoder_input = self.beams
        # transpose (BxK) => (BxKx1)
        self.beams = self.beams.view(self.batch_size, self.k, 1)
        for di in range(self.max_len-1):
            if self._is_done():
                break
            # For each beam, get class classfication distribution (shape: BxKxC)
            predicted_softmax = self._forward_step(decoder_input, encoder_outputs)
            step_output = predicted_softmax.squeeze(1)
            # get top k distribution (shape: BxKxK)
            child_ps, child_vs = step_output.topk(self.k)
            # get child probability (applying length penalty)
            child_ps = (self.beam_scores.view(self.batch_size, 1, self.k) + child_ps) * self._get_length_penalty(length=di+1, alpha=1.2, min_length=5)
            # Transpose (BxKxK) => (BxK^2)
            child_ps, child_vs = child_ps.view(self.batch_size, self.k * self.k), child_vs.view(self.batch_size, self.k * self.k)
            # Select Top k in K^2 (shape: BxK)
            topk_child_ps, topk_child_indices = child_ps.topk(self.k)
            # Initiate topk_child_vs (shape: BxK)
            topk_child_vs = torch.LongTensor(self.batch_size, self.k)
            # Initiate parent_beams (shape: BxKxS)
            parent_beams = torch.LongTensor(self.beams.size(0), self.beams.size(1), self.beams.size(2))
            # indices // k => indices of topk_child`s parent node
            parent_beams_indices = (topk_child_indices // self.k).view(self.batch_size, self.k)

            for batch_num, batch in enumerate(topk_child_indices):
                for beam_num, topk_child_index in enumerate(batch):
                    topk_child_vs[batch_num, beam_num] = child_vs[batch_num, topk_child_index]
                    parent_beams[batch_num, beam_num] = self.beams[batch_num, parent_beams_indices[batch_num, beam_num]]
            # append new_topk_child (shape: BxKx(S) => BxKx(S+1))
            self.beams = torch.cat([parent_beams, topk_child_vs.view(self.batch_size, self.k, 1)], dim=2)
            self.beam_scores = topk_child_ps

            if torch.any(topk_child_vs == self.eos_id):
                done_indices = torch.where(topk_child_vs == self.eos_id)
                count = [1] * self.batch_size
                for done_index in done_indices:
                    batch_num, beam_num = done_index[0], done_index[1]
                    self.done_beams[batch_num].append(self.beams[batch_num, beam_num])
                    self.done_beam_scores[batch_num].append(self.beam_scores[batch_num, beam_num])
                    self._replace_beam(
                        child_ps=child_ps,
                        child_vs=child_vs,
                        done_beam_index=[batch_num, beam_num],
                        count=count[batch_num]
                    )
                    count[batch_num] += 1
            # update decoder_input by topk_child_vs
            decoder_input = topk_child_vs
        y_hats = self._get_best()
        return y_hats

    def _get_best(self):
        """ get sentences which has the highest probability at each batch, stack it, and return it as 2d torch """
        y_hats = []

        # done_beams has <eos> terminate sentences during decoding process
        for batch_num, batch in enumerate(self.done_beams):
            if len(batch) == 0:
                # if there is no terminated sentences, bring ongoing sentence which has the highest probability instead
                self.beam_scores = torch.Tensor(self.beam_scores[batch_num]).to(self.device)
                top_beam_index = int(self.beam_scores.topk(1)[1])
                y_hats.append(self.beams[batch_num, top_beam_index])
            else:
                # bring highest probability sentence
                top_beam_index = int(torch.Tensor(self.done_beam_scores[batch_num]).topk(1)[1])
                y_hats.append(self.done_beams[batch_num][top_beam_index])
        y_hats = self._match_len(y_hats).to(self.device)
        return y_hats

    def _match_len(self, y_hats):
        max_len = -1
        for y_hat in y_hats:
            if len(y_hat) > max_len:
                max_len = len(y_hat)

        matched = torch.LongTensor(self.batch_size, max_len)
        for batch_num, y_hat in enumerate(y_hats):
            matched[batch_num, :len(y_hat)] = y_hat
            matched[batch_num, len(y_hat):] = 0

        return matched

    def _is_done(self):
        """ check if all beam search process has terminated """
        for done in self.done_beams:
            if len(done) < self.k:
                return False
        return True

    def _forward_step(self, decoder_input, encoder_outputs):
        """ forward one step on each decoder cell """
        decoder_input = decoder_input.to(self.device)
        output_size = decoder_input.size(1)
        embedded = self.embedding(decoder_input).to(self.device)
        embedded = self.input_dropout(embedded)
        decoder_output, hidden = self.rnn(embedded, self.decoder_hidden)  # decoder output

        if self.use_attention:
            output = self.attention(decoder_output, encoder_outputs)
        else:
            output = decoder_output
        predicted_softmax = self.function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1)
        predicted_softmax = predicted_softmax.view(self.batch_size,output_size,-1)
        return predicted_softmax

    def _get_length_penalty(self, length, alpha=1.2, min_length=5):
        """
        Calculate length-penalty.
        because shorter sentence usually have bigger probability.
        using alpha = 1.2, min_length = 5 usually.
        """
        return ((1+length) / (1+min_length)) ** alpha

    def _replace_beam(self, child_ps, child_vs, done_beam_index, count):
        """ Replaces a beam that ends with <eos> with a beam with the next higher probability. """
        done_batch_num, done_beam_num = done_beam_index[0], done_beam_index[1]
        tmp_indices = child_ps.topk(self.k + count)[1]
        new_child_index = tmp_indices[done_batch_num, -1]
        new_child_p = child_ps[done_batch_num, new_child_index].to(self.device)
        new_child_v = child_vs[done_batch_num, new_child_index].to(self.device)
        parent_beam_index = (new_child_index // self.k)
        parent_beam = self.beams[done_batch_num, parent_beam_index].to(self.device)
        parent_beam = parent_beam[:-1]
        new_beam = torch.cat([parent_beam, new_child_v.view(1)])
        self.beams[done_batch_num, done_beam_num] = new_beam
        self.beam_scores[done_batch_num, done_beam_num] = new_child_p