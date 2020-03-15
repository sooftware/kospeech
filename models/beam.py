import torch

class Beam:
    r"""
    Applying Beam-Search during decoding process.

    Args:
        k (int) : size of beam
        decoder_hidden (torch.Tensor) : hidden state of decoder
        batch_size (int) : mini-batch size during infer
        max_len (int) :  a maximum allowed length for the sequence to be processed
        function (torch.nn.Module) : A function used to generate symbols from RNN hidden state
        (default : torch.nn.functional.log_softmax)
        decoder (torch.nn.Module) : get pointer of decoder object to get multiple parameters at once
        beams (torch.Tensor) : ongoing beams for decoding
        probs (torch.Tensor) : cumulative probability of beams (score of beams)
        sentences (list) : store beams which met <eos> token and terminated decoding process.
        sentence_probs (list) : score of sentences

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
        self.w = decoder.w
        self.eos_id = decoder.eos_id
        self.beams = None
        self.probs = None
        self.sentences = [[] for _ in range(self.batch_size)]
        self.sentence_probs = [[] for _ in range(self.batch_size)]
        self.device = device

    def search(self, decoder_input, encoder_outputs):
        """
        Beam-Search Decoding (Top-K Decoding)

        Examples::

            >>> beam = Beam(k, decoder_hidden, decoder, batch_size, max_len, F.log_softmax)
            >>> y_hats = beam.search(inputs, encoder_outputs)
        """
        # Comment Notation
        # B : batch size
        # K : beam size
        # C : classfication number
        # S : sequence length
        # get class classfication distribution (shape: BxC)
        step_outputs = self._forward_step(decoder_input, encoder_outputs).squeeze(1)
        # get top K probability & idx (shape: BxK)
        self.probs, self.beams = step_outputs.topk(self.k)
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
            child_ps = self.probs.view(self.batch_size, 1, self.k) + child_ps
            child_ps /= self._get_length_penalty(length=di+1, alpha=1.2, min_length=5)
            # Transpose (BxKxK) => (BxK^2)
            child_ps = child_ps.view(self.batch_size, self.k * self.k)
            child_vs = child_vs.view(self.batch_size, self.k * self.k)
            # Select Top k in K^2 (shape: BxK)
            topk_child_ps, topk_child_ids = child_ps.topk(self.k)
            # Initiate topk_child_vs (shape: BxK)
            topk_child_vs = torch.LongTensor(self.batch_size, self.k)
            # Initiate parent_beams (shape: BxKxS)
            parent_beams = torch.LongTensor(self.beams.size())
            # ids // k => ids of topk_child`s parent node
            parent_beams_ids = (topk_child_ids // self.k).view(self.batch_size, self.k)

            for batch_num, batch in enumerate(topk_child_ids):
                for beam_idx, topk_child_idx in enumerate(batch):
                    topk_child_vs[batch_num, beam_idx] = child_vs[batch_num, topk_child_idx]
                    parent_beams[batch_num, beam_idx] = self.beams[batch_num, parent_beams_ids[batch_num, beam_idx]]
            # append new_topk_child (shape: BxKx(S) => BxKx(S+1))
            self.beams = torch.cat([parent_beams, topk_child_vs.view(self.batch_size, self.k, 1)], dim=2).to(self.device)
            self.probs = topk_child_ps.to(self.device)

            if torch.any(topk_child_vs == self.eos_id):
                done_ids = torch.where(topk_child_vs == self.eos_id)
                count = [1] * self.batch_size # count done beams
                for (batch_num, beam_idx) in zip(*done_ids):
                    self.sentences[batch_num].append(self.beams[batch_num, beam_idx])
                    self.sentence_probs[batch_num].append(self.probs[batch_num, beam_idx])
                    self._replace_beam(
                        child_ps=child_ps,
                        child_vs=child_vs,
                        done_ids=(batch_num, beam_idx),
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

        for batch_num, batch in enumerate(self.sentences):
            if len(batch) == 0:
                # if there is no terminated sentences, bring ongoing sentence which has the highest probability instead
                prob_batch = self.probs[batch_num].to(self.device)
                top_beam_idx = int(prob_batch.topk(1)[1])
                y_hats.append(self.beams[batch_num, top_beam_idx])
            else:
                # bring highest probability sentence
                top_beam_idx = int(torch.FloatTensor(self.sentence_probs[batch_num]).topk(1)[1])
                y_hats.append(self.sentences[batch_num][top_beam_idx])
        y_hats = self._match_len(y_hats).to(self.device)
        return y_hats

    def _match_len(self, y_hats):
        max_len = -1
        for y_hat in y_hats:
            if len(y_hat) > max_len:
                max_len = len(y_hat)

        matched = torch.LongTensor(self.batch_size, max_len).to(self.device)
        for batch_num, y_hat in enumerate(y_hats):
            matched[batch_num, :len(y_hat)] = y_hat
            matched[batch_num, len(y_hat):] = 0

        return matched

    def _is_done(self):
        """ check if all beam search process has terminated """
        for done in self.sentences:
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
        predicted_softmax = self.function(self.w(output.contiguous().view(-1, self.hidden_size)), dim=1)
        predicted_softmax = predicted_softmax.view(self.batch_size,output_size,-1)
        return predicted_softmax

    def _get_length_penalty(self, length, alpha=1.2, min_length=5):
        """
        Calculate length-penalty.
        because shorter sentence usually have bigger probability.
        using alpha = 1.2, min_length = 5 usually.
        """
        return ((min_length + length) / (min_length + 1)) ** alpha

    def _replace_beam(self, child_ps, child_vs, done_ids, count):
        """ Replaces a beam that ends with <eos> with a beam with the next higher probability. """
        done_batch_num, done_beam_idx = done_ids[0], done_ids[1]
        tmp_ids = child_ps.topk(self.k + count)[1]
        new_child_idx = tmp_ids[done_batch_num, -1]
        new_child_p = child_ps[done_batch_num, new_child_idx].to(self.device)
        new_child_v = child_vs[done_batch_num, new_child_idx].to(self.device)
        parent_beam_idx = (new_child_idx // self.k)
        parent_beam = self.beams[done_batch_num, parent_beam_idx].to(self.device)
        parent_beam = parent_beam[:-1]
        new_beam = torch.cat([parent_beam, new_child_v.view(1)])
        self.beams[done_batch_num, done_beam_idx] = new_beam
        self.probs[done_batch_num, done_beam_idx] = new_child_p