import torch
from package.definition import char2id


class Beam:
    r"""
    Applying Beam-Search during decoding process.

    Args:
        k (int) : size of beam
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

        >>> beam = Beam(k, decoder, batch_size, max_len, F.log_softmax)
        >>> y_hats = beam.search(inputs, encoder_outputs)
    """
    def __init__(self, k, decoder, batch_size, max_len, function, device):

        #assert k > 1, "beam size (k) should be bigger than 1"

        self.k = k
        self.max_len = max_len
        self.function = function
        self.n_layers = decoder.n_layers
        self.rnn = decoder.rnn
        self.embedding = decoder.embedding
        self.use_attention = decoder.use_attention
        self.attention = decoder.attention
        self.hidden_size = decoder.hidden_size
        self.out = decoder.w
        self.eos_id = decoder.eos_id
        self.beams = None
        self.cumulative_probs = None
        self.sentences = [[] for _ in range(batch_size)]
        self.sentence_probs = [[] for _ in range(batch_size)]
        self.device = device


    def search(self, input, encoder_outputs):
        """ Beam-Search Decoding (Top-K Decoding) """
        batch_size = encoder_outputs.size(0)

        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        step_outputs, hidden = self.forward_step(input, hidden, encoder_outputs)
        self.cumulative_probs, self.beams = step_outputs.topk(self.k) # BxK

        input = self.beams
        self.beams = self.beams.unsqueeze(2)

        for di in range(self.max_len-1):
            if self._is_done():
                break

            step_outputs, hidden = self.forward_step(input, hidden, encoder_outputs)
            probs, values = step_outputs.topk(self.k)

            self.cumulative_probs /= self._get_length_penalty(length=di+1, alpha=1.2, min_length=5)
            probs = self.cumulative_probs.unsqueeze(1) + probs

            probs = probs.view(batch_size, self.k * self.k)
            values = values.view(batch_size, self.k * self.k)

            topk_probs, topk_status_ids = probs.topk(self.k)
            topk_values = torch.LongTensor(batch_size, self.k)

            prev_beams = torch.LongTensor(self.beams.size())
            prev_beams_ids = (topk_status_ids // self.k).view(batch_size, self.k)

            for batch_num, batch in enumerate(topk_status_ids):
                for beam_idx, topk_status_idx in enumerate(batch):
                    topk_values[batch_num, beam_idx] = values[batch_num, topk_status_idx]
                    prev_beams[batch_num, beam_idx] = self.beams[batch_num, prev_beams_ids[batch_num, beam_idx]]

            self.beams = torch.cat([prev_beams, topk_values.unsqueeze(2)], dim=2).to(self.device)
            self.cumulative_probs = topk_probs.to(self.device)

            # if any beam encounter eos_id
            if torch.any(topk_values == self.eos_id):
                done_ids = torch.where(topk_values == self.eos_id)
                next = [1] * batch_size

                for (batch_num, beam_idx) in zip(*done_ids):
                    self.sentences[batch_num].append(self.beams[batch_num, beam_idx])
                    self.sentence_probs[batch_num].append(self.cumulative_probs[batch_num, beam_idx])
                    self._replace_beam(
                        probs = probs,
                        values = values,
                        done_ids = (batch_num, beam_idx),
                        next = next[batch_num]
                    )
                    next[batch_num] += 1

            input = topk_values

        return self._get_best()


    def forward_step(self, input, hidden, encoder_outputs):
        """ forward one step on each decoder cell """
        batch_size = encoder_outputs.size(0)
        seq_length = input.size(1)

        embedded = self.embedding(input).to(self.device)
        output, hidden = self.rnn(embedded, hidden)

        if self.use_attention:
            output = self.attention(output, encoder_outputs)

        predicted_softmax = self.function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, seq_length,-1)
        step_outputs = predicted_softmax.squeeze(1)

        return step_outputs, hidden


    def _get_best(self):
        """ get sentences which has the highest probability at each batch, stack it, and return it as 2d torch """
        y_hats = []

        for batch_num, batch in enumerate(self.sentences):
            # if there is no terminated sentences, bring ongoing sentence which has the highest probability instead
            if len(batch) == 0:
                prob_batch = self.cumulative_probs[batch_num].to(self.device)
                top_beam_idx = int(prob_batch.topk(1)[1])
                y_hats.append(self.beams[batch_num, top_beam_idx])

            # bring highest probability sentence
            else:
                top_beam_idx = int(torch.FloatTensor(self.sentence_probs[batch_num]).topk(1)[1])
                y_hats.append(self.sentences[batch_num][top_beam_idx])

        y_hats = self._match_len(y_hats).to(self.device)

        return y_hats


    def _match_len(self, y_hats):
        batch_size = y_hats.size(0)
        max_length = -1

        for y_hat in y_hats:
            if len(y_hat) > max_length:
                max_length = len(y_hat)

        matched = torch.LongTensor(batch_size, max_length).to(self.device)

        for batch_num, y_hat in enumerate(y_hats):
            matched[batch_num, :len(y_hat)] = y_hat
            matched[batch_num, len(y_hat):] = int(char2id[' '])

        return matched



    def _is_done(self):
        """ check if all beam search process has terminated """
        for done in self.sentences:
            if len(done) < self.k:
                return False

        return True


    def _get_length_penalty(self, length, alpha=1.2, min_length=5):
        """
        Calculate length-penalty.
        because shorter sentence usually have bigger probability.
        using alpha = 1.2, min_length = 5 usually.
        """
        return ((min_length + length) / (min_length + 1)) ** alpha


    def _replace_beam(self, probs, values, done_ids, next):
        """ Replaces a beam that ends with <eos> with a beam with the next higher probability. """
        done_batch_num, done_beam_idx = done_ids

        replace_ids = probs.topk(self.k + next)[1]
        replace_idx = replace_ids[done_batch_num, -1]

        new_prob = probs[done_batch_num, replace_idx].to(self.device)
        new_value = values[done_batch_num, replace_idx].to(self.device)

        prev_beam_idx = (replace_idx // self.k)
        prev_beam = self.beams[done_batch_num, prev_beam_idx]
        prev_beam = prev_beam[:-1].to(self.device)

        new_beam = torch.cat([prev_beam, new_value.view(1)])

        self.beams[done_batch_num, done_beam_idx] = new_beam
        self.cumulative_probs[done_batch_num, done_beam_idx] = new_prob