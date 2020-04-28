import torch
import torch.nn.functional as F
import copy
from package.definition import char2id, id2char
from package.utils import label_to_string


class Beam:
    r"""
    Applying Beam-Search during decoding process.
    Args:
        k (int) : size of beam
        decoder (torch.nn.Module) : get pointer of decoder object to get multiple parameters at once
        batch_size (int) : mini-batch size during infer
        max_length (int) :  a maximum allowed length for the sequence to be processed
    Inputs: decoder_input, encoder_outputs
        - **decoder_input** (torch.Tensor): initial input of decoder - <sos>
        - **encoder_outputs** (torch.Tensor): tensor with containing the outputs of the encoder.
    Returns: y_hats
        - **y_hats** (batch, seq_len): predicted y values (y_hat) by the model
    Examples::
        >>> beam = Beam(k, decoder, batch_size, max_length, F.log_softmax)
        >>> y_hats = beam.search(inputs, encoder_outputs)
    """

    def __init__(self, k, decoder, batch_size, max_length, device):

        # assert k > 1, "beam size (k) should be bigger than 1"

        self.max_length = max_length
        self.n_layers = decoder.n_layers
        self.rnn = decoder.rnn
        self.embedding = decoder.embedding
        self.attention = decoder.attention
        self.hidden_dim = decoder.hidden_dim
        self.decoder = decoder
        self.fc = decoder.fc
        self.eos_id = decoder.eos_id
        self.beams = None
        self.cumulative_probs = None
        self.sentences = [[] for _ in range(batch_size)]
        self.sentence_probs = [[] for _ in range(batch_size)]
        self.device = device
        self.k = k

    def search(self, input_, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        h_state = self.decoder.init_state(batch_size)

        # input_ : Bx1 (sos_id)   encoder_outputs : BxSxH
        step_outputs, h_state = self.forward_step(input_, h_state, encoder_outputs)
        # step_outputs : BxC   h_state : KxBxH
        self.cumulative_probs, self.beams = step_outputs.topk(self.k)  # BxK
        # self.cumulative_probs : BxK  확률   self.beam : BxK  인덱스

        input_ = copy.deepcopy(self.beams)
        self.beams = self.beams.unsqueeze(2)  # BxK => BxKx1

        repeated_h_state = h_state.repeat(1, 1, self.k)
        repeated_encoder_outputs = encoder_outputs.repeat(1, 1, self.k)
        # input_ : BxK   self.beams : BxKx1

        for di in range(self.max_length - 1):
            if self._is_done():
                break

            step_outputs, repeated_h_state = self.forward_beam_step(input_, repeated_h_state, repeated_encoder_outputs)

            # step_outputs: BxKxC     probs: BxKxK     values: BxKxK
            probs, values = step_outputs.topk(self.k)
            # probs = probs.unsqueeze(1)  # k = 1
            probs = (probs.permute(0, 2, 1) + self.cumulative_probs.unsqueeze(1)).permute(0, 2, 1)

            probs = probs.view(batch_size, self.k * self.k)
            values = values.view(batch_size, self.k * self.k)

            topk_probs, topk_status_ids = probs.topk(self.k)  # BxK^2 = > BxK
            prev_beams_ids = (topk_status_ids // self.k)

            topk_values = torch.zeros((batch_size, self.k), dtype=torch.long)
            prev_beams = torch.zeros(self.beams.size(), dtype=torch.long)

            for batch_num, batch in enumerate(topk_status_ids):
                for beam_idx, topk_status_idx in enumerate(batch):
                    topk_values[batch_num, beam_idx] = values[batch_num, topk_status_idx]
                    prev_beams[batch_num, beam_idx] = copy.deepcopy(self.beams[batch_num, prev_beams_ids[batch_num, beam_idx]])

            self.beams = torch.cat([prev_beams, topk_values.unsqueeze(2)], dim=2).to(self.device)
            self.cumulative_probs = topk_probs.to(self.device)

            # if any beam encounter eos_id
            if torch.any(topk_values == self.eos_id):
                done_ids = torch.where(topk_values == self.eos_id)
                next_ = [1] * batch_size

                for (batch_num, beam_idx) in zip(*done_ids):
                    self.sentences[batch_num].append(copy.deepcopy(self.beams[batch_num, beam_idx]))
                    self.sentence_probs[batch_num].append(copy.deepcopy(self.cumulative_probs[batch_num, beam_idx]))
                    eos_count = self._replace_beam(
                        probs=probs,
                        values=values,
                        done_ids=(batch_num, beam_idx),
                        next_=next_[batch_num],
                        eos_count=1
                    )

                    next_[batch_num] += eos_count

            input_ = copy.deepcopy(self.beams[:, :, -1])

        return self.get_best()

    def forward_step(self, input_, h_state, encoder_outputs):
        """
        :param input_: (batch_size, beam_size)
        :param h_state: (beam_size, batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, seq_len, hidden_dim)
        :return: step_outputs: (batch_size, beam_size, class_num)
        """

        batch_size = encoder_outputs.size(0)
        seq_length = input_.size(1)

        embedded = self.embedding(input_).to(self.device)

        output, h_state = self.rnn(embedded, h_state)
        context = self.attention(output, encoder_outputs)

        predicted_softmax = F.log_softmax(self.fc(context.contiguous().view(-1, self.hidden_dim)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, seq_length, -1)
        step_outputs = predicted_softmax.squeeze(1)

        return step_outputs, h_state

    def forward_beam_step(self, input_, h_state, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        seq_length = input_.size(1)

        embedded = self.embedding(input_).to(self.device)

        output, h_state = self.rnn(embedded, h_state)
        context = self.attention(output, encoder_outputs)

        predicted_softmax = F.log_softmax(self.fc(context.contiguous().view(-1, self.hidden_dim)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, seq_length, -1)
        step_outputs = predicted_softmax.squeeze(1)

        return step_outputs, h_state

    def get_best(self):
        y_hats = list()

        for batch_num, batch in enumerate(self.sentences):
            for beam_idx, beam in enumerate(batch):
                self.sentence_probs[batch_num][beam_idx] /= self.get_length_penalty(len(beam))

        for batch_num, batch in enumerate(self.sentences):
            # if there is no terminated sentences, bring ongoing sentence which has the highest probability instead
            if len(batch) == 0:
                prob_batch = self.cumulative_probs[batch_num].to(self.device)
                top_beam_idx = int(prob_batch.topk(1)[1])
                y_hats.append(copy.deepcopy(self.beams[batch_num, top_beam_idx]))

            # bring highest probability sentence
            else:
                top_beam_idx = int(torch.FloatTensor(self.sentence_probs[batch_num]).topk(1)[1])
                y_hats.append(copy.deepcopy(self.sentences[batch_num][top_beam_idx]))

        y_hats = self._match_len(y_hats).to(self.device)

        return y_hats

    def _match_len(self, y_hats):
        batch_size = len(y_hats)
        max_length = -1

        for y_hat in y_hats:
            if len(y_hat) > max_length:
                max_length = len(y_hat)

        matched = torch.zeros((batch_size, max_length), dtype=torch.long).to(self.device)

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

    def get_length_penalty(self, length, alpha=1.2, min_length=5):
        """
        Calculate length-penalty.
        because shorter sentence usually have bigger probability.
        using alpha = 1.2, min_length = 5 usually.
        """
        return ((min_length + length) / (min_length + 1)) ** alpha

    def _replace_beam(self, probs, values, done_ids, next_, eos_count):
        """ Replaces a beam that ends with <eos> with a beam with the next higher probability.

        probs BxK^2
        """
        done_batch_num, done_beam_idx = done_ids

        replace_ids = probs.topk(self.k + next_)[1]
        replace_idx = replace_ids[done_batch_num, -1]

        new_prob = probs[done_batch_num, replace_idx].to(self.device)
        new_value = values[done_batch_num, replace_idx].to(self.device)

        prev_beam_idx = (replace_idx // self.k)
        prev_beam = copy.deepcopy(self.beams[done_batch_num, prev_beam_idx])
        prev_beam = prev_beam[:-1].to(self.device)

        new_beam = torch.cat([prev_beam, new_value.view(1)])

        if int(new_value) == self.eos_id:
            self.sentences[done_batch_num].append(copy.deepcopy(new_beam))
            self.sentence_probs[done_batch_num].append(copy.deepcopy(new_prob))
            eos_count = self._replace_beam(probs, values, done_ids, next_ + eos_count, eos_count + 1)

        else:
            self.beams[done_batch_num, done_beam_idx] = copy.deepcopy(new_beam)
            self.cumulative_probs[done_batch_num, done_beam_idx] = copy.deepcopy(new_prob)

        return eos_count
