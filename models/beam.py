"""
Copyright 2020- Kai.Lib
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import numpy as np

class Beam:
    """
    Applying Beam-Search during decoding process.

    Args:
        - k (int) : size of beam
        - speller_hidden (torch.Tensor) : hidden state of speller
        - batch_size (int) : mini-batch size during infer
        - max_len (int) :  a maximum allowed length for the sequence to be processed
        - decode_func (torch.nn.Module) : A function used to generate symbols from RNN hidden state (default : torch.nn.functional.log_softmax)
    Inputs:

    Outputs:
    """

    def __init__(self, k, speller_hidden, decoder,
                 batch_size, max_len, decode_func):
        self.k = k
        self.speller_hidden = speller_hidden
        self.batch_size = batch_size
        self.max_len = max_len
        self.decode_func = decode_func
        self.rnn = decoder.rnn
        self.embedding = decoder.embedding
        self.input_dropout = decoder.input_dropout
        self.use_attention = decoder.use_attention
        self.attention = decoder.attention
        self.hidden_size = decoder.hidden_size
        self.out = decoder.out
        self.eos_id = decoder.eos_id
        self.beam_scores = None
        self.beams = None
        self.done_beams = [[] for _ in range(self.batch_size)]
        self.done_beam_scores = [[] for _ in range(self.batch_size)]

    def search(self, init_speller_input, listener_outputs):
        """
        Beam-Search

        Comment Notation:
            - **B**: batch_size
            - **K**: size of beam
            - **C**: number of classfication
            - **S**: sequence length
        """
        # get class classfication distribution (shape: BxC)
        init_step_output = self._forward_step(init_speller_input, listener_outputs).squeeze(1)
        # get top K probability & index (shape: BxK)
        self.beam_scores, self.beams = init_step_output.topk(self.k)
        speller_input = self.beams
        # transpose (BxK) => (BxKx1)
        self.beams = self.beams.view(self.batch_size, self.k, 1)

        for di in range(self.max_len-1):
            if self._is_done():
                break
            # For each beam, get class classfication distribution (shape: BxKxC)
            step_output = self._forward_step(speller_input, listener_outputs).squeeze(1)
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
                for beam_num, topk_child_idx in enumerate(batch):
                    topk_child_vs[batch_num, beam_num] = child_vs[batch_num, topk_child_idx]
                    parent_beams[batch_num, beam_num] = self.beams[batch_num, parent_beams_indices[batch_num, beam_num]]
            # append new_topk_child (shape: BxKx(S) => BxKx(S+1))
            self.beams = torch.cat([parent_beams, topk_child_vs.view(self.batch_size, self.k, 1)], dim=2)
            self.beam_scores = topk_child_ps

            if torch.any(topk_child_vs == self.eos_id):
                done_indices = torch.where(topk_child_vs == self.eos_id)
                count = [1 * self.k]
                for done_idx in done_indices:
                    batch_num, beam_num = done_idx[0], done_idx[1]
                    self.done_beams[batch_num].append(self.beams[batch_num, beam_num])
                    self.done_beam_scores[batch_num].append(self.beam_scores[batch_num, beam_num])
                    self._replace_beam(child_ps=child_ps, child_vs=child_vs, done_beam_idx=[batch_num, beam_num], count=count[batch_num])
                    count[batch_num] += 1
            # update speller_input by select_ch
            speller_input = topk_child_vs
        y_hats = self.get_best()
        return y_hats

    def get_best(self):
        y_hats = list()
        for batch_num, batch in enumerate(self.done_beams):
            if batch == []:
                # 진행중인 놈중 가장 높은 놈 갖고와야함
                top_beam_idx = self.beam_scores[batch_num].topk(1)[1]
            else:
                top_beam_idx = self.done_beam_scores[batch_num].topk(1)[1]
            y_hats.append(*self.beams[batch_num, top_beam_idx])
        return torch.stack(y_hats, dim=0)


    def _is_done(self):
        for done in self.done_beams:
            if len(done) < self.k:
                return False
        return True

    def _forward_step(self, speller_input, listener_outputs):
        output_size = speller_input.size(1)
        embedded = self.embedding(speller_input)
        embedded = self.input_dropout(embedded)
        speller_output, hidden = self.rnn(embedded, self.speller_hidden)  # speller output

        if self.use_attention:
            output = self.attention(decoder_output=speller_output, encoder_output=listener_outputs)
        else: output = speller_output
        predicted_softmax = self.decode_func(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(self.batch_size,output_size,-1)
        return predicted_softmax

    def _get_length_penalty(self, length, alpha=1.2, min_length=5):
        """
        Calculate length-penalty.
        because shorter sentence usually have bigger probability.
        Using alpha = 1.2, min_length = 5 usually.
        """
        return ((1+length) / (1+min_length)) ** alpha

    def _replace_beam(self, child_ps, child_vs, done_beam_idx, count):
        """ Replaces a beam that ends with EOS with a beam with the next higher probability. """
        done_batch_num, done_beam_num = done_beam_idx[0], done_beam_idx[1]
        tmp_indices = child_ps.topk(self.k + count)[1]
        new_child_idx = tmp_indices[done_batch_num, -1]
        new_child_p = child_ps[done_batch_num, new_child_idx]
        new_child_v = child_vs[done_batch_num, new_child_idx]
        parent_beam_idx = (new_child_idx // self.k)
        parent_beam = self.beams[done_batch_num, parent_beam_idx]
        new_beam = torch.LongTensor(np.append(parent_beam[:-1].numpy(), new_child_v))
        self.beams[done_batch_num, done_beam_num] = new_beam
        self.beam_scores[done_batch_num, done_beam_num] = new_child_p