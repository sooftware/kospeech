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
    def __init__(self, k, speller_hidden, batch_size,
                 max_len, decode_func, rnn, embedding,
                 input_dropout, use_attention, attention,
                 hidden_size, out, eos_id):
        self.k = k
        self.speller_hidden = speller_hidden
        self.batch_size = batch_size
        self.max_len = max_len
        self.decode_func = decode_func
        self.rnn = rnn
        self.embedding = embedding
        self.input_dropout = input_dropout
        self.use_attention = use_attention
        self.attention = attention
        self.hidden_size = hidden_size
        self.out = out
        self.eos_id = eos_id
        self.cumulative_p = None
        self.beams = None
        self.done_list = []
        for _ in range(self.batch_size):
            self.done_list.append([])
        self.done_p = []
        for _ in range(self.batch_size):
            self.done_p.append([])

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
        self.cumulative_p, self.beams = init_step_output.topk(self.k)
        speller_input = self.beams
        # transpose (BxK) => (BxKx1)
        self.beams = self.beams.view(self.batch_size, self.k, 1)

        for di in range(self.max_len-1):
            if self._is_done():
                break
            # For each beam, get class classfication distribution (shape: BxKxC)
            step_output = self._forward_step(speller_input, listener_outputs).squeeze(1)
            # get top k distribution (shape: BxKxK)
            child_p, child_v = step_output.topk(self.k)
            # get child probability (applying length penalty)
            child_p = (self.cumulative_p.view(self.batch_size, 1, self.k) + child_p) * self._get_length_penalty(length=di+1, alpha=1.2, min_length=5)
            # Transpose (BxKxK) => (BxK^2)
            child_p = child_p.view(self.batch_size, self.k * self.k)
            child_v = child_v.view(self.batch_size, self.k * self.k)
            # Select Top k in K^2 (shape: BxK)
            topk_child_p, topk_child_indices = child_p.topk(self.k)
            topk_child_v = torch.LongTensor(self.batch_size, self.k)
            # Initiate Tensor (shape: BxKxS)
            parent_beams = torch.LongTensor(self.beams.size(0), self.beams.size(1), self.beams.size(2))
            # index % k => index of parent node
            parent_beams_indices = (topk_child_indices % self.k).view(self.batch_size, self.k)

            for batch_num, batch in enumerate(topk_child_indices):
                for beam_num, topk_child_idx in enumerate(batch):
                    topk_child_v[batch_num, beam_num] = child_v[batch_num, topk_child_idx]
                    parent_beams[batch_num, beam_num] = self.beams[batch_num, parent_beams_indices[batch_num, beam_num]]
            # BxKx(S) => BxKx(S+1)
            self.beams = torch.cat([parent_beams, topk_child_v.view(self.batch_size, self.k, 1)], dim=2)

            """ 여기 확인해봐야함 """
            if torch.any(topk_child_v == self.eos_id):
                eos_coords = torch.where(topk_child_v == self.eos_id)
                for sub_num, eos_coord in enumerate(eos_coords):
                    for batch_num, beam_idx in eos_coord:
                        self.done_list.append(self.beams[batch_num, beam_idx])
                        self.done_p.append(self.cumulative_p[batch_num, beam_idx])
                        self._replace_beam(child_p, child_v, batch_num, sub_num, di)
            """ ================ """
            # update speller_input by select_ch
            speller_input = topk_child_v


    def _is_done(self):
        for done in self.done_list:
            if len(done) < self.k:
                return False
        return True

    def _replace_beam(self, candidate_p, child_v, batch_num, beam_idx, sub_num, step):
        # Bx(K+1)
        sub_p, sub_indice = candidate_p.topk(self.k + sub_num + 1)
        # Bx1
        sub_p = sub_p[:, (self.k + sub_num):]
        sub_indice = sub_indice[:, (self.k + sub_num):]
        # Bx1
        parent_beams_indices = (sub_indice % self.k).view(self.batch_size, 1)
        prev_beam = self.beams[batch_num, parent_beams_indices[batch_num, 0]]
        prev_beam_p = self.cumulative_p[batch_num, parent_beams_indices[batch_num, 0]]
        sub_v = child_v[batch_num, sub_indice[batch_num, 0]]
        new_beam = torch.cat([prev_beam, sub_v])
        self.beams[batch_num, beam_idx] = new_beam
        self.cumulative_p[batch_num, beam_idx] = (prev_beam_p + sub_p) *  self._get_length_penalty(length=step+1, alpha=1.2, min_length=5)

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