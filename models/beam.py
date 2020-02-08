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
        - rnn (torch.nn.Module) :
    Inputs:

    Outputs:
    """
    def __init__(self, k, speller_hidden,
                 batch_size, max_len, decode_func, rnn,
                 embedding, input_dropout, use_attention, attention,
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
        self.done_p = []

    def search(self, init_speller_input, listener_outputs):
        """
        Notation:
            - **B**: batch_size
            - **K**: size of beam
            - **C**: number of classfication
            - **S**: sequence length
        """
        # get class classfication distribution (shape: BxC)
        init_step_output = self.forward_step(init_speller_input, listener_outputs).squeeze(1)
        # get top K probability & index (shape: BxK)
        self.cumulative_p, self.beams = init_step_output.topk(self.k)
        speller_input = self.beams
        # transpose (BxK) => (BxKx1)
        self.beams = self.beams.view(self.batch_size, self.k, 1)

        for di in range(self.max_len-1):
            if len(self.done_list) >= self.k:
                break
            # For each beam, get class classfication distribution (shape: BxKxC)
            step_output = self.forward_step(speller_input, listener_outputs).squeeze(1)
            # get top k distribution (shape: BxKxK)
            candidate_p, candidate_v = step_output.topk(self.k)
            # Transpose (BxKxK) => (BxK^2)
            candidate_p = (self.cumulative_p + candidate_p).view(self.batch_size, self.k * self.k)
            candidate_v = candidate_v.view(self.batch_size, self.k * self.k)
            # Select Top k in K^2 (shape: BxK)
            select_p, select_indice = candidate_p.topk(self.k)

            select_v = torch.LongTensor(self.batch_size, self.k)
            # Initiate Tensor (shape: BxKxS)
            prev_beams = torch.LongTensor(self.beams.size(0), self.beams.size(1), self.beams.size(2))
            # index % k => index of parent node
            parent_node = (select_indice % self.k).view(self.batch_size, self.k)

            for batch_num, beam in enumerate(select_indice):
                for beam_idx, select_idx in enumerate(beam):
                    select_v[batch_num, beam_idx] = candidate_v[batch_num, select_idx]
                    prev_beams[batch_num, beam_idx] = self.beams[batch_num, parent_node[batch_num, beam_idx]]
            # BxKx(S) => BxKx(S+1)
            self.beams = torch.cat([prev_beams, select_v.view(self.batch_size, self.k, 1)], dim=2)
            # get cumulative probability (applying length penalty)
            self.cumulative_p = select_p * self.get_length_penalty(length=select_p.size(1), alpha=1.2, min_length=5)
            # update speller_input by select_ch
            speller_input = select_v
            """ eos 처리 """
            done_coords = torch.where(select_v == self.eos_id)
            for done_coord in done_coords:
                for batch_num, beam_idx in done_coord:
                    self.done_list.append(self.beams[batch_num, beam_idx])
                    self.done_p.append(self.cumulative_p[batch_num, beam_idx])


    def forward_step(self, speller_input, listener_outputs):
        output_size = speller_input.size(1)
        embedded = self.embedding(speller_input)
        embedded = self.input_dropout(embedded)
        speller_output, hidden = self.rnn(embedded, self.speller_hidden)  # speller output

        if self.use_attention:
            output = self.attention(decoder_output=speller_output, encoder_output=listener_outputs)
        else: output = speller_output
        predicted_softmax = self.decode_func(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(self.batch_size,output_size,-1)
        return predicted_softmax

    def get_length_penalty(self, length, alpha=1.2, min_length=5):
        """
        Calculate length-penalty.
        because shorter sentence usually have bigger probability.
        Using alpha = 1.2, min_length = 5 usually.
        """
        return ((1+length) / (1+min_length)) ** alpha