import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor, LongTensor
from typing import Optional, Any, Tuple
from kospeech.models.modules import Linear
from kospeech.models.modules import BaseRNN
from kospeech.models.acoustic.transformer.sublayers import AddNorm
from kospeech.models.attention import (
    LocationAwareAttention,
    MultiHeadAttention,
    AdditiveAttention,
    ScaledDotProductAttention
)


def _inflate(tensor: Tensor, n_repeat: int, dim: int) -> Tensor:
    """ Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times  """
    repeat_dims = [1] * len(tensor.size())
    repeat_dims[dim] *= n_repeat

    return tensor.repeat(*repeat_dims)


class SpeechDecoderRNN(BaseRNN):
    """
    Converts higher level features (from encoder) into output utterances
    by specifying a probability distribution over sequences of characters.

    Args:
        num_classes (int): number of classfication
        max_length (int): a maximum allowed length for the sequence to be processed
        hidden_dim (int): dimension of RNN`s hidden state vector
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        attn_mechanism (str): type of attention mechanism (default: dot)
        num_heads (int): number of attention heads. (default: 4)
        num_layers (int, optional): number of recurrent layers (default: 1)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        dropout_p (float, optional): dropout probability (default: 0.3)
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: inputs, encoder_outputs, teacher_forcing_ratio, return_decode_dict
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_dim): tensor with containing the outputs of the listener.
          Used for attention mechanism (default is `None`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).
        - **return_decode_dict** (dict): dictionary which contains decode informations.

    Returns: decoder_outputs, decode_dict
        - **decoder_outputs** (seq_len, batch, num_classes): list of tensors containing
          the outputs of the decoding function.
        - **decode_dict**: dictionary containing additional information as follows {*KEY_ATTENTION_SCORE* : list of scores
          representing encoder outputs, *KEY_SEQUENCE_SYMBOL* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """
    KEY_ATTENTION_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE_SYMBOL = 'sequence_symbol'

    def __init__(self,
                 num_classes: int,                    # number of classfication
                 max_length: int = 120,               # a maximum allowed length for the sequence to be processed
                 hidden_dim: int = 1024,              # dimension of RNN`s hidden state vector
                 sos_id: int = 1,                     # start of sentence token`s id
                 eos_id: int = 2,                     # end of sentence token`s id
                 attn_mechanism: str = 'multi-head',  # type of attention mechanism
                 num_heads: int = 4,                  # number of attention heads
                 num_layers: int = 2,                 # number of RNN layers
                 rnn_type: str = 'lstm',              # type of RNN cell
                 dropout_p: float = 0.3,              # dropout probability
                 device: str = 'cuda') -> None:       # device - 'cuda' or 'cpu'
        super(SpeechDecoderRNN, self).__init__(hidden_dim, hidden_dim, num_layers, rnn_type, dropout_p, False, device)
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.attn_mechanism = attn_mechanism.lower()
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        self.input_dropout = nn.Dropout(dropout_p)

        if self.attn_mechanism == 'loc':
            self.attention = AddNorm(LocationAwareAttention(hidden_dim, smoothing=True), hidden_dim)
        elif self.attn_mechanism == 'multi-head':
            self.attention = AddNorm(MultiHeadAttention(hidden_dim, num_heads), hidden_dim)
        elif self.attn_mechanism == 'additive':
            self.attention = AddNorm(AdditiveAttention(hidden_dim), hidden_dim)
        elif self.attn_mechanism == 'scaled-dot':
            self.attention = AddNorm(ScaledDotProductAttention(hidden_dim), hidden_dim)
        else:
            raise ValueError("Unsupported attention: %s".format(attn_mechanism))

        self.projection = AddNorm(Linear(hidden_dim, hidden_dim, bias=True), hidden_dim)
        self.generator = Linear(hidden_dim, num_classes, bias=False)

    def forward_step(self, input_var: Tensor, hidden: Optional[Any],
                     encoder_outputs: Tensor, attn: Tensor) -> Tuple[Tensor, Optional[Any], Tensor]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)

        embedded = self.embedding(input_var).to(self.device)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(embedded, hidden)

        if self.attn_mechanism == 'loc':
            context, attn = self.attention(output, encoder_outputs, attn)
        else:
            context, attn = self.attention(output, encoder_outputs, encoder_outputs)

        output = self.projection(context.view(-1, self.hidden_dim)).view(batch_size, -1, self.hidden_dim)
        output = self.generator(torch.tanh(output).contiguous().view(-1, self.hidden_dim))

        step_output = F.log_softmax(output, dim=1)
        step_output = step_output.view(batch_size, output_lengths, -1).squeeze(1)

        return step_output, hidden, attn

    def forward(self, inputs: Tensor, encoder_outputs: Tensor,
                teacher_forcing_ratio: float = 1.0, return_decode_dict: bool = False) -> Tuple[Tensor, dict]:
        hidden, attn = None, None
        result, decode_dict = list(), dict()

        if not self.training:
            decode_dict[SpeechDecoderRNN.KEY_ATTENTION_SCORE] = list()
            decode_dict[SpeechDecoderRNN.KEY_SEQUENCE_SYMBOL] = list()

        inputs, batch_size, max_length = self.validate_args(inputs, encoder_outputs, teacher_forcing_ratio)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        lengths = np.array([max_length] * batch_size)

        if use_teacher_forcing:
            inputs = inputs[inputs != self.eos_id].view(batch_size, -1)

            if self.attn_mechanism == 'loc' or self.attn_mechanism == 'additive':
                for di in range(inputs.size(1)):
                    input_var = inputs[:, di].unsqueeze(1)
                    step_output, hidden, attn = self.forward_step(input_var, hidden, encoder_outputs, attn)
                    result.append(step_output)

            else:
                step_outputs, hidden, attn = self.forward_step(inputs, hidden, encoder_outputs, attn)

                for di in range(step_outputs.size(1)):
                    step_output = step_outputs[:, di, :]
                    result.append(step_output)

        else:
            input_var = inputs[:, 0].unsqueeze(1)

            for di in range(max_length):
                step_output, hidden, attn = self.forward_step(input_var, hidden, encoder_outputs, attn)
                result.append(step_output)
                input_var = result[-1].topk(1)[1]

                if not self.training:
                    decode_dict[SpeechDecoderRNN.KEY_ATTENTION_SCORE].append(attn)
                    decode_dict[SpeechDecoderRNN.KEY_SEQUENCE_SYMBOL].append(input_var)
                    eos_batches = input_var.data.eq(self.eos_id)

                    if eos_batches.dim() > 0:
                        eos_batches = eos_batches.cpu().view(-1).numpy()
                        update_idx = ((lengths > di) & eos_batches) != 0
                        lengths[update_idx] = len(decode_dict[SpeechDecoderRNN.KEY_SEQUENCE_SYMBOL])

        if return_decode_dict:
            decode_dict[SpeechDecoderRNN.KEY_LENGTH] = lengths
            result = (result, decode_dict)
        else:
            del decode_dict

        return result

    def validate_args(self, inputs: Optional[Any] = None, encoder_outputs: Tensor = None,
                      teacher_forcing_ratio: float = 1.0) -> Tuple[Tensor, int, int]:
        """ Validate arguments """
        assert encoder_outputs is not None
        batch_size = encoder_outputs.size(0)

        if inputs is None:  # inference
            inputs = LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length

            if torch.cuda.is_available():
                inputs = inputs.cuda()

            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")

        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length


class SpeechTopKDecoder(nn.Module):
    """
    Top-K decoding with beam search.

    Args:
        decoder (Seq2seqGreedyDecoder): decoder to which beam search will be applied
        beam_size (int): size of beam

    Inputs: input_var, encoder_outputs
        - **input_var** : sequence of sos_id
        - **encoder_outputs** : tensor containing the encoded features of the input sequence

    Returns: decoder_outputs
        - **decoder_outputs** :  list of tensors containing the outputs of the decoding function.
    """

    def __init__(self, decoder: SpeechDecoderRNN, beam_size: int = 3) -> None:
        super(SpeechTopKDecoder, self).__init__()
        self.num_classes = decoder.num_classes
        self.max_length = decoder.max_length
        self.hidden_dim = decoder.hidden_dim
        self.forward_step = decoder.forward_step
        self.validate_args = decoder.validate_args
        self.decoder = decoder
        self.pos_index = None
        self.beam_size = beam_size
        self.sos_id = decoder.sos_id
        self.eos_id = decoder.eos_id
        self.min_length = 5
        self.alpha = 1.2
        self.device = decoder.device

    def forward(self, input_var=None, encoder_outputs: Tensor = None) -> list:
        inputs, batch_size, max_length = self.validate_args(input_var, encoder_outputs, 0.0)
        self.pos_index = (LongTensor(range(batch_size)) * self.beam_size).view(-1, 1).to(self.device)

        hidden, attn = None, None
        inflated_encoder_outputs = _inflate(encoder_outputs, self.beam_size, 0)

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        # sequence_scores: tensor([[-inf], [-inf], [-inf], [-inf], [-inf], [-inf], [-inf], [-inf]])
        sequence_scores = encoder_outputs.new_zeros(batch_size * self.beam_size, 1)
        sequence_scores.fill_(-float('Inf'))

        # If beam_size is three, tensor([ 0,  3,  6,  9, 12, 15, 18, 21])
        # sequence_scores: tensor([[0.], [-inf], [-inf], [0.], [-inf], [-inf], [0.], [-inf] ..])  BKx1
        fill_index = LongTensor([i * self.beam_size for i in range(batch_size)]).to(self.device)
        sequence_scores.index_fill_(0, fill_index, 0.0)

        # Initialize the input vector
        input_var = torch.transpose(LongTensor([[self.sos_id] * batch_size * self.beam_size]), 0, 1)  # 1xBK => BKx1
        input_var = input_var.to(self.device)

        # Store decisions for backtracking
        stored_outputs = list()
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()

        for di in range(max_length):
            # Run the RNN one step forward
            step_output, hidden, attn = self.forward_step(input_var, hidden, inflated_encoder_outputs, attn)
            stored_outputs.append(step_output.unsqueeze(1))

            # force the output to be longer than self.min_length
            if di < self.min_length:
                step_output[:, self.eos_id] = -float('Inf')

            sequence_scores = _inflate(sequence_scores, self.decoder.num_classes, dim=1)  # BKx1 => BKxC
            sequence_scores += step_output  # step_output: BKxC, Cumulative Score
            scores, candidates = sequence_scores.view(batch_size, -1).topk(self.beam_size, dim=1)  # BKxC => BxKC => BxK

            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            input_var = (candidates % self.decoder.num_classes).view(batch_size * self.beam_size, 1)  # BxK => BKx1
            sequence_scores = scores.view(batch_size * self.beam_size, 1)  # BKx1

            # Update fields for next timestep
            # self.pos_index.expand_as(candidates)
            predecessors = (candidates / self.decoder.num_classes + self.pos_index.expand_as(candidates))
            predecessors = predecessors.view(batch_size * self.beam_size, 1)  # BKx1

            if isinstance(hidden, tuple):
                hidden = tuple([h.index_select(1, predecessors.squeeze()) for h in hidden])
            else:
                hidden = hidden.index_select(1, predecessors.squeeze())  # hidden: LxBKxD
                                                                         # predecessors: BK

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone() / self.get_length_penalty(di + 1))
            eos_indices = input_var.data.eq(self.eos_id)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('Inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var)

        # Do backtracking to return the optimal values
        output = self._backtrack(stored_outputs, stored_predecessors, stored_emitted_symbols, stored_scores, batch_size)

        decoder_outputs = [step[:, 0, :] for step in output]
        return decoder_outputs

    def _backtrack(self, stored_outputs, stored_predecessors, stored_emitted_symbols, stored_scores, batch_size):
        """Backtracks over batch to generate optimal k-sequences.
        Args:
            stored_outputs [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
            stored_predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
            stored_emitted_symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            batch_size: Size of the batch
        Returns:
            output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
        """

        # initialize return variables given different types
        output = list()

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = stored_scores[-1].view(batch_size, self.beam_size).topk(self.beam_size)  # BxK
        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        # the number of EOS found
        # in the backward loop below for each batch
        batch_eos_found = [0] * batch_size

        t = self.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(batch_size * self.beam_size)  # BxK => BK

        while t >= 0:
            # Re-order the variables with the back pointer
            current_output = stored_outputs[t].index_select(0, t_predecessors)

            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = stored_predecessors[t].index_select(0, t_predecessors).squeeze()

            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = stored_emitted_symbols[t].data.squeeze(1).eq(self.eos_id).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0) - 1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / self.beam_size)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.beam_size - (batch_eos_found[b_idx] % self.beam_size) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.beam_size + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = stored_predecessors[t][idx[0]]
                    current_output[res_idx, :] = stored_outputs[t][idx[0], :]

            # record the back tracked results
            output.append(current_output)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(self.beam_size)
        re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(batch_size * self.beam_size)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        output = [step.index_select(0, re_sorted_idx).view(batch_size, self.beam_size, -1) for step in reversed(output)]
        return output

    def get_length_penalty(self, length: int) -> float:
        """
        Calculate length-penalty.
        because shorter sentence usually have bigger probability.
        using alpha = 1.2, min_length = 5 usually.
        """
        return ((self.min_length + length) / (self.min_length + 1)) ** self.alpha
