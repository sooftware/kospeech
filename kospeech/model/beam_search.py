import torch
import torch.nn as nn


def _inflate(tensor, n_repeat, dim):
    """ Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times  """
    repeat_dims = [1] * len(tensor.size())
    repeat_dims[dim] *= n_repeat

    return tensor.repeat(*repeat_dims)


class BeamSearchDecoder(nn.Module):
    """
    Top-K decoding with beam search.

    Args:
        decoder (e2e.model.speller.Speller): decoder to which beam search will be applied
        beam_size (int): size of beam

    Inputs: input_var, encoder_outputs
        - **input_var** : sequence of sos_id
        - **encoder_outputs** : tensor containing the encoded features of the input sequence

    Returns: decoder_outputs
        - **decoder_outputs** :  list of tensors containing the outputs of the decoding function.
    """

    def __init__(self, decoder, beam_size):
        super(BeamSearchDecoder, self).__init__()
        self.num_classes = decoder.num_classes
        self.max_length = decoder.max_length
        self.hidden_dim = decoder.hidden_dim
        self.forward_step = decoder.forward_step
        self.validate_args = decoder.validate_args
        self.pos_index = None
        self.beam_size = beam_size
        self.sos_id = decoder.sos_id
        self.eos_id = decoder.eos_id
        self.min_length = 5
        self.alpha = 1.2
        self.device = decoder.device

    def forward(self, input_var, encoder_outputs, teacher_forcing_ratio=0.0):
        inputs, batch_size, max_length = self.validate_args(input_var, encoder_outputs, 0.0)
        self.pos_index = (torch.LongTensor(range(batch_size)) * self.beam_size).view(-1, 1).to(self.device)

        hidden, attn = None, None
        inflated_encoder_outputs = _inflate(encoder_outputs, self.beam_size, 0)

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        # sequence_scores: tensor([[-inf], [-inf], [-inf], [-inf], [-inf], [-inf], [-inf], [-inf]])
        sequence_scores = encoder_outputs.new_zeros(batch_size * self.beam_size, 1)
        sequence_scores.fill_(-float('Inf'))

        # If beam_size is three, tensor([ 0,  3,  6,  9, 12, 15, 18, 21])
        # sequence_scores: tensor([[0.], [-inf], [-inf], [0.], [-inf], [-inf], [0.], [-inf] ..])  BKx1
        fill_index = torch.LongTensor([i * self.beam_size for i in range(batch_size)]).to(self.device)
        sequence_scores.index_fill_(0, fill_index, 0.0)

        # Initialize the input vector
        input_var = torch.transpose(torch.LongTensor([[self.sos_id] * batch_size * self.beam_size]), 0, 1)  # 1xBK => BKx1
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

    def get_length_penalty(self, length):
        """
        Calculate length-penalty.
        because shorter sentence usually have bigger probability.
        using alpha = 1.2, min_length = 5 usually.
        """
        return ((self.min_length + length) / (self.min_length + 1)) ** self.alpha
