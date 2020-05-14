import torch
import torch.nn as nn
from torch.autograd import Variable


def _inflate(tensor, n_repeat, dim):
    """ Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times  """
    repeat_dims = [1] * len(tensor.size())
    repeat_dims[dim] *= n_repeat

    return tensor.repeat(*repeat_dims)


class TopKDecoder(nn.Module):
    r"""Top-K decoding with beam search.

    Args:
        decoder (nn.Module): decoder to which beam search will be applied
        k (int): size of beam

    Inputs: input_var, encoder_outputs
        - **input_var** : sequence of sos_id
        - **encoder_outputs** : tensor containing the encoded features of the input sequence

    Returns: hypothesis
        - **hypothesis** : predicted y values (y_hat) by the model
    """
    def __init__(self, decoder, k):
        super(TopKDecoder, self).__init__()
        self.sos_id = decoder.sos_id
        self.eos_id = decoder.eos_id
        self.k = k
        self.num_class = decoder.num_class
        self.hidden_dim = decoder.hidden_dim
        self.forward_step = decoder.forward_step

    def forward(self, input_var, encoder_outputs, teacher_forcing_ratio=0.0):
        inputs, batch_size, max_length = self.validate_args(input_var, encoder_outputs, 0.0)
        self.pos_index = Variable(torch.LongTensor(range(batch_size)) * self.k).view(-1, 1)

        hidden = None
        inflated_encoder_outputs = _inflate(encoder_outputs, self.k, 1)

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = torch.Tensor(batch_size * self.k, 1)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.LongTensor([i * self.k for i in range(0, batch_size)]), 0.0)
        sequence_scores = Variable(sequence_scores)

        # Initialize the input vector
        input_var = Variable(torch.transpose(torch.LongTensor([[self.sos_id] * batch_size * self.k]), 0, 1))

        # Store decisions for backtracking
        stored_outputs = list()
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        stored_hidden = list()

        for _ in range(max_length):
            # Run the RNN one step forward
            predicted_softmax, hidden = self.forward_step(input_var, hidden, inflated_encoder_outputs)

            sequence_scores = _inflate(sequence_scores, self.num_class, 1)
            sequence_scores += predicted_softmax.squeeze(1)
            scores, candidates = sequence_scores.view(batch_size, -1).topk(self.k, dim=1)

            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            input_var = (candidates % self.V).view(batch_size * self.k, 1)
            sequence_scores = scores.view(batch_size * self.k, 1)

            # Update fields for next timestep
            predecessors = (candidates / self.num_class + self.pos_index.expand_as(candidates))
            predecessors = predecessors.view(batch_size * self.k, 1)

            if isinstance(hidden, tuple):
                hidden = tuple([h.index_select(1, predecessors.squeeze()) for h in hidden])
            else:
                hidden = hidden.index_select(1, predecessors.squeeze())

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = input_var.data.eq(self.eos_id)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var)
            stored_hidden.append(hidden)

        # Do backtracking to return the optimal values
        output, h_t, h_n, s, l, p = self._backtrack(stored_outputs, stored_hidden,
                                                    stored_predecessors, stored_emitted_symbols,
                                                    stored_scores, batch_size, self.hidden_dim)

        decoder_outputs = [step[:, 0, :] for step in output]

        return decoder_outputs

    def _backtrack(self, nw_output, nw_hidden, predecessors, symbols, scores, batch_size, hidden_dim):
        """Backtracks over batch to generate optimal k-sequences.
        Args:
            nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
            nw_hidden [(num_layers, batch*k, hidden_dim)] * sequence_length: A Tensor of hidden states from network
            predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
            symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            batch_size: Size of the batch
            hidden_dim: Size of the hidden state

        Returns:
            output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_t [(batch, k, hidden_dim)] * sequence_length: A list containing the output features (h_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_n(batch, k, hidden_dim): A Tensor containing the last hidden state for all top-k sequences.
            score [batch, k]: A list containing the final scores for all top-k sequences
            length [batch, k]: A list specifying the length of each sequence in the top-k candidates
            p (batch, k, sequence_len): A Tensor containing predicted sequence
        """

        lstm = isinstance(nw_hidden[0], tuple)

        # initialize return variables given different types
        output = list()
        h_t = list()
        p = list()
        # Placeholder for last hidden state of top-k sequences.
        # If a (top-k) sequence ends early in decoding, `h_n` contains
        # its hidden state when it sees EOS.  Otherwise, `h_n` contains
        # the last hidden state of decoding.
        if lstm:
            state_size = nw_hidden[0][0].size()
            h_n = tuple([torch.zeros(state_size), torch.zeros(state_size)])
        else:
            h_n = torch.zeros(nw_hidden[0].size())
        l = [[self.rnn.max_length] * self.k for _ in range(batch_size)]  # Placeholder for lengths of top-k sequences
                                                                         # Similar to `h_n`

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(batch_size, self.k).topk(self.k)
        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * batch_size   # the number of EOS found
                                             # in the backward loop below for each batch

        t = self.rnn.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(batch_size * self.k)

        while t >= 0:
            # Re-order the variables with the back pointer
            current_output = nw_output[t].index_select(0, t_predecessors)
            if lstm:
                current_hidden = tuple([h.index_select(1, t_predecessors) for h in nw_hidden[t]])
            else:
                current_hidden = nw_hidden[t].index_select(1, t_predecessors)
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()

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
            eos_indices = symbols[t].data.squeeze(1).eq(self.eos_id).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / self.k)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.k - (batch_eos_found[b_idx] % self.k) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.k + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_output[res_idx, :] = nw_output[t][idx[0], :]
                    if lstm:
                        current_hidden[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :]
                        current_hidden[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :]
                        h_n[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :].data
                        h_n[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :].data
                    else:
                        current_hidden[:, res_idx, :] = nw_hidden[t][:, idx[0], :]
                        h_n[:, res_idx, :] = nw_hidden[t][:, idx[0], :].data
                    current_symbol[res_idx, :] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            output.append(current_output)
            h_t.append(current_hidden)
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(self.k)
        for b_idx in range(batch_size):
            l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx, :]]

        re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(batch_size * self.k)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        output = [step.index_select(0, re_sorted_idx).view(batch_size, self.k, -1) for step in reversed(output)]
        p = [step.index_select(0, re_sorted_idx).view(batch_size, self.k, -1) for step in reversed(p)]
        if lstm:
            h_t = [tuple([h.index_select(1, re_sorted_idx).view(-1, batch_size, self.k, hidden_dim) for h in step]) for step in reversed(h_t)]
            h_n = tuple([h.index_select(1, re_sorted_idx.data).view(-1, batch_size, self.k, hidden_dim) for h in h_n])
        else:
            h_t = [step.index_select(1, re_sorted_idx).view(-1, batch_size, self.k, hidden_dim) for step in reversed(h_t)]
            h_n = h_n.index_select(1, re_sorted_idx.data).view(-1, batch_size, self.k, hidden_dim)
        s = s.data

        return output, h_t, h_n, s, l, p
