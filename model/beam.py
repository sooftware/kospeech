import torch


class Beam(object):
    """Class for managing the internals of the beam search process.
    Takes care of beams, back pointers, and scores.

    Args:
        k (int): Number of beams to use.
        bos_id (int): Magic integer in output vocab.
        eos_id (int): Magic integer in output vocab.
        min_length (int): Shortest acceptable generation, not counting begin-of-sentence or end-of-sentence.
    """

    def __init__(self, k, bos_id, eos_id, min_length=3):

        self.k = k
        self.scores = torch.FloatTensor(k).zero_()
        self.next_ys = [torch.LongTensor(k).fill_(bos_id)]
        self.eos_id = eos_id
        self.all_scores = list()
        self.all_probs = list()
        self.prev_ks = list()
        self.finished = list()
        self.min_length = min_length

    @property
    def current_predictions(self):
        return self.next_ys[-1]

    @property
    def current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    def advance(self, predicted_softmax):
        predicted_softmax_clone = predicted_softmax.clone()
        num_words = predicted_softmax.size(1)

        # force the output to be longer than self.min_length
        current_length = len(self.next_ys)
        if current_length <= self.min_length:
            # assumes there are len(predicted_softmax_clone) predictions OTHER
            # than EOS that are greater than -1e20
            for idx in range(len(predicted_softmax)):
                predicted_softmax[idx][self.eos_id] = -1e10

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = predicted_softmax + self.scores.unsqueeze(1)  # beam_width * predicted_softmax
            # Don't let EOS have children.
            for i in range(self.k):
                if self.next_ys[-1][i] == self.eos_id:
                    beam_scores[i] = -1e10

        else:
            beam_scores = predicted_softmax[0]

        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.k, 0, True, True)
        self.all_probs.append(predicted_softmax_clone)
        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))

        for i in range(self.k):
            if self.next_ys[-1][i] == self.eos_id:
                length = len(self.next_ys) - 1
                # score = self.scores[i] / length
                score = self.scores[i] / self.get_length_penalty(length)
                self.finished.append((score, length, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self.eos_id:
            self.all_scores.append(self.scores)

    def sort_finished(self):
        if len(self.finished) == 0:
            for i in range(self.k):
                length = len(self.next_ys) - 1
                score = self.scores[i] / self.get_length_penalty(length)
                self.finished.append((score, length, i))

        self.finished = sorted(self.finished, key=lambda obj: obj[0], reverse=True)

        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]

        return scores, ks

    def get_hypoyhesis(self, timestep, k):
        """Walk back to construct the full hypothesis."""
        hypothesis = list()
        key_index = list()
        log_prob = list()

        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hypothesis.append(self.next_ys[j + 1][k])
            key_index.append(k)
            log_prob.append(self.all_probs[j][k])
            k = self.prev_ks[j][k]

        return hypothesis[::-1], key_index[::-1], log_prob[::-1]

    def fill_empty_sequence(self, stack, max_length):
        for i in range(stack.size(0), max_length):
            stack = torch.cat([stack, stack[0].unsqueeze(0)])
        return stack

    def get_length_penalty(self, length, alpha=1.2, min_length=5):
        """
        Calculate length-penalty.
        because shorter sentence usually have bigger probability.
        using alpha = 1.2, min_length = 5 usually.
        """
        return ((min_length + length) / (min_length + 1)) ** alpha
