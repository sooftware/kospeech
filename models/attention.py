import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    r"""
    Applies an self attention mechanism on the output features from the decoder.
    「A Structured Self-Attentive Sentence Embedding」 Paper
    https://arxiv.org/abs/1703.03130

    .. math::
        \begin{array}{ll}
        x = contexts*output \\
        attn = exp(x_i) / sum_j exp(x_j) \\
        output = \tanh(w * (attn * contexts) + b * output)
        \end{array}

    Args:
        decoder_hidden_size (int): The number of expected features in the output

    Inputs: decoder_output, contexts
        - **decoder_output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **contexts** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Returns: output
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.

    Examples::

        >>> attention = Attention(hidden_size)
        >>> contexts = attention(decoder_output, contexts)
    """
    def __init__(self, decoder_hidden_size):
        super(Attention, self).__init__()
        self.w = nn.Linear(decoder_hidden_size*2, decoder_hidden_size)

    def forward(self, decoder_output, contexts):
        batch_size = decoder_output.size(0)
        hidden_size = decoder_output.size(2)
        input_size = contexts.size(1)

        # get attention score
        attn_score = torch.bmm(decoder_output, contexts.transpose(1, 2))
        # get attention distribution
        attn_distribution = F.softmax(attn_score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        # get attention value
        attn_val = torch.bmm(attn_distribution, contexts)
        # concatenate attn_val & decoder_output
        combined = torch.cat((attn_val, decoder_output), dim=2)
        output = F.tanh(self.w(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        return output