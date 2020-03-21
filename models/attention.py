import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, decoder_hidden_size, n_head=4):
        super(Attention, self).__init__()
        self.hidden_size = decoder_hidden_size
        self.w = nn.Linear(decoder_hidden_size*2, decoder_hidden_size)
        self.d_k = 128
        self.n_head = n_head
        self.w_q = nn.Linear(decoder_hidden_size, self.d_k*4)
        self.w_k = nn.Linear(decoder_hidden_size, self.d_k*4)

    def forward(self, decoder_output, encoder_outputs):
        bs, len_d, _ = decoder_output.size()
        bs, len_e, _ = encoder_outputs.size()
        q = decoder_output.contiguous()
        v = encoder_outputs.contiguous()
        
        decoder_output = self.w_q(decoder_output).view(bs, len_d, self.n_head, self.d_k).permute(2, 0, 1, 3)
        encoder_outputs = self.w_q(encoder_outputs).view(bs, len_e, self.n_head, self.d_k).permute(2, 0, 1, 3)
        decoder_output = decoder_output.contiguous().view(-1, len_d, self.d_k)
        encoder_outputs = encoder_outputs.contiguous().view(-1, len_e, self.d_k)
        # get attention score
        attn_score = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2))
        
        # get attention distribution
        attn_distribution = F.softmax(attn_score, dim=2)
        
        # get context vector
        context = torch.bmm(attn_distribution, encoder_outputs).view(self.n_head, bs, len_d, self.d_k).permute(1, 2, 0, 3).contiguous()
        context = context.view(bs, len_d, -1)
        
        # concatenate attn_val & decoder_output
        combined = torch.cat((context, q), dim=2)
        output = torch.tanh(self.w(combined.view(-1, 2 * self.hidden_size))).view(bs, -1, self.hidden_size)
        return output
