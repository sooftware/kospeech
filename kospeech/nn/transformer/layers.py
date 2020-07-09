import torch.nn as nn
from torch import Tensor
from typing import Tuple
from kospeech.nn.transformer.sublayers import MultiHeadAttention, PoswiseFeedForwardNet, AddNorm


class TransformerEncoderLayer(nn.Module):
    """
    EncoderLayer is made up of self-attention and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048,
                 dropout_p: float = 0.3, ffnet_style: str = 'ff') -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PoswiseFeedForwardNet(d_model, d_ff, dropout_p, ffnet_style), d_model)

    def forward(self, inputs: Tensor, inputs_mask: Tensor) -> Tuple[Tensor, Tensor]:
        output, attn = self.self_attention(inputs, inputs, inputs, inputs_mask)
        output = self.feed_forward(output)
        return output, attn


class TransformerDecoderLayer(nn.Module):
    """
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048,
                 dropout_p: float = 0.3, ffnet_style: str = 'ff') -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.encoder_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PoswiseFeedForwardNet(d_model, d_ff, dropout_p, ffnet_style), d_model)

    def forward(self, inputs: Tensor, memory: Tensor,
                inputs_mask: Tensor, memory_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        output, self_attn = self.self_attention(inputs, inputs, inputs, inputs_mask)
        output, encoder_attn = self.encoder_attention(output, memory, memory, memory_mask)
        output = self.feed_forward(output)
        return output, self_attn, encoder_attn
