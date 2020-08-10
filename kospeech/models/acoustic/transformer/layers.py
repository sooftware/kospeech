import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Any
from kospeech.models.attention import MultiHeadAttention
from kospeech.models.acoustic.transformer.sublayers import (
    PositionWiseFeedForwardNet,
    AddNorm
)


class SpeechTransformerEncoderLayer(nn.Module):
    """
    EncoderLayer is made up of self-attention and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048,
                 dropout_p: float = 0.3, ffnet_style: str = 'ff') -> None:
        super(SpeechTransformerEncoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff, dropout_p, ffnet_style), d_model)

    def forward(self, inputs: Tensor, non_pad_mask: Optional[Any] = None,
                self_attn_mask: Optional[Any] = None) -> Tuple[Tensor, Tensor]:
        output, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)

        if non_pad_mask is not None:
            output *= non_pad_mask

        output = self.feed_forward(output)

        if non_pad_mask is not None:
            output *= non_pad_mask

        return output, attn


class SpeechTransformerDecoderLayer(nn.Module):
    """
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    """
    def __init__(self,
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 dropout_p: float = 0.3,
                 ffnet_style: str = 'ff') -> None:
        super(SpeechTransformerDecoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.memory_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff, dropout_p, ffnet_style), d_model)

    def forward(self, inputs: Tensor, memory: Tensor,
                non_pad_mask: Optional[Any] = None, self_attn_mask: Optional[Any] = None,
                memory_mask: Optional[Any] = None) -> Tuple[Tensor, Tensor, Tensor]:
        output, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)

        if non_pad_mask is not None:
            output *= non_pad_mask

        output, memory_attn = self.memory_attention(output, memory, memory, memory_mask)

        if non_pad_mask is not None:
            output *= non_pad_mask

        output = self.feed_forward(output)

        if non_pad_mask is not None:
            output *= non_pad_mask

        return output, self_attn, memory_attn
