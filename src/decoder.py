import torch
from src.multihead_attention import MultiheadAttention
from src.position_wise_feed_forward import PositionWiseFeedForward


class DecoderLayer(torch.nn.Module):
    def __init__(self, input_size=512, p_dropout=0.1):
        super().__init__()
        self.masked_multihead_attention = MultiheadAttention(
            d_model=input_size, masked=True
        )
        self.multihead_attention = MultiheadAttention(
            d_model=input_size,
        )
        self.ff = PositionWiseFeedForward(input_size=input_size)
        self.layer_norm = torch.nn.LayerNorm(input_size)
        self.dropout = torch.nn.Dropout(p=p_dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor):
        # Sublayer 1:
        # Masked Multihead Attention
        # Add & Norm
        q = torch.clone(x)
        k = torch.clone(x)
        v = torch.clone(x)
        sublayer_output = self.masked_multihead_attention(q, k, v)
        norm = self.layer_norm(x + self.dropout(sublayer_output))

        # Sublayer 2:
        # Multihead Attention
        # Add & Norm
        q = torch.clone(norm)
        k = torch.clone(encoder_output)
        v = torch.clone(encoder_output)
        sublayer_output = self.multihead_attention(q, k, v)
        norm = self.layer_norm(norm + self.dropout(sublayer_output))

        # Sublayer 3:
        # Position-Wise Feed Forward
        # Add & Norm
        sublayer_output = self.ff(torch.clone(norm))
        norm = self.layer_norm(norm + self.dropout(sublayer_output))

        return norm

    def initialize(self):
        self.masked_multihead_attention.initialize()
        self.multihead_attention.initialize()
        self.ff.initialize()


class Decoder(torch.nn.Module):
    def __init__(self, input_size=512, num_layers=6, p_dropout=0.1):
        super().__init__()
        self.layers = []
        self.dropout = torch.nn.Dropout(p=p_dropout)
        for i in range(num_layers):
            layer = DecoderLayer(input_size, p_dropout)
            self.layers.append(layer)

            # Register this layer so Pytorch tracks its parameters
            setattr(self, f"layer{i}", layer)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor):
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x

    def initialize(self):
        for layer in self.layers:
            if isinstance(layer, DecoderLayer):
                layer.initialize()
