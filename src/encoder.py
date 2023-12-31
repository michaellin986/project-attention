import torch
from src.multihead_attention import MultiheadAttention
from src.position_wise_feed_forward import PositionWiseFeedForward


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_size=512, p_dropout=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()
        self.multihead_attention = MultiheadAttention(
            d_model=input_size,
        ).to(self.device)
        self.ff = PositionWiseFeedForward(input_size=input_size).to(self.device)
        self.layer_norm = torch.nn.LayerNorm(input_size)
        self.dropout = torch.nn.Dropout(p=p_dropout)

    def forward(self, x: torch.Tensor):
        # Sublayer 1:
        # Multihead Attention
        # Add & Norm
        q = torch.clone(x)
        k = torch.clone(x)
        v = torch.clone(x)
        sublayer_output = self.multihead_attention(q, k, v)
        norm = self.layer_norm(x + self.dropout(sublayer_output))

        # Sublayer 2:
        # Position-Wise Feed Forward
        # Add & Norm
        sublayer_output = self.ff(torch.clone(norm))
        norm = self.layer_norm(norm + self.dropout(sublayer_output))

        return norm

    def initialize(self):
        self.multihead_attention.initialize()
        self.ff.initialize()


class Encoder(torch.nn.Module):
    def __init__(self, input_size=512, num_layers=6, p_dropout=0.1):
        super().__init__()
        layers = [EncoderLayer(input_size, p_dropout) for _ in range(num_layers)]
        self.net = torch.nn.Sequential(
            torch.nn.Dropout(p_dropout),
            *layers,
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def initialize(self):
        for layer in self.net:
            if isinstance(layer, EncoderLayer):
                layer.initialize()
