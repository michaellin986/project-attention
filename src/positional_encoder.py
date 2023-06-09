"""
Code inspired from blog post:
https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
"""

import math
import torch


class PositionalEncoder(torch.nn.Module):
    """
    This is meant to be used to get the next positional encoding automatically.
    This can be called directly like how `nn.Modules` can.

    Example:
        encoder = PositionalEncoder()

        encoder()  # gets first positional encoding
        encoder()  # gets second positional encoding
        encoder(0) # gets first positional encoding
        encoder()  # gets third positional encoding
    """

    def encoding_element(self, pos: int, i: int):
        """
        Returns the `i`th element of the positional encoded vector at position `pos`
        """
        angle = pos / (10000 ** (2 * (i // 2) / self.d_model))

        if i % 2 == 0:
            return math.sin(angle)
        return math.cos(angle)

    def encoding(self, pos: int):
        """
        Returns the positional encoded vector at position `pos`
        """
        if pos < 0:
            raise ValueError(f"`pos` must be >= 0. Received {pos}")

        return torch.tensor(
            [self.encoding_element(pos, i) for i in range(self.d_model)]
        )

    def __init__(self, d_model=512, max_length=512):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.pe = torch.zeros(max_length, d_model).to(self.device)
        self.pe.requires_grad = False
        for pos in range(max_length):
            self.pe[pos] = self.encoding(pos)

    def forward(self, x):
        return x + self.pe[: x.size(1), :]
