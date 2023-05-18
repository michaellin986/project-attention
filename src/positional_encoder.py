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
    
    def __init__(self, d_model=16, max_len = 512):
        super().__init__()
        self.d_model = d_model  # dimension of the model
        self.maxlen = max_len
        self.pe = torch.zeros(max_len, self.d_model)
        self.pe.requires_grad = False
        for pos in range(self.maxlen):
            self.pe[pos] = self.encoding(pos)


    # def __call__(self, pos=None):
    #     """
    #     This calls self.encoding().

    #     If `pos` is not specified, then
    #         - It will use the current self.pos value
    #         - It will update self.pos to the next step

    #     If `pos` is specified, then
    #         - It will get the respective positional encoding
    #         - It will not update the self.pos
    #     """
    #     _specified = pos is not None

    #     if not _specified:
    #         pos = self.pos

    #     encoding = self.encoding(pos)

    #     if not _specified:
    #         self.pos += 1

    #     return encoding
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :]

