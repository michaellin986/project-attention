import torch
import numpy as np


class MultiheadAttention(torch.nn.Module):
    """
    -------
    Encoder self-attention
    Q, K, V: output from previous encoder layer

    Decoder self-attention
    Q, K, V: output from previous decoder layer

    Encoder-decoder self attention
    Q: output from previous decoder layer
    K, V: output of encoder
    -------

    -------
    Dimensions
    dim(input embedding) = d_model = 512
    dim(K) = dim(Q) = d_k = 64
    dim(V) = d_v = 64
    Each linear layer for Q, K, V has [input x output] dimension:
        [d_model x d_k] or [d_model x d_v]
    The final linear layer has [input x output] dimension:
        [h*d_v x d_model]
    -------

    """

    def __init__(
        self,
        h=8,
        d_model=512,
        d_k=64,
        d_v=64,
        masked=False,
    ):
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.masked = masked
        self.linear_q_layers = [
            torch.nn.Linear(self.d_model, self.d_k) for _ in range(h)
        ]
        self.linear_k_layers = [
            torch.nn.Linear(self.d_model, self.d_k) for _ in range(h)
        ]
        self.linear_v_layers = [
            torch.nn.Linear(self.d_model, self.d_v) for _ in range(h)
        ]
        self.linear_output_layer = torch.nn.Linear(self.h * self.d_v, self.d_model)

    def forward(self, q, k, v):
        attentions = []
        for i in range(self.h):
            # 1. Linear layers
            curr_q = self.linear_q_layers[i](torch.clone(q))
            curr_k = self.linear_k_layers[i](torch.clone(k))
            curr_v = self.linear_v_layers[i](torch.clone(v))

            # 2. Scaled dot product attention
            attention = self.scaled_dot_product_attention(curr_q, curr_k, curr_v)
            attentions.append(attention)

        # 3. Concatenate
        concatenated = torch.cat(attentions, dim=2)

        # 4. Output linear layer
        output = self.linear_output_layer(concatenated)
        return output

    def scaled_dot_product_attention(self, q, k, v):

        scaled = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k**0.5)

        if self.masked:
            tri_len = scaled.size(1)
            mask = torch.tensor(np.triu(np.ones(tri_len), k=1))
            scaled = scaled.masked_fill(mask == 0, -1e20)

        softmaxed = torch.softmax(scaled, dim=1)
        return torch.matmul(softmaxed, v)

    def initialize(self):
        for layers in [
            self.linear_q_layers,
            self.linear_k_layers,
            self.linear_v_layers,
        ]:
            for layer in layers:
                torch.nn.init.xavier_uniform_(layer.weight)

        torch.nn.init.xavier_uniform_(self.linear_output_layer.weight)
