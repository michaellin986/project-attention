import torch
import math
from src.encoder import Encoder
from src.decoder import Decoder
from src.positional_encoder import PositionalEncoder


class Transformer(torch.nn.Module):
    """
    Putting everything together
    """

    def __init__(
        self,
        input_size=100,
        num_layers=6,
        vocab_size=10_000,
        embed_size=100,
        p_dropout=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            input_size=embed_size, num_layers=num_layers, p_dropout=p_dropout
        )
        self.decoder = Decoder(
            input_size=embed_size, num_layers=num_layers, p_dropout=p_dropout
        )
        self.final_ff = torch.nn.Linear(embed_size, vocab_size)
        # TODO:
        # Add embedding logic
        self.embedds = torch.nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoder(embed_size)
        self.d_model = input_size

    def forward(self, input_string, output_string):
        """
        Eventually want to change inputs to "input_string"
        and "output_string"; still need to implement
        embedding code
        """
        input_embedding = self.embedds(input_string) * math.sqrt(self.d_model)
        input_embedding = self.positional_encoding(input_embedding)
        output_embedding = self.embedds(output_string)
        output_embedding = self.positional_encoding(output_embedding)
        encoder_output = self.encoder(input_embedding)
        decoder_output = self.decoder(output_embedding, encoder_output)
        pre_softmax = self.final_ff(decoder_output)
        return torch.softmax(pre_softmax, dim=2)

    def initialize(self):
        self.encoder.initialize()
        self.decoder.initialize()
        torch.nn.init.xavier_uniform_(self.final_ff.weight)
