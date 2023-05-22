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
        d_model=100,
        num_layers=6,
        vocab_size=10_000,
        p_dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(
            input_size=self.d_model, num_layers=num_layers, p_dropout=p_dropout
        )
        self.decoder = Decoder(
            input_size=self.d_model, num_layers=num_layers, p_dropout=p_dropout
        )
        self.final_ff = torch.nn.Linear(self.d_model, vocab_size)
        self.embedds = torch.nn.Embedding(vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoder(self.d_model)

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
