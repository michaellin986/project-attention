import torch
from src.encoder import Encoder
from src.decoder import Decoder


class Transformer(torch.nn.Module):
    """
    Putting everything together
    """

    def __init__(
        self,
        input_size=512,
        num_layers=6,
        vocab_size=10_000,
        p_dropout=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            input_size=input_size, num_layers=num_layers, p_dropout=p_dropout
        )
        self.decoder = Decoder(
            input_size=input_size, num_layers=num_layers, p_dropout=p_dropout
        )
        self.final_ff = torch.nn.Linear(input_size, vocab_size)
        # TODO:
        # Add embedding logic

    def forward(self, input_embedding, output_embedding):
        """
        Eventually want to change inputs to "input_string"
        and "output_string"; still need to implement
        embedding code
        """
        # TODO:
        # Add embedding + positional encoding logic
        encoder_output = self.encoder(input_embedding)
        decoder_output = self.decoder(output_embedding, encoder_output)
        pre_softmax = self.final_ff(decoder_output)
        return torch.softmax(pre_softmax, dim=1)

    def initialize(self):
        self.encoder.initialize()
        self.decoder.initialize()
        torch.nn.init.xavier_uniform_(self.final_ff.weight)
