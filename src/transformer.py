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
        d_model=512,
        num_layers=6,
        vocab_size=10_000,
        max_length=512,
        p_dropout=0.1,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.encoder = Encoder(
            input_size=self.d_model, num_layers=num_layers, p_dropout=p_dropout
        ).to(self.device)
        self.decoder = Decoder(
            input_size=self.d_model, num_layers=num_layers, p_dropout=p_dropout
        ).to(self.device)
        self.final_ff = torch.nn.Linear(self.d_model, vocab_size)
        self.embedding = torch.nn.Embedding(vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoder(self.d_model, max_length).to(self.device)

    def forward(self, input_sentence, output_sentence):
        """
        Eventually want to change inputs to "input_sentence"
        and "output_sentence"; still need to implement
        embedding code
        """
        input_embedding = self.embedding(input_sentence) * math.sqrt(self.d_model)
        positional_input_embedding = self.positional_encoding(input_embedding)

        output_embedding = self.embedding(output_sentence)
        positional_output_embedding = self.positional_encoding(output_embedding)

        encoder_output = self.encoder(positional_input_embedding)
        decoder_output = self.decoder(positional_output_embedding, encoder_output)

        pre_softmax = self.final_ff(decoder_output)
        return torch.softmax(pre_softmax, dim=2)

    def initialize(self):
        self.encoder.initialize()
        self.decoder.initialize()
        torch.nn.init.xavier_uniform_(self.final_ff.weight)
