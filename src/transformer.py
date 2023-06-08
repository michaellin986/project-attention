import torch
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
        max_length=512,
        p_dropout=0.1,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.encoder = Encoder(
            input_size=self.d_model, num_layers=num_layers, p_dropout=p_dropout
        )
        self.decoder = Decoder(
            input_size=self.d_model, num_layers=num_layers, p_dropout=p_dropout
        )
        self.positional_encoding = PositionalEncoder(self.d_model, max_length)

    def forward(self, input_embedding, output_embedding):
        """
        Parameters
        ----------
        input_embedding : torch.Tensor
            Tensor of non-English embedding
            Tensor of size [batch_size, max_length, d_model]
        output_embedding : torch.Tensor
            Tensor of English embedding
            Tensor of size [batch_size, max_length, d_model]
        """
        positional_input_embedding = self.positional_encoding(input_embedding).to(
            self.device
        )
        positional_output_embedding = self.positional_encoding(output_embedding).to(
            self.device
        )

        encoder_output = self.encoder(positional_input_embedding).to(self.device)
        decoder_output = self.decoder(positional_output_embedding, encoder_output).to(
            self.device
        )
        return decoder_output

    def initialize(self):
        self.encoder.initialize()
        self.decoder.initialize()
