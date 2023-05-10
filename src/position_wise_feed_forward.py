import torch


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(
        self,
        input_size=512,
        hidden_size=2048,
    ):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_size),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def initialize(self):
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
