import torch
from src.data import RandomTokenDataset
from src.trainer_testing_transformer import TransformerTrainer

from src.transformer import Transformer


def test_transformer():
    max_length = 16
    dataset = RandomTokenDataset(num_examples=30, max_length=max_length)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    model = Transformer(
        d_model=512,
        num_layers=6,
        vocab_size=100,
        max_length=max_length,
        p_dropout=0.1,
    )
    model.initialize()

    trainer = TransformerTrainer(
        optimizer=torch.optim.Adam,
        model=model,
        loss_func=torch.nn.CrossEntropyLoss(),
        lr=0.002,
    )

    trainer.train(data_loader, 20)
