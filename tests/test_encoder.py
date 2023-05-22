import torch
from src.data import RandomEmbeddingDataset
from src.trainer import Trainer

from src.encoder import EncoderLayer, Encoder


def test_encoder_layer():
    dataset = RandomEmbeddingDataset(num_examples=1000, d_model=512, num_inputs=1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    model = EncoderLayer(input_size=512, p_dropout=0.1)
    model.initialize()

    trainer = Trainer(
        optimizer=torch.optim.Adam,
        model=model,
        loss_func=torch.nn.CrossEntropyLoss(),
        lr=0.002,
    )

    trainer.train(data_loader, 20)


def test_encoder():
    dataset = RandomEmbeddingDataset(num_examples=1000, d_model=512, num_inputs=1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    model = Encoder(input_size=512, num_layers=6, p_dropout=0.1)
    model.initialize()

    trainer = Trainer(
        optimizer=torch.optim.Adam,
        model=model,
        loss_func=torch.nn.CrossEntropyLoss(),
        lr=0.002,
    )

    trainer.train(data_loader, 20)
