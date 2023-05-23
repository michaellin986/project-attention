import torch
from src.data import RandomEmbeddingDataset
from src.trainer import Trainer

from src.decoder import DecoderLayer, Decoder


def test_decoder_layer():
    dataset = RandomEmbeddingDataset(
        num_examples=100,
        num_tokens=10,
        d_model=512,
        num_inputs=2,
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    model = DecoderLayer(input_size=512, p_dropout=0.1)
    model.initialize()

    trainer = Trainer(
        optimizer=torch.optim.Adam,
        model=model,
        loss_func=torch.nn.CrossEntropyLoss(),
        lr=0.002,
    )

    trainer.train(data_loader, 20)


def test_decoder():
    dataset = RandomEmbeddingDataset(
        num_examples=100,
        num_tokens=10,
        d_model=512,
        num_inputs=2,
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    model = Decoder(input_size=512, num_layers=6, p_dropout=0.1)
    model.initialize()

    trainer = Trainer(
        optimizer=torch.optim.Adam,
        model=model,
        loss_func=torch.nn.CrossEntropyLoss(),
        lr=0.002,
    )

    trainer.train(data_loader, 20)
