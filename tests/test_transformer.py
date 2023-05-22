import torch
from src.data import RandomEmbeddingDataset
from src.trainer_testing_transformer import Trainer

from src.transformer import Transformer


def test_transformer():
    dataset = RandomEmbeddingDataset(num_examples=30, d_model=512, num_inputs=1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    model = Transformer(d_model=512, num_layers=6, p_dropout=0.1)
    model.initialize()

    trainer = Trainer(
        optimizer=torch.optim.Adam,
        model=model,
        loss_func=torch.nn.CrossEntropyLoss(),
        lr=0.002,
    )

    trainer.train(data_loader, 5)
