import torch
from src.data import RandomEmbeddingDataset
from src.trainer import Trainer
from src.position_wise_feed_forward import PositionWiseFeedForward


def test_position_wise_feed_forward():
    dataset = RandomEmbeddingDataset(num_examples=1000, d_model=512, num_inputs=1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    model = PositionWiseFeedForward(input_size=512, hidden_size=2048)
    model.initialize()

    trainer = Trainer(
        optimizer=torch.optim.Adam,
        model=model,
        loss_func=torch.nn.CrossEntropyLoss(),
        lr=0.002,
    )

    trainer.train(data_loader, 20)
