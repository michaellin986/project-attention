import torch

from src.data import RandomEmbeddingDataset
from src.trainer import Trainer
from src.multihead_attention import MultiheadAttention


def test_run_multihead_attention_unmasked():
    dataset = RandomEmbeddingDataset(num_examples=1000, d_model=512)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    model = MultiheadAttention(h=8, d_model=512, d_k=64, d_v=64, masked=False)
    model.initialize()

    trainer = Trainer(
        optimizer=torch.optim.Adam,
        model=model,
        loss_func=torch.nn.CrossEntropyLoss(),
        lr=0.002,
    )

    trainer.train(data_loader, 20)


def test_run_multihead_attention_masked():
    # TODO: implement
    pass
