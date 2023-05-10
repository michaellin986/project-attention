import torch

from src.data import RandomEmbeddingDataset
from src.trainer import Trainer
from src.multihead_attention import MultiheadAttention


def run(dataset_type, num_examples=1000, d_model=512, batch_size=100, lr=0.002):
    """
    Start training with specifications under __main__ with:

    `python -m src.runner`
    """

    dataset = dataset_type(num_examples=num_examples, d_model=d_model, num_inputs=3)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    model = MultiheadAttention(h=8, d_model=d_model, d_k=64, d_v=64, masked=False)
    model.initialize()

    trainer = Trainer(
        optimizer=torch.optim.Adam,
        model=model,
        loss_func=torch.nn.CrossEntropyLoss(),
        lr=lr,
    )

    trainer.train(data_loader, 200)


if __name__ == "__main__":
    run(RandomEmbeddingDataset)
