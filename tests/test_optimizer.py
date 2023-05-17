import torch
from src.data import RandomEmbeddingDataset
from src.encoder import Encoder
from src.optimizer import DynamicLRAdam
from src.trainer import Trainer


def test_dynamic_lr_adam_optimizer_on_encoder():
    dataset = RandomEmbeddingDataset(num_examples=1000, d_model=512, num_inputs=1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    model = Encoder(input_size=512, num_layers=6, p_dropout=0.1)
    model.initialize()

    trainer = Trainer(
        optimizer=DynamicLRAdam,
        model=model,
        loss_func=torch.nn.CrossEntropyLoss(),
        lr=0,
        betas=(0.9, 0.98),
        eps=10e-9,
        d_model=512,
        warmup_steps=20,
    )

    trainer.train(data_loader, 20)
