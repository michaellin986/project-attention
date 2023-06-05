import torch
from transformers import AutoTokenizer

from src.data import TokenizedLanguagePairDataset
from src.transformer_trainer import TransformerTrainer

from src.transformer import Transformer
from src.optimizer import DynamicLRAdam


def test_transformer():
    d_model = 512
    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    xx_tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_uncased")
    dataset = TokenizedLanguagePairDataset(
        file_name="./data/fr-en/test.json",
        en_tokenizer=en_tokenizer,
        xx_tokenizer=xx_tokenizer,
        num_examples=30,
        max_length=10,
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    model = Transformer(
        d_model=d_model,
        num_layers=6,
        en_vocab_size=100,
        max_length=16,
        p_dropout=0.1,
    )
    model.initialize()

    trainer = TransformerTrainer(
        optimizer=DynamicLRAdam,
        model=model,
        loss_func=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        en_tokenizer=en_tokenizer,
        xx_tokenizer=xx_tokenizer,
        lr=0,
        betas=(0.9, 0.98),
        eps=1e-9,
        warmup_steps=4000,
        d_model=d_model,
    )

    trainer.train(data_loader, n_epochs=5)
