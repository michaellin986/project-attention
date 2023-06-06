import os
import torch
from transformers import AutoTokenizer
from datetime import datetime

import sys

from src.data import TokenizedLanguagePairDataset
from src.optimizer import DynamicLRAdam
from src.transformer import Transformer
from src.transformer_trainer import TransformerTrainer

D_MODEL = 512


def run(
    train,
    num_examples,
    batch_size,
    max_length,
    n_epochs,
):
    """
    "bert-base-uncased" instead of "gpt2" because bert-base-uncased `0`th token
    is '[PAD]' whereas gpt2 is '!', and [PAD] is closer to our intentions.

    Note that bert-base-uncased adds a start token `[CLS]` and end token
    `[SEP]` to the encoding.
    """
    start_time = datetime.now()

    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    xx_tokenizer = AutoTokenizer.from_pretrained(
        "flaubert/flaubert_base_uncased"
    )  # TODO: make general, i.e. not just fr

    model_args = {
        "d_model": D_MODEL,
        "num_layers": 6,
        "p_dropout": 0.1,
        "en_vocab_size": len(en_tokenizer),
        "xx_vocab_size": len(xx_tokenizer),
        "max_length": max_length,
    }

    trainer_args = {
        "loss_func": torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        "en_tokenizer": en_tokenizer,
        "xx_tokenizer": xx_tokenizer,
        "optimizer": DynamicLRAdam,
        "lr": 0,
        "betas": (0.9, 0.98),
        "eps": 1e-9,
        "d_model": D_MODEL,
        "warmup_steps": 4000,
    }

    if train:
        train_file = "./data/fr-en/test.json"  # TODO: change for actual training

        dataset = TokenizedLanguagePairDataset(
            train_file,
            en_tokenizer,
            xx_tokenizer,
            num_examples,
            max_length,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        folder_name = f"{start_time.strftime('%Y%m%d_%H%M%S')}_N{num_examples}_B{batch_size}_L{max_length}_E{n_epochs}"
        os.mkdir(f"./models/{folder_name}")

        model = Transformer(**model_args)
        model.initialize()

        trainer = TransformerTrainer(model=model, **trainer_args)

        trainer.train(data_loader, n_epochs=n_epochs, folder_name=folder_name)

    else:
        valid_file = "./data/fr-en/validation.json"
        dataset = TokenizedLanguagePairDataset(
            valid_file,
            en_tokenizer,
            xx_tokenizer,
            num_examples,
            max_length,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        model = torch.load(
            "./models/20230605_045651_N2000_B100_L50_E100/epoch_100.pt",
        )
        trainer = TransformerTrainer(model=model, **trainer_args)
        trainer.train(data_loader, n_epochs=1, train=False)


if __name__ == "__main__":
    try:
        num_examples = int(sys.argv[1])
        batch_size = int(sys.argv[2])
        max_length = int(sys.argv[3])
        n_epochs = int(sys.argv[4])
        train = int(sys.argv[5]) == 1
    except IndexError:
        raise Exception(
            """
            Command requires num_examples, batch_size, max_length, n_epochs, train as arguments:
            python -m src.transformer_runner <num_examples> <batch_size> <max_length> <n_epochs> <train (1 | 0)>

            Example: python -m src.transformer_runner 1000 200 100 2000 0
            """
        )
    run(
        train,
        num_examples,
        batch_size,
        max_length,
        n_epochs,
    )
