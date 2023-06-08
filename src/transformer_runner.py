'''
Code inspired from lecture notebooks.
'''

import json
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
LANGUAGE_PAIRS = ["fr-en", "de-en"]
PRETRAINED_MODEL_NAME = {
    "fr-en": "flaubert/flaubert_base_uncased",
    "de-en": "dbmdz/bert-base-german-uncased",
}


def run(
    train,
    num_examples,
    batch_size,
    max_length,
    n_epochs,
    language_pair,
    loss,
):
    """
    "bert-base-uncased" instead of "gpt2" because bert-base-uncased `0`th token
    is '[PAD]' whereas gpt2 is '!', and [PAD] is closer to our intentions.

    Note that bert-base-uncased adds a start token `[CLS]` and end token
    `[SEP]` to the encoding.
    """
    start_time = datetime.now()

    if language_pair not in LANGUAGE_PAIRS:
        raise ValueError(f"Unknown language pair: {language_pair}")

    if loss == "mse":
        loss_func = torch.nn.MSELoss()
    elif loss == "cross_entropy":
        loss_func = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    xx_tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME[language_pair])

    model_args = {
        "d_model": D_MODEL,
        "num_layers": 6,
        "p_dropout": 0.1,
        "max_length": max_length,
    }

    trainer_args = {
        "loss_func": loss_func,
        "en_tokenizer": en_tokenizer,
        "xx_tokenizer": xx_tokenizer,
        "optimizer": DynamicLRAdam,
        "lr": 0,
        "betas": (0.9, 0.98),
        "eps": 1e-9,
        "d_model": D_MODEL,
        "warmup_steps": 4000,
    }

    train_file = f"./data/{language_pair}/train.json"
    train_dataset = TokenizedLanguagePairDataset(
        train_file,
        en_tokenizer,
        xx_tokenizer,
        num_examples,
        max_length,
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    validation_file = f"./data/{language_pair}/validation.json"
    validation_dataset = TokenizedLanguagePairDataset(
        validation_file,
        en_tokenizer,
        xx_tokenizer,
        num_examples,
        max_length,
    )
    validation_data_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    if train:
        folder_name = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{language_pair}_N{num_examples}_B{batch_size}_L{max_length}_E{n_epochs}"
        os.mkdir(f"./models/{folder_name}")

        model = Transformer(**model_args)
        model.initialize()

        trainer = TransformerTrainer(model=model, **trainer_args)
        (
            train_losses,
            train_bleu_scores,
            validation_losses,
            validation_bleu_scores,
        ) = trainer.train_and_validate(
            train_data_loader,
            validation_data_loader,
            n_epochs=n_epochs,
            folder_name=folder_name,
        )
        losses = {
            "train_losses": train_losses,
            "train_bleu_scores": train_bleu_scores,
            "validation_losses": validation_losses,
            "validation_bleu_scores": validation_bleu_scores,
        }

        save_path = f"./models/{folder_name}/epoch_{n_epochs}_losses.json"
        with open(save_path, "w") as f:
            json.dump(losses, f)

    else:
        model = torch.load(
            # Change this to the path of the model you want to load
            "./models/20230605_045651_fr-en_N2000_B100_L50_E100/epoch_100.pt",
        )
        trainer = TransformerTrainer(model=model, **trainer_args)
        trainer.train(validation_data_loader, n_epochs=1, train=False)


if __name__ == "__main__":
    try:
        num_examples = int(sys.argv[1])
        batch_size = int(sys.argv[2])
        max_length = int(sys.argv[3])
        n_epochs = int(sys.argv[4])
        train = int(sys.argv[5]) == 1
        language_pair = sys.argv[6]
        loss = sys.argv[7]
    except IndexError:
        raise Exception(
            """
            Command requires num_examples, batch_size, max_length, n_epochs, train as arguments:
            python -m src.transformer_runner <num_examples> <batch_size> <max_length> <n_epochs> <train (1 | 0)> <language_pair (fr-en | de-en)> <loss (mse | cross_entropy)>

            Example: python -m src.transformer_runner 1000 200 100 2000 0 fr-en mse
            """
        )
    run(
        train,
        num_examples,
        batch_size,
        max_length,
        n_epochs,
        language_pair,
        loss,
    )
