import json
import os
from random import sample
import torch

from datasets import load_dataset


class RandomEmbeddingDataset(torch.utils.data.Dataset):
    """
    Create random dataset for testing purposes
    """

    def __init__(self, num_examples, d_model):
        self.num_examples = num_examples
        self.d_model = d_model
        self.data = torch.rand(self.num_examples, self.d_model)
        self.label = torch.rand(self.num_examples, self.d_model)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        return (
            self.data[index],
            self.label[index],
        )


class LanguagePairDataset(torch.utils.data.Dataset):
    """
    Used to interface with wmt14 datasets saved in the "data/{language_pair}/" directory
    """

    def __init__(self, file_name, num_examples):
        self.num_examples = num_examples

        with open(file_name, "r", encoding="UTF-8") as file:
            data = json.loads(file.read())

        self.data = sample(data, self.num_examples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return tuple(self.data[index]["translation"].values())


def download_data(language_pair="fr-en", n_train=20000, n_validation=3000, n_test=3000):
    """
    Download wmt14 data for language pair using Hugging Face's `load_dataset` API

    Training, validation, and test datasets are saved to the "data/{language_pair}/" directory
    """
    dataset = load_dataset("wmt14", language_pair, streaming=True)

    train = list(dataset["train"].take(n_train))
    validation = list(dataset["validation"].take(n_validation))
    test = list(dataset["test"].take(n_test))

    directory = f"data/{language_pair}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f"{directory}/train.json", "w", encoding="UTF-8") as train_file:
        train_obj = json.dumps(train, indent=4)
        train_file.write(train_obj)

    with open(f"{directory}/validation.json", "w", encoding="UTF-8") as validation_file:
        validation_obj = json.dumps(validation, indent=4)
        validation_file.write(validation_obj)

    with open(f"{directory}/test.json", "w", encoding="UTF-8") as test_file:
        test_obj = json.dumps(test, indent=4)
        test_file.write(test_obj)


if __name__ == "__main__":
    download_data(language_pair="fr-en", n_train=20000, n_validation=3000, n_test=3000)
    download_data(language_pair="de-en", n_train=20000, n_validation=3000, n_test=3000)
