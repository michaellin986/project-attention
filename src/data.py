import json
import os
from random import sample
import torch

from datasets import load_dataset


class RandomEmbeddingDataset(torch.utils.data.Dataset):
    """
    Create random dataset for testing purposes

    Data input is a tuple of row tensor (this is so that
    modules like Multihead Attention, which takes 3
    inputs, can be tested with the same data structure).

    Data output is a single row tensor.
    """

    def __init__(self, num_examples, num_tokens, d_model, num_inputs):
        self.num_examples = num_examples
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.num_inputs = num_inputs
        self.data = torch.rand(self.num_examples, self.num_tokens, self.d_model)
        self.label = torch.rand(self.num_examples, self.num_tokens, self.d_model)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        return (
            (self.data[index],) * self.num_inputs,
            self.label[index],
        )


class RandomTokenDataset(torch.utils.data.Dataset):
    """
    Create random token dataset for testing purposes

    Data input is a tuple of row tensor (this is so that
    modules like Multihead Attention, which takes 3
    inputs, can be tested with the same data structure).

    Data output is a single row tensor.
    """

    def __init__(self, num_examples, max_length):
        self.num_examples = num_examples
        self.max_length = max_length
        self.data = torch.randint(high=100, size=(self.num_examples, self.max_length))
        self.label = torch.randint(high=100, size=(self.num_examples, self.max_length))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        return (
            self.data[index],
            self.label[index],
        )


class LanguagePairDataset(torch.utils.data.Dataset):
    """
    Used to interface with wmt14 datasets saved in the
    "data/{language_pair}/" directory

    Data input is a tuple of row tensor (this is so that
    modules like Multihead Attention, which takes 3
    inputs, can be tested with the same data structure).

    Data output is a single row tensor.
    """

    def __init__(self, file_name, num_examples):
        self.num_examples = num_examples

        with open(file_name, "r", encoding="UTF-8") as file:
            data = json.loads(file.read())

        self.data = sample(data, self.num_examples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pair = tuple(self.data[index]["translation"].values())
        return (
            pair[0],
            pair[1],
        )


class TokenizedLanguagePairDataset(LanguagePairDataset):
    """
    LanguagePairDataset but tokenized and encoded (not embedded).
    Note that encoding simply refers to the mapping from token to vocab index.
    """

    def __init__(self, file_name, en_tokenizer, xx_tokenizer, num_examples, max_length):
        super().__init__(file_name, num_examples)

        languages = self.data[0]["translation"].keys()
        for i in range(len(self.data)):
            for language in languages:
                text = self.data[i]["translation"][language]

                # Tokenize and encode
                # .encode() tokenizes `text` and then encodes it
                # This is sliced to have a max length of `max_length`.
                if language == "en":
                    encoded_text = en_tokenizer.encode(text)[:max_length]
                else:
                    encoded_text = xx_tokenizer.encode(text)[:max_length]

                if len(encoded_text) != max_length:
                    # Pad with 0 at the end
                    encoded_text.extend([0] * (max_length - len(encoded_text)))

                encoded_text = torch.tensor(encoded_text)

                self.data[i]["translation"][language] = encoded_text


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
