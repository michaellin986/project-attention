import torch


class RandomEmbeddingDataset(torch.utils.data.Dataset):
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
