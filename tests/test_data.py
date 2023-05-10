from src.data import LanguagePairDataset


def test_language_pair_dataset():
    file_name = "tests/fixtures/test_language_pair_dataset.json"
    dataset = LanguagePairDataset(file_name, num_examples=2)
    assert len(dataset) == 2

    example = dataset[0]
    assert len(example) == 2
