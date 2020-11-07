import torch
import numpy as np
from data.parse_data import get_data_parser


class DataToTextDataset(torch.utils.data.Dataset):

    def __init__(self, model, special_tokens, inputs, labels, max_decode_length=128):
        self.input_tokenizer = model.get_input_tokenizer(
            special_tokens=special_tokens)
        self.label_tokenizer = model.get_label_tokenizer()

        self.input_tokens = self.input_tokenizer(
            inputs, truncation=True, padding=True)
        self.label_tokens = self.label_tokenizer(
            labels, truncation=True, padding=True, max_length=max_decode_length)

        self.true_labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.input_tokens.items()}
        item["labels"] = torch.tensor(self.label_tokens["input_ids"][idx])
        # item.update({"decoder_{}".format(key): torch.tensor(val[idx])
        #              for key, val in self.label_tokens.items()})
        return item

    def __len__(self):
        return len(self.label_tokens["input_ids"])


def load_dataset(data_dir,
                 dataset_key,
                 linearize_strategy,
                 model,
                 max_permutations=-1,
                 logger=None):

    # Define DataParser
    data_parser = get_data_parser(dataset_key,
                                  linearize_strategy,
                                  max_permutations,
                                  logger)

    # Load train, val and test data
    data_dict = data_parser.load_and_parse(data_dir)

    # Print samples
    if logger:
        logger.info("\n\nSamples:\n")
        for split_name, split in data_dict.items():
            idx = np.random.choice(range(len(split[0])))
            s = "----------------- {} -----------------\n".format(split_name)
            s += "Source : {}\n".format(split[0][idx])
            s += "Target : {}\n".format(split[1][idx])
            logger.info(s)

    # Create datasets
    dataset_dict = {k: DataToTextDataset(model=model,
                                         special_tokens=data_parser.get_special_tokens(),
                                         inputs=v[0],
                                         labels=v[1])
                    for k, v in data_dict.items()}

    return dataset_dict
