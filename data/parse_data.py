import glob
import os
import json
from itertools import permutations
import numpy as np


class DARTDataParser:

    def __init__(self, linearize_strategy, max_permutations, logger):
        self.linearize_strategy = linearize_strategy
        self.max_permutations = max_permutations
        if logger:
            logger.info("Using linearization strategy: {}".format(
                linearize_strategy))
            logger.info(
                "Max permutations per train tripleset : {}".format(max_permutations))

    def get_special_tokens(self):
        if self.linearize_strategy == 0:
            return []
            # return [
            #     "<TRIPLE>",
            #     "</TRIPLE>"
            # ]

    def load_and_parse(self, data_dir):
        # NOTE: DART data path -> dart-<version>-full-<split>.json
        data_dict = dict()
        for split_name in ["train", "dev", "test"]:
            data_dict[split_name] = (list(), list())
            with open(glob.glob(os.path.join(data_dir, "dart-v*-full-{}.json".format(split_name)))[0], "r") as f:
                for example in json.load(f):
                    triples = example["tripleset"]
                    target_texts = [a["text"] for a in example["annotations"]]

                    linearized_triples_list = self.linearize_triples(triples,
                                                                     is_train=split_name == "train")

                    for linearized_triples in linearized_triples_list:
                        for target_text in target_texts:
                            data_dict[split_name][0].append(linearized_triples)
                            data_dict[split_name][1].append(target_text)

        return data_dict

    def linearize_triples(self, triples, is_train=True):
        linearized_triples_list = set()
        if self.max_permutations > 0 and is_train:
            # Permutations
            for i in range(self.max_permutations):
                permutation = np.random.permutation(triples)
                linearized_triples_list.add(" ".join(
                    [self.linearize_triple(triple) for triple in permutation]))
        else:
            # No permutation
            linearized_triples_list.add(" ".join(
                [self.linearize_triple(triple) for triple in triples]))

        return list(linearized_triples_list)

    def linearize_triple(self, triple):
        if self.linearize_strategy in {0, 1}:
            return "<TRIPLE> {} </TRIPLE>".format(" ".join(triple))


def get_data_parser(dataset_key, linearize_strategy, max_permutations, logger):
    return {
        "DART": DARTDataParser,
        # "ToTTo": ToTToDataParser,
        # "WebNLG": WebNLGDataParser
    }[dataset_key](linearize_strategy, max_permutations, logger)
