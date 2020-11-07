import glob
import os
import json
import numpy as np
from anytree import Node, RenderTree, LoopError


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
        if self.linearize_strategy == 0:
            # No permutation
            linearized_triples_list.add(" ".join(
                ["<TRIPLE> {} </TRIPLE>".format(
                    " ".join(triple)) for triple in triples]))
        elif self.linearize_strategy == 1:
            # Permutations
            if self.max_permutations > 0 and is_train:
                for i in range(self.max_permutations):
                    permutation = np.random.permutation(triples)
                    linearized_triples_list.add(" ".join(
                        ["<TRIPLE> {} </TRIPLE>".format(
                            " ".join(triple)) for triple in permutation]))
        elif self.linearize_strategy == 2:
            linearized_triples_list.add(
                self.linearize_tree(self.generate_tree(triples)).strip())

        return list(linearized_triples_list)

    def generate_tree(self, triples):
        s_nodes = {}
        o_nodes = {}
        for triple in triples:
            S, P, O = triple
            s_nodes[S] = Node(S, parent=None, predicate=None)
            o_nodes[(P, O)] = Node(O, parent=None, predicate=None)
        for triple in triples:
            S, P, O = triple
            if O in s_nodes:
                try:
                    s_nodes[O].parent = s_nodes[S]
                    s_nodes[O].predicate = P
                except LoopError:
                    o_nodes[(P, O)].parent = s_nodes[S]
                    o_nodes[(P, O)].predicate = P
            else:
                o_nodes[(P, O)].parent = s_nodes[S]
                o_nodes[(P, O)].predicate = P

        roots = [n for n in s_nodes.values() if n.is_root]
        return roots

    def linearize_tree(self, roots):
        tree_str = ""
        if not roots:
            return ""
        for root in roots:
            predicate = root.predicate
            if not root.predicate:
                predicate = "HEAD"

            tree_str += " <CHILD> {} {} <CHILDREN>{} </CHILDREN> </CHILD>".format(
                predicate, root.name, self.linearize_tree(root.children))

        return tree_str


def get_data_parser(dataset_key, linearize_strategy, max_permutations, logger):
    return {
        "DART": DARTDataParser,
        # "ToTTo": ToTToDataParser,
        # "WebNLG": WebNLGDataParser
    }[dataset_key](linearize_strategy, max_permutations, logger)
