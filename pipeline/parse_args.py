from argparse import ArgumentParser
import socket
import time


def parse_args(args):

    parser = ArgumentParser()

    parser.add_argument("--data_dir",
                        type=str,
                        default=None,
                        help="Path to the data. Each dataset has a unique expected directory structure")
    parser.add_argument("--dataset",
                        type=str,
                        default="DART",
                        help="Dataset to use. Can be DART, ToTTo, RotoWire, etc.")
    parser.add_argument("--linearize",
                        type=bool,
                        default=False,
                        help="Whether to read raw JSON/XML data and linearize to src and tgt")
    parser.add_argument("--linearize_strategy",
                        type=int,
                        default=0,
                        help="The int ID for the linearization strategy to use."
                        "Corresponds to the strategy defined in the dataset")
    parser.add_argument("--run_id",
                        type=str,
                        default="{}-{}".format(socket.gethostname(),
                                               time.strftime("%Y%m%d-%H%M%S")),
                        help="Name to associate the run with")
    parser.add_argument("--models_dir",
                        type=str,
                        default="models/",
                        help="Location to save models")
    parser.add_argument("--model",
                        type=str,
                        default="BART",
                        help="Model variant to use for training and inference")
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="Number of examples per batch")
    parser.add_argument("--use_mixed_precision",
                        type=bool,
                        default=False,
                        help="If float16 should be used for models")
    parser.add_argument("--disable_gpu",
                        type=bool,
                        default=False,
                        help="If CPU should be used for training")
    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Random seed")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=5,
                        help="Number of epochs for model training")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=5e-5,
                        help="Learning rate for optimizer")
    parser.add_argument("--warmup_steps",
                        type=int,
                        default=500,
                        help="Number of steps for a linear warm up from 0 to learning rate")
    parser.add_argument("--overwrite",
                        type=bool,
                        default=False,
                        help="Overwrite logs and models directories")
    parser.add_argument("--max_grad_norm",
                        type=float,
                        default=1.0,
                        help="Gradient norm for clipping")
    parser.add_argument("--max_permutations",
                        type=int,
                        default=-1,
                        help="Permute triples at training time. Dev and Test are not permuted")

    return parser.parse_args(args)
