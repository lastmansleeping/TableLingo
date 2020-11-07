from datasets import load_metric
import numpy as np
import sys
from bleurt import score as bleurt_score
import bert_score

# Script arguments as tf flags
from tensorflow.compat.v1 import flags
FLAGS = flags.FLAGS
flags.DEFINE_string(name="ref",
                    default=None,
                    help="True labels file")
flags.DEFINE_string("pred",
                    default=None,
                    help="Predictions file")
flags.DEFINE_string("depth",
                    default=None,
                    help="File containing depth of triple tree")


def evaluate(pred_labels, true_labels):
    bleu = load_metric("sacrebleu")
    meteor = load_metric("meteor")

    bleurt_ = bleurt_score.BleurtScorer("pipeline/models/bleurt")
    bert_score_ = bert_score

    return {
        "BLEU": bleu.compute(
            predictions=pred_labels,
            references=[[label] for label in true_labels])["score"],
        "METEOR": meteor.compute(
            predictions=pred_labels,
            references=true_labels)["meteor"],
        "BLEURT": np.mean(bleurt_.score(
            references=true_labels, candidates=pred_labels)),
        "BERTScore F1": np.mean(bert_score_.score(
            cands=pred_labels,
            refs=true_labels,
            lang="en",
            device="cuda")[2].numpy())
    }


def main(argv):
    true_labels = open(FLAGS.ref).read().strip().split("\n")
    pred_labels = open(FLAGS.pred).read().strip().split("\n")
    depth_list = None
    if FLAGS.depth:
        depth_list = [int(d) for d in open(
            FLAGS.depth).read().strip().split("\n")]

    # Evaluate Metrics on the whole dataset
    metrics_all = evaluate(pred_labels=pred_labels, true_labels=true_labels)
    for metric in sorted(metrics_all):
        print("{} : {}".format(metric, metrics_all[metric]))


if __name__ == '__main__':
    main(sys.argv[1:])
