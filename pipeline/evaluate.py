import tensorflow as tf
from tensorflow.compat.v1 import flags
from datasets import load_metric
import numpy as np
from bleurt import score as bleurt_score
import bert_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Disable tf debugging logs
tf.get_logger().setLevel("ERROR")

# Script arguments as tf flags
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

# Metrics
bleu = load_metric("sacrebleu")
meteor = load_metric("meteor")

bleurt_ = bleurt_score.BleurtScorer("pipeline/models/bleurt")
bert_score_ = bert_score


def evaluate(pred_labels, true_labels):
    # Compute metrics all
    return {
        "Count": len(true_labels),
        "BLEU": bleu.compute(
            predictions=pred_labels,
            references=[[label] for label in true_labels])["score"],
        "METEOR": meteor.compute(
            predictions=pred_labels,
            references=true_labels)["meteor"],
        "BLEURT": np.mean(bleurt_.score(
            references=true_labels,
            candidates=pred_labels,
            batch_size=32)),
        "BERTScore F1": np.mean(bert_score_.score(
            cands=pred_labels,
            refs=true_labels,
            lang="en",
            device="cuda")[2].numpy())
    }


def evaluate_by_depth(pred_labels, true_labels, depth_labels):
    # Compute metrics for tree depth 1 vs >1
    assert len(depth_labels) == len(pred_labels) == len(true_labels)
    return (
        evaluate(pred_labels=[pred_labels[i] for i in range(len(depth_labels))
                              if depth_labels[i] == 1],
                 true_labels=[true_labels[i] for i in range(len(depth_labels))
                              if depth_labels[i] == 1]),
        evaluate(pred_labels=[pred_labels[i] for i in range(len(depth_labels))
                              if depth_labels[i] > 1],
                 true_labels=[true_labels[i] for i in range(len(depth_labels))
                              if depth_labels[i] > 1]),
    )


def print_line(char="."):
    print(char * 75)


def print_metrics(metrics):
    for metric in ["Count", "BLEU", "METEOR", "BLEURT", "BERTScore F1"]:
        print("{} : {}".format(metric, metrics[metric]))


def run_pipeline(predictions_file, references_file, depth_file=None):
    print("\n\n")
    print_line("#")
    print("{} || {}".format(references_file, predictions_file))
    print_line("#")
    print("\n\n")

    true_labels = open(references_file).read().strip().split("\n")
    pred_labels = open(predictions_file).read().strip().split("\n")
    depth_labels = None
    if depth_file:
        depth_labels = [int(d) for d in open(
            depth_file).read().strip().split("\n")]

    # Evaluate Metrics on the whole dataset
    metrics = evaluate(pred_labels=pred_labels,
                       true_labels=true_labels)
    print_line()
    print("All Metrics")
    print_metrics(metrics)
    print_line()

    # Evaluate metrics sliced by height
    metrics = evaluate_by_depth(pred_labels=pred_labels,
                                true_labels=true_labels,
                                depth_labels=depth_labels)
    print("Metrics for tree depth equals 1")
    print_metrics(metrics[0])
    print_line()
    print("Metrics for tree depth greater than 1")
    print_metrics(metrics[1])
    print_line()

    print("\n\n")
    print_line("#")
    print("\n\n")


def main():
    run_pipeline(predictions_file=FLAGS.pred,
                 references_file=FLAGS.ref,
                 depth_file=FLAGS.depth)


if __name__ == '__main__':
    main()
