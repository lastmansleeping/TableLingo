from datasets import load_metric
import sys
import numpy as np


bleu = load_metric("sacrebleu")
meteor = load_metric("meteor")

bertscore = load_metric("bertscore")
bleurt = load_metric("bleurt")


def evaluate(pred_labels, true_labels):
    print("BLEU : {}".format(
        bleu.compute(predictions=pred_labels, references=[[l] for l in true_labels])["score"]))
    print("METEOR : {}".format(
        meteor.compute(predictions=pred_labels, references=true_labels)))
    print("BLEURT : {}".format(
        np.mean(bleurt.compute(predictions=pred_labels, references=true_labels)["scores"])))
    print("BERTScore : {}".format(
        bertscore.compute(predictions=pred_labels, references=true_labels, lang="en")))


def main():
    pass


if __name__ == '__main__':
    main()
